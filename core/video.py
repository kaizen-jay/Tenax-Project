"""
Video composition, rebuilt to remove the fragile dependency chain that kept
breaking across machines (MoviePy -> ImageMagick -> system font
resolution). Every frame is drawn directly with PIL (slide + avatar +
captions, all just image compositing — no external binary involved), then
encoded with imageio-ffmpeg, which ships its own ffmpeg binary inside the
pip package. Nothing here requires `brew install` anything.

Failure handling: every external call in here (TTS, ffmpeg encode) is
wrapped so a failure degrades the OUTPUT (e.g. silent audio, or a
text-only fallback one level up in app.py) rather than crashing the app.
"""
from __future__ import annotations

import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

from .lesson_planner import LessonBeat
from . import visuals, tts, avatar

VIDEO_SIZE = (1280, 720)
FPS = 15  # 15 is plenty for a mostly-static slide + small avatar; keeps encode time and frame count down

import matplotlib.font_manager as _fm
_CAPTION_FONT_PATH = _fm.findfont(_fm.FontProperties(family="DejaVu Sans", weight="bold"))
_CAPTION_FONT = ImageFont.truetype(_CAPTION_FONT_PATH, 26)


def build_narration_text(beat: LessonBeat) -> str:
    parts = [beat.title + "."]
    parts.extend(beat.explanation_points)
    if beat.example:
        parts.append(f"For example: {beat.example}")
    if beat.check_in_question:
        parts.append(beat.check_in_question)
    return " ".join(parts)


def _prepare_background(slide_path: str) -> Image.Image:
    """Load the matplotlib/pygments-rendered slide and fit it to the video
    canvas, letterboxed on a white background if aspect ratios differ."""
    slide = Image.open(slide_path).convert("RGB")
    canvas = Image.new("RGB", VIDEO_SIZE, (255, 255, 255))
    slide_ratio = slide.width / slide.height
    canvas_ratio = VIDEO_SIZE[0] / VIDEO_SIZE[1]
    if slide_ratio > canvas_ratio:
        new_w = VIDEO_SIZE[0]
        new_h = int(new_w / slide_ratio)
    else:
        new_h = VIDEO_SIZE[1]
        new_w = int(new_h * slide_ratio)
    slide = slide.resize((new_w, new_h), Image.LANCZOS)
    canvas.paste(slide, ((VIDEO_SIZE[0] - new_w) // 2, (VIDEO_SIZE[1] - new_h) // 2))
    return canvas


def _draw_caption(frame: Image.Image, text: str) -> None:
    """Draws a semi-transparent caption bar directly onto the frame with
    PIL — no ImageMagick/TextClip involved."""
    lines = textwrap.wrap(text, 70)[:4]
    if not lines:
        return
    draw = ImageDraw.Draw(frame, "RGBA")
    line_height = 32
    bar_height = line_height * len(lines) + 24
    bar_top = VIDEO_SIZE[1] - bar_height
    draw.rectangle([0, bar_top, VIDEO_SIZE[0], VIDEO_SIZE[1]], fill=(0, 0, 0, 160))
    y = bar_top + 12
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=_CAPTION_FONT)
        text_w = bbox[2] - bbox[0]
        x = (VIDEO_SIZE[0] - text_w) // 2
        draw.text((x, y), line, font=_CAPTION_FONT, fill=(255, 255, 255, 255))
        y += line_height


def render_beat_video(
    beat: LessonBeat,
    language: str,
    work_dir: str,
    models_dir: str,
    fps: int = FPS,
) -> str:
    beat_dir = os.path.join(work_dir, f"beat_{beat.beat_id}")
    os.makedirs(beat_dir, exist_ok=True)

    # 1. narration audio (Piper -> macOS say -> silent, handled inside tts.py)
    narration = build_narration_text(beat)
    wav_path = os.path.join(beat_dir, "narration.wav")
    tts.synthesize(narration, language, models_dir, wav_path)
    duration = tts.wav_duration_seconds(wav_path)

    # 2. avatar amplitude envelope, one value per frame
    envelope = tts.get_amplitude_envelope(wav_path, fps=fps)
    if not envelope:
        envelope = [0.2] * max(1, int(duration * fps))

    # 3. background slide, prepared once (not per-frame — the slide itself
    #    doesn't change during a beat, only the avatar does)
    slide_path = visuals.render_visual(beat, beat_dir)
    background = _prepare_background(slide_path)

    caption_text = narration
    avatar_size = int(VIDEO_SIZE[1] * 0.45)
    avatar_pos = (VIDEO_SIZE[0] - avatar_size - 20, VIDEO_SIZE[1] - avatar_size - 20)

    # 4. compose + encode frames directly with imageio-ffmpeg (bundled
    #    binary, no system ffmpeg/ImageMagick install required)
    silent_video_path = os.path.join(beat_dir, "silent.mp4")
    writer = imageio_ffmpeg.write_frames(
        silent_video_path, VIDEO_SIZE, fps=fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
    )
    writer.send(None)  # prime the generator-based writer
    try:
        for i, amp in enumerate(envelope):
            frame = background.copy()
            blink = (i % int(fps * 3)) < 2
            mouth_open = 0.15 + min(1.0, amp * 3.0) * 0.85
            avatar_frame = avatar.draw_avatar_frame(avatar_size, mouth_open, blink)
            frame.paste(avatar_frame, avatar_pos, avatar_frame)  # alpha-composited paste
            _draw_caption(frame, caption_text)
            writer.send(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        writer.close()

    # 5. mux narration audio into the silent video
    out_path = os.path.join(beat_dir, "beat.mp4")
    _mux_audio(silent_video_path, wav_path, out_path)
    return out_path


def _mux_audio(silent_video_path: str, wav_path: str, out_path: str) -> None:
    """Combines the (silent) encoded video with the narration audio track
    using the ffmpeg binary bundled inside imageio-ffmpeg — no system
    ffmpeg install required. Falls back to the silent video if muxing
    fails for any reason, so a broken audio step never blocks the demo."""
    import subprocess
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", silent_video_path,
        "-i", wav_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(proc.stderr.decode(errors="ignore"))
    except Exception:
        # audio muxing failed for some reason -> ship the silent video
        # rather than failing the whole beat; captions still convey the content
        import shutil
        shutil.copy(silent_video_path, out_path)


def concatenate_beats(beat_video_paths: list[str], out_path: str) -> str:
    """Concatenates already-rendered beat videos into one file (used for a
    final exportable lesson recording, not during live teaching)."""
    import subprocess
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    list_file = out_path + ".txt"
    with open(list_file, "w") as f:
        for p in beat_video_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_path]
    subprocess.run(cmd, capture_output=True, timeout=300)
    os.remove(list_file)
    return out_path
