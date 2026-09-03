"""
A deliberately simple, fully-local 2D avatar: a friendly illustrated face
whose mouth opens/closes in sync with the audio's amplitude envelope.

Why not a photorealistic talking-head model (Wav2Lip/SadTalker/etc.)? Those
need a GPU to render at usable speed; on a CPU-only M-series Mac they'd be
far too slow to iterate on in a 2-3 day hackathon, and a slow/broken demo
scores worse than a simple one that reliably works. The rubric weights
"Human-Like Teaching and Adaptation" (20 pts, the orchestrator/RAG logic)
far above "Voice and AI Avatar" (10 pts) — so this trades avatar realism
for guaranteed reliability, and puts the saved time into the teaching logic.

If you get this working end-to-end with time to spare, swapping this
module for Wav2Lip (given a single reference photo + the generated wav) is
a drop-in upgrade — render_avatar_frames() is the only function that would
need replacing; everything downstream (video.py) just consumes a folder of
frames.
"""
from __future__ import annotations

import os
import math
from PIL import Image, ImageDraw

FACE_COLOR = (255, 224, 189)
OUTLINE = (60, 40, 30)


def draw_avatar_frame(size: int, mouth_open: float, blink: bool) -> Image.Image:
    """Returns a single RGBA avatar frame as an in-memory PIL Image (no disk
    I/O) — used directly by video.py's frame loop for speed and reliability.
    """
    r = size // 3
    cx, cy = size // 2, size // 2
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _draw_face(draw, cx, cy, r, mouth_open, blink)
    return img


def _draw_face(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, mouth_open: float, blink: bool):
    # head
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FACE_COLOR, outline=OUTLINE, width=4)
    # eyes
    eye_y = cy - r // 6
    eye_dx = r // 3
    eye_r = r // 10 if not blink else max(1, r // 40)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.ellipse([ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r], fill=(30, 30, 30))
    # eyebrows
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.line([ex - eye_r * 1.5, eye_y - eye_r * 3, ex + eye_r * 1.5, eye_y - eye_r * 3.5],
                  fill=OUTLINE, width=3)
    # mouth: ellipse height driven by amplitude
    mouth_y = cy + r // 2
    mouth_w = r // 2
    mouth_h = max(2, int(r // 6 * mouth_open))
    draw.ellipse([cx - mouth_w, mouth_y - mouth_h, cx + mouth_w, mouth_y + mouth_h],
                 fill=(150, 60, 60), outline=OUTLINE, width=3)


def render_avatar_frames(envelope: list[float], out_dir: str, size: int = 400, fps: int = 24) -> str:
    """File-based variant, kept for standalone testing/inspection. video.py
    uses draw_avatar_frame() directly instead (in-memory, no disk I/O, much
    faster for a full lesson's worth of frames)."""
    os.makedirs(out_dir, exist_ok=True)
    for i, amp in enumerate(envelope):
        blink = (i % int(fps * 3)) < 3
        mouth_open = 0.15 + min(1.0, amp * 3.0) * 0.85
        img = draw_avatar_frame(size, mouth_open, blink)
        img.save(os.path.join(out_dir, f"frame_{i:05d}.png"))
    return out_dir
