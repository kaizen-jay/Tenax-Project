"""
Text-to-speech with a fallback chain, because a hackathon demo can't afford
a single TTS backend as the one point of failure:

  1. Piper (if a voice model is present in models_dir) — best quality,
     proper multilingual support, but needs models downloaded separately.
  2. macOS `say` command — zero setup, ships with every Mac, used
     automatically if Piper isn't set up. Not available on Windows/Linux.
  3. Silent WAV of an estimated duration — guarantees the lesson can always
     proceed with captions even if no TTS backend works on this machine.

Callers never need to know which backend actually ran; synthesize() always
returns a valid wav path.
"""
from __future__ import annotations

import os
import platform
import subprocess
import wave
import struct


VOICE_MAP_PIPER = {
    "english": "en_US-lessac-medium.onnx",
    "hindi": "hi_IN-pratham-medium.onnx",
    "hinglish": "hi_IN-pratham-medium.onnx",
}

# Common built-in macOS voice names. If a name isn't installed on a given
# Mac, we catch the failure and fall back further rather than erroring out.
VOICE_MAP_MACOS_SAY = {
    "english": "Samantha",
    "hindi": "Lekha",
    "hinglish": "Lekha",
}

WORDS_PER_SECOND = 2.3  # rough average speaking rate, used only for the silent fallback's duration


class TTSError(RuntimeError):
    pass


def _try_piper(text: str, language: str, models_dir: str, out_wav_path: str) -> bool:
    voice_file = VOICE_MAP_PIPER.get(language.lower())
    voice_path = os.path.join(models_dir, voice_file) if voice_file else None
    if not voice_path or not os.path.exists(voice_path):
        return False
    try:
        proc = subprocess.run(
            ["piper", "--model", voice_path, "--output_file", out_wav_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=120,
        )
        return proc.returncode == 0 and os.path.exists(out_wav_path)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _try_macos_say(text: str, language: str, out_wav_path: str) -> bool:
    if platform.system() != "Darwin":
        return False
    voice = VOICE_MAP_MACOS_SAY.get(language.lower(), "Samantha")
    # `say` can emit a real WAV directly via --data-format, no conversion step needed.
    for v in (voice, None):  # try the language-specific voice, then fall back to system default voice
        cmd = ["say", "-o", out_wav_path, "--data-format=LEI16@22050"]
        if v:
            cmd += ["-v", v]
        cmd.append(text)
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=120)
            if proc.returncode == 0 and os.path.exists(out_wav_path):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return False


def _write_silent_wav(text: str, out_wav_path: str, rate: int = 22050) -> None:
    duration = max(1.5, len(text.split()) / WORDS_PER_SECOND)
    n_frames = int(duration * rate)
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    with wave.open(out_wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % n_frames, *([0] * n_frames)))


def synthesize(text: str, language: str, models_dir: str, out_wav_path: str) -> str:
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)

    if os.path.isdir(models_dir) and _try_piper(text, language, models_dir, out_wav_path):
        return out_wav_path

    if _try_macos_say(text, language, out_wav_path):
        return out_wav_path

    # Last resort: silent audio so the video still generates and the lesson
    # can continue with on-screen captions, instead of the whole beat failing.
    _write_silent_wav(text, out_wav_path)
    return out_wav_path


def wav_duration_seconds(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)


def estimate_duration_seconds(text: str, words_per_minute: int = 150) -> float:
    """Fallback duration estimate when no audio is available (Piper missing
    or failed) — lets the avatar animation still run for a plausible length
    instead of the lesson just not showing anything for this beat."""
    word_count = max(1, len(text.split()))
    return max(2.0, (word_count / words_per_minute) * 60.0)


def synthetic_envelope(duration_seconds: float, fps: int = 12) -> list:
    """A plausible-looking mouth-movement pattern for when there's no real
    audio to drive the avatar from. Not synced to actual speech, but reads
    as "talking" well enough for a demo — much better than a frozen face."""
    import math
    n_frames = max(1, int(duration_seconds * fps))
    envelope = []
    for i in range(n_frames):
        t = i / fps
        # overlapping sine waves at speech-like rates, clipped to look like
        # natural pauses between words rather than a smooth pure tone
        val = 0.5 + 0.5 * math.sin(t * 9.0) * math.sin(t * 2.3)
        envelope.append(max(0.0, min(1.0, val)))
    return envelope


def get_amplitude_envelope(wav_path: str, fps: int = 24):
    """Coarse amplitude envelope, one value per video frame, used to drive
    the avatar's mouth-open/close animation."""
    import numpy as np
    with wave.open(wav_path, "rb") as w:
        n_frames = w.getnframes()
        rate = w.getframerate()
        raw = w.readframes(n_frames)
        sampwidth = w.getsampwidth()
        n_channels = w.getnchannels()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    samples_per_frame = max(1, int(rate / fps))
    n_video_frames = max(1, len(audio) // samples_per_frame)
    peak = max(1.0, float(np.abs(audio).max())) if len(audio) else 1.0
    envelope = []
    for i in range(n_video_frames):
        chunk = audio[i * samples_per_frame:(i + 1) * samples_per_frame]
        envelope.append(float(np.abs(chunk).mean()) / peak if len(chunk) else 0.0)
    return envelope
