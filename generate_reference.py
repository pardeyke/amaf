"""
Generate a reference WAV file for audio quality measurement.

Structure:
  [2s chirp @ -3dBFS] [1s silence] [track1] [1s silence] [track2] ... [trackN] [1s silence]

The chirp is a linear sweep 20Hz-20kHz used for sample-accurate alignment
after the processed audio is captured from the livestream pipeline.

Can be used standalone (CLI) or imported as a library by the web UI.
"""

import json
import subprocess
import numpy as np
import soundfile as sf
import os

SR = 44100
CHIRP_DURATION = 2.0
SILENCE_DURATION = 1.0
CHIRP_AMPLITUDE = 10 ** (-3.0 / 20)  # -3 dBFS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQAM_DIR = os.path.join(BASE_DIR, "SQAM_FLAC_00s9l4")

DEFAULT_TRACKS = [69, 35, 40, 50, 53, 60, 10]


def generate_chirp(duration, sr, f0=20.0, f1=20000.0, amplitude=CHIRP_AMPLITUDE):
    """Linear chirp from f0 to f1 Hz, stereo (identical L/R)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    mono = (amplitude * np.sin(phase)).astype(np.float64)
    return np.column_stack([mono, mono])


def generate_silence(duration, sr):
    return np.zeros((int(sr * duration), 2), dtype=np.float64)


def get_sqam_tracks():
    """Return metadata for all available SQAM tracks."""
    tracks = []
    if not os.path.isdir(SQAM_DIR):
        return tracks
    for fname in sorted(os.listdir(SQAM_DIR)):
        if not fname.endswith(".flac"):
            continue
        num = int(fname.replace(".flac", ""))
        path = os.path.join(SQAM_DIR, fname)
        # Get title and duration via ffprobe
        title = f"Track {num:02d}"
        duration = 0.0
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_entries", "format=duration:format_tags=title", path],
                capture_output=True, text=True, timeout=5,
            )
            info = json.loads(result.stdout)
            fmt = info.get("format", {})
            title = fmt.get("tags", {}).get("title", title)
            duration = float(fmt.get("duration", 0))
        except Exception:
            pass
        tracks.append({
            "num": num,
            "title": title,
            "duration": duration,
            "filename": fname,
        })
    return tracks


def build_reference(track_nums, output_path=None):
    """Build a reference WAV from the given SQAM track numbers.

    Returns (output_path, total_duration_seconds).
    """
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "reference.wav")

    parts = []

    # Sync chirp
    chirp = generate_chirp(CHIRP_DURATION, SR)
    parts.append(chirp)

    # Silence after chirp
    parts.append(generate_silence(SILENCE_DURATION, SR))

    # Concatenate tracks with silence gaps
    for track_num in track_nums:
        path = os.path.join(SQAM_DIR, f"{track_num:02d}.flac")
        audio, sr = sf.read(path, dtype="float64", always_2d=True)
        assert sr == SR, f"Expected {SR}Hz, got {sr}Hz for {path}"

        # If mono, duplicate to stereo
        if audio.shape[1] == 1:
            audio = np.column_stack([audio[:, 0], audio[:, 0]])

        parts.append(audio)

        # Silence between tracks (and after last track)
        parts.append(generate_silence(SILENCE_DURATION, SR))

    reference = np.concatenate(parts, axis=0)
    total_duration = len(reference) / SR

    sf.write(output_path, reference, SR, subtype="PCM_24")
    return output_path, total_duration


def build_video_reference(track_nums, resolution="1920x1080", fps=30,
                          output_path=None):
    """Build a video reference with luminance chirp sync + test pattern + audio.

    Returns (output_path, total_duration).
    """
    from video import build_reference_video
    return build_reference_video(track_nums, resolution=resolution, fps=fps,
                                 output_path=output_path)


def main():
    print("Building reference from default tracks:", DEFAULT_TRACKS)
    path, duration = build_reference(DEFAULT_TRACKS)
    print(f"Wrote {path} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
