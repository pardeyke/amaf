"""
Generate a reference WAV file for audio quality measurement.

Structure:
  [2s chirp @ -3dBFS] [1s silence] [track1] [1s silence] [track2] ... [trackN] [1s silence]

The chirp is a linear sweep 20Hz-20kHz used for sample-accurate alignment
after the processed audio is captured from the livestream pipeline.
"""

import numpy as np
import soundfile as sf
import os

SR = 44100
CHIRP_DURATION = 2.0
SILENCE_DURATION = 1.0
CHIRP_AMPLITUDE = 10 ** (-3.0 / 20)  # -3 dBFS

SQAM_DIR = os.path.join(os.path.dirname(__file__), "SQAM_FLAC_00s9l4")

TRACKS = [69, 35, 40, 50, 53, 60, 10]


def generate_chirp(duration, sr, f0=20.0, f1=20000.0, amplitude=CHIRP_AMPLITUDE):
    """Linear chirp from f0 to f1 Hz, stereo (identical L/R)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    mono = (amplitude * np.sin(phase)).astype(np.float64)
    return np.column_stack([mono, mono])


def generate_silence(duration, sr):
    return np.zeros((int(sr * duration), 2), dtype=np.float64)


def main():
    parts = []

    # Sync chirp
    chirp = generate_chirp(CHIRP_DURATION, SR)
    parts.append(chirp)
    print(f"Chirp: {len(chirp)} samples ({CHIRP_DURATION}s)")

    # Silence after chirp
    parts.append(generate_silence(SILENCE_DURATION, SR))

    # Concatenate tracks with silence gaps
    for i, track_num in enumerate(TRACKS):
        path = os.path.join(SQAM_DIR, f"{track_num:02d}.flac")
        audio, sr = sf.read(path, dtype="float64", always_2d=True)
        assert sr == SR, f"Expected {SR}Hz, got {sr}Hz for {path}"

        # If mono, duplicate to stereo
        if audio.shape[1] == 1:
            audio = np.column_stack([audio[:, 0], audio[:, 0]])

        parts.append(audio)
        print(f"Track {track_num:02d}: {len(audio)} samples ({len(audio)/SR:.1f}s)")

        # Silence between tracks (and after last track)
        parts.append(generate_silence(SILENCE_DURATION, SR))

    reference = np.concatenate(parts, axis=0)
    total_duration = len(reference) / SR

    out_path = os.path.join(os.path.dirname(__file__), "reference.wav")
    sf.write(out_path, reference, SR, subtype="PCM_24")
    print(f"\nWrote {out_path}")
    print(f"Total: {len(reference)} samples ({total_duration:.1f}s)")


if __name__ == "__main__":
    main()
