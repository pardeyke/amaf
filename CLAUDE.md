# AMAF — Audio Multi-Method Assessment Fusion

Audio/video quality measurement tool for broadcast engineers. Compares a known reference signal against a captured output to quantify degradation across audio pipelines (codecs, streaming, DACs, etc.).

## Architecture

Single-directory Python project — no packages, no src layout. All modules live at the root:

| File | Role | Lines |
|---|---|---|
| `web.py` | Flask web GUI — entry point (`amaf` CLI), routes, HTML templates (inline `render_template_string`), SQLite DB, background processing threads | ~1810 |
| `measure.py` | Measurement pipeline — chirp-based alignment, spectral analysis, SNR/THD+N, PESQ, POLQA, PEAQ, ViSQOL. Two entry points: `run_measurement()` (chirp mode) and `run_comparison()` (arbitrary ref vs processed) | ~820 |
| `report.py` | PDF report generation (matplotlib PdfPages) and web plot PNG generation | ~708 |
| `generate_reference.py` | Builds reference WAV files from EBU SQAM tracks with prepended sync chirp | ~134 |
| `video.py` | Video analysis — detection, audio extraction, luminance-chirp alignment, VMAF/PSNR/SSIM via ffmpeg | ~558 |
| `setup.py` | Package setup — `pip install -e .` creates `amaf` console script pointing to `web:main` |

## Key design decisions

- **All HTML is inline** in `web.py` via `render_template_string` — three large template strings: `PAGE_INDEX`, `PAGE_REFERENCE`, `PAGE_RESULT`. No separate templates directory.
- **SQLite** database at `data/amaf.db` — single `measurements` table with columns for all metric types. Schema migrations are handled inline in `init_db()` via `ALTER TABLE ADD COLUMN` with exception swallowing.
- **Background processing** via daemon `threading.Thread` — measurements run async after upload, status polled from the UI.
- **File storage** in `data/` — each measurement gets `{uuid}.{ext}` (uploaded file), `{uuid}.pdf` (report), `{uuid}_plots/` (PNGs), `{uuid}_audio/` (aligned WAV files for A/B playback).
- **Alignment** uses a 2s linear chirp sweep (20Hz–20kHz @ -3dBFS) prepended to the reference. Cross-correlation finds the chirp in the captured audio for sample-accurate sync.
- **Two measurement modes**: "chirp" (uses built-in reference.wav with sync chirp) and "compare" (user supplies both reference and processed files, alignment via full-signal cross-correlation).

## Running

```bash
pip install -e .
amaf                    # web UI on http://localhost:5000
python measure.py <file>  # CLI mode
```

Requires `ffmpeg` for non-WAV formats and video analysis.

## Dependencies

Core: numpy, scipy, soundfile, pesq, matplotlib, flask

Optional (detected at runtime, skipped if absent):
- `polqa` — POLQA (ITU-T P.863) via CLI binary (commercial, from OPTICOM/Rohde & Schwarz)
- `gstpeaq` — PEAQ (ITU-R BS.1387) via GStreamer plugin
- `visqol` — ViSQOL perceptual quality
- `ffmpeg`/`ffprobe` — media decoding, video metrics (VMAF/PSNR/SSIM)

## Data / assets

- `SQAM_FLAC_00s9l4/` — 70 EBU SQAM tracks (FLAC), bundled in repo, used to build reference files
- `reference.wav` — pre-built reference file tracked via git-lfs
- `data/` — runtime output directory (gitignored), contains SQLite DB and measurement artifacts

## Conventions

- Sample rate: 44100 Hz throughout
- Audio always loaded as `float64`, `always_2d=True` (even mono is 2D)
- Chirp parameters must match between `generate_reference.py` and `measure.py` (duplicated constants: `SR`, `CHIRP_DURATION`, `CHIRP_F0`, `CHIRP_F1`, `CHIRP_AMPLITUDE`)
- No test suite
- No type annotations
- No linter config
