# AMAF — Audio Multi-Method Assessment Fusion

Measure audio quality loss across livestreaming pipelines by comparing a known reference signal against the captured output. AMAF combines multiple measurement methods into a single automated workflow: chirp-based sample-accurate alignment, spectral analysis, perceptual quality scoring, and PDF/web reporting.

Built for broadcast engineers who need to quantify degradation across chains like:

```
Playout → Mixer → SDI Embedder → Encoder → CDN → YouTube → Capture
```

## What it does

1. **Generate a reference file** — concatenates EBU SQAM test material with a sync chirp for alignment
2. **Align automatically** — cross-correlates the chirp to find sample-accurate offset, even after 20+ seconds of pipeline latency
3. **Measure with multiple methods:**
   - **Spectral difference / null test** — per-band (low/mid/high) magnitude comparison
   - **SNR / THD+N** — overall and segmental (1s windows)
   - **PESQ** (ITU-T P.862) — perceptual speech quality, wideband MOS-LQO
   - **PEAQ** (ITU-R BS.1387) — perceptual audio quality (requires [gstpeaq](https://github.com/HSU-ANT/gstpeaq))
   - **ViSQOL** — perceptual similarity (requires [visqol](https://github.com/google/visqol))
4. **Report** — generates a PDF with spectrograms, spectral plots, and a metrics summary
5. **Web GUI** — upload captures, browse results, view interactive zoomable diagrams

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Also need ffmpeg for non-WAV input formats
brew install ffmpeg  # macOS
```

### 1. Get test audio

Download the [EBU SQAM](https://qc.ebu.io/testmaterial/523/) FLAC files and place them in `SQAM_FLAC_00s9l4/`:

```
SQAM_FLAC_00s9l4/
├── 01.flac
├── 02.flac
├── ...
└── 70.flac
```

### 2. Generate the reference file

```bash
python generate_reference.py
```

This creates `reference.wav` — a 24-bit 44.1kHz stereo WAV containing a sync chirp followed by selected SQAM tracks with silence gaps. Play this file through your pipeline.

### 3. Capture the output

Record or download the audio from the end of your pipeline (e.g., YouTube stream download, loopback recording). Any format ffmpeg can decode is supported.

### 4. Measure

**Command line:**

```bash
python measure.py captured_audio.wav
```

Accepts WAV, M4A, MP3, FLAC, or any format ffmpeg supports. Outputs metrics to stdout and generates a PDF report.

**Web GUI:**

```bash
python web.py
# Open http://localhost:5000
```

Upload captured audio files through the browser. Results are stored in a SQLite database and include interactive zoomable diagrams.

## Output

### Metrics

| Metric | What it tells you |
|--------|-------------------|
| **SNR** | Overall signal-to-noise ratio of the degradation |
| **Segmental SNR** | Per-second SNR — reveals where the codec struggles |
| **Null depth** | RMS level of the difference signal relative to the original |
| **THD+N** | Total error power as a percentage of the output |
| **Spectral difference** | Per-band frequency response deviation (low/mid/high) |
| **PESQ MOS-LQO** | Perceptual quality score (1–4.64 scale) |
| **PEAQ ODG** | Objective Difference Grade (-4 to 0, 0 = transparent) |
| **ViSQOL MOS** | Perceptual similarity score |

### PDF report

Three-page report with spectrograms, spectral difference plots, segmental SNR over time, PESQ per chunk, and a full metrics table. Ready to hand to management.

### Web dashboard

Dark-themed dashboard with:
- Upload and auto-analysis
- Color-coded metrics overview (green/amber/red)
- Full-screen zoomable diagrams with keyboard navigation
- Collapsible plot sections
- PDF download per measurement
- Persistent results in SQLite

## How the sync chirp works

The reference file starts with a 2-second linear sweep from 20 Hz to 20 kHz at -3 dBFS. This signal:

- Survives multiple lossy AAC/Opus re-encodings
- Produces a sharp, unambiguous peak when cross-correlated with the captured audio
- Enables sample-accurate alignment even through 20+ seconds of pipeline latency
- Is robust to level changes and mild EQ applied by the pipeline

## Project structure

```
amaf/
├── generate_reference.py   # Build the reference WAV from SQAM source material
├── measure.py              # CLI measurement pipeline + library API
├── report.py               # PDF report and web plot generation
├── web.py                  # Flask web GUI
├── requirements.txt
└── SQAM_FLAC_00s9l4/       # EBU SQAM source files (not included, download separately)
```

## Optional: PEAQ and ViSQOL

PESQ, spectral analysis, and SNR/THD+N work out of the box. For the additional perceptual metrics:

**PEAQ** (ITU-R BS.1387) — requires GStreamer + gstpeaq plugin:

```bash
brew install gstreamer gst-plugins-base autoconf automake libtool
git clone https://github.com/HSU-ANT/gstpeaq.git
cd gstpeaq && ./autogen.sh && ./configure && make && make install
```

**ViSQOL** — requires building from source (has known issues on Apple Silicon):

```bash
git clone https://github.com/google/visqol.git
cd visqol && bazel build :visqol -c opt
```

Both are detected automatically at runtime. If not found, they are skipped gracefully.

## License

MIT
