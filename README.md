# AMAF — Audio Multi-Method Assessment Fusion

Measure audio quality loss across livestreaming pipelines by comparing a known reference signal against the captured output. AMAF combines multiple measurement methods into a single automated workflow: chirp-based sample-accurate alignment, spectral analysis, perceptual quality scoring, and PDF/web reporting.

Built for broadcast engineers who need to quantify degradation across chains like:

```
Playout → Mixer → SDI Embedder → Encoder → CDN → YouTube → Capture
```

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

# ffmpeg is needed for non-WAV input formats (M4A, MP3, etc.)
brew install ffmpeg  # macOS
```

Then start AMAF:

```bash
amaf
```

Open http://localhost:5000 — that's it. Everything happens in the web UI.

## How it works

### 1. Build a reference

Go to **Build Reference** in the web UI. Browse all 70 EBU SQAM tracks organised by category (strings, brass, percussion, speech, classical, pop), click to select in the order you want, and hit **Build**. A sync chirp is automatically prepended for alignment.

A pre-built `reference.wav` is included and ready to use out of the box — it covers pop, speech, classical, and transient material (~6 min).

> The SQAM source files are bundled in the repo under `SQAM_FLAC_00s9l4/`. No separate download needed.

### 2. Play through your pipeline

Play `reference.wav` from your playout system through the full signal chain. Download the reference file from the web UI or use it directly from the repo.

### 3. Upload the capture

Record or download the audio from the end of your pipeline (YouTube stream download, loopback recording, file from CDN). Back in the AMAF web UI, upload it with an optional label (e.g. "YouTube 1080p AAC"). Any format ffmpeg can decode is supported (WAV, M4A, MP3, FLAC, Opus, etc.).

### 4. Get results

AMAF automatically aligns the capture to the reference using the sync chirp and runs all measurements. Results appear in seconds with interactive zoomable diagrams, color-coded metrics, and plain-English explanations of every chart and number.

Download a PDF report for management with one click.

## Metrics

| Metric | What it tells you |
|--------|-------------------|
| **SNR** | Overall signal-to-noise ratio of the degradation |
| **Segmental SNR** | Per-second SNR — reveals where the codec struggles most |
| **Null depth** | RMS level of the difference signal relative to the original |
| **THD+N** | Total error power as a percentage of the output |
| **Spectral difference** | Per-band frequency response deviation (low/mid/high) |
| **PESQ MOS-LQO** | Perceptual quality score (1–4.64 scale) |
| **PEAQ ODG** | Objective Difference Grade (-4 to 0, 0 = transparent) |
| **ViSQOL MOS** | Perceptual similarity score |

## Web dashboard

- **Reference builder** — browse SQAM tracks by category, select in order, see running duration, build with one click
- **Upload and auto-analysis** with background processing
- **Color-coded metrics** (green/amber/red) at a glance
- **Plain-English explanations** for every diagram and metric
- **Full-screen zoomable diagrams** with keyboard navigation (arrow keys, Escape)
- **Collapsible plot sections** — waveform, null test, spectral difference, spectrograms, segmental SNR, PESQ per chunk, magnitude spectrum
- **PDF report download** per measurement
- **Persistent results** in SQLite

## CLI usage

For scripting or headless use, the measurement pipeline also works from the command line:

```bash
python measure.py captured_audio.wav
```

Outputs metrics to stdout and generates a PDF report.

## How the sync chirp works

The reference file starts with a 2-second linear sweep from 20 Hz to 20 kHz at -3 dBFS. This signal:

- Survives multiple lossy AAC/Opus re-encodings
- Produces a sharp, unambiguous peak when cross-correlated with the captured audio
- Enables sample-accurate alignment even through 20+ seconds of pipeline latency
- Is robust to level changes and mild EQ applied by the pipeline

## Project structure

```
amaf/
├── reference.wav             # Pre-built reference file, ready to use
├── generate_reference.py     # Reference builder (library for web UI + CLI fallback)
├── measure.py                # Measurement pipeline (CLI + library)
├── report.py                 # PDF report and web plot generation
├── web.py                    # Flask web GUI (entry point for `amaf` command)
├── setup.py                  # Package setup with `amaf` console script
├── requirements.txt
└── SQAM_FLAC_00s9l4/         # EBU SQAM source files (70 tracks)
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
