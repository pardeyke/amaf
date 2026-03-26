# AMAF — Audio Multi-Method Assessment Fusion

Measure audio quality loss across livestreaming pipelines by comparing a known reference signal against the captured output. AMAF combines multiple measurement methods into a single automated workflow: chirp-based sample-accurate alignment, spectral analysis, perceptual quality scoring, and PDF/web reporting.

Built for broadcast engineers who need to quantify degradation across chains like:

```
Playout → Mixer → SDI Embedder → Encoder → CDN → YouTube → Capture
```

## What it does

1. **Build a reference file** — pick tracks from the EBU SQAM library via the web UI or CLI; a sync chirp is prepended automatically
2. **Play it through your pipeline** — the included `reference.wav` is ready to use out of the box
3. **Capture the output** — download from YouTube, loopback record, or grab the file from your CDN
4. **Measure** — upload the capture and get results in seconds:
   - **Spectral difference / null test** — per-band (low/mid/high) magnitude comparison
   - **SNR / THD+N** — overall and segmental (1s windows)
   - **PESQ** (ITU-T P.862) — perceptual speech quality, wideband MOS-LQO
   - **PEAQ** (ITU-R BS.1387) — perceptual audio quality (requires [gstpeaq](https://github.com/HSU-ANT/gstpeaq))
   - **ViSQOL** — perceptual similarity (requires [visqol](https://github.com/google/visqol))
5. **Report** — PDF with spectrograms and metrics, or interactive web diagrams with zoom

## Quick start

```bash
pip install -r requirements.txt

# ffmpeg is needed for non-WAV input formats (M4A, MP3, etc.)
brew install ffmpeg  # macOS
```

### 1. Get test audio

Download the [EBU SQAM](https://qc.ebu.io/testmaterial/523/) FLAC files and place them in `SQAM_FLAC_00s9l4/`. This is only needed if you want to build a custom reference track — a default `reference.wav` is included in the repo.

### 2. Build a reference (optional)

A pre-built `reference.wav` (24-bit, 44.1 kHz stereo, ~6 min) is included and ready to use. It contains a sync chirp followed by a selection of SQAM tracks covering pop, speech, classical, and transient material.

To build a custom reference with different tracks:

**Web UI** (recommended):

```bash
python web.py
# Open http://localhost:5000 → click "Build Reference"
```

Browse all 70 SQAM tracks organised by category (strings, brass, percussion, speech, etc.), click to select in order, see the running duration, and generate with one click.

**CLI:**

```bash
python generate_reference.py
```

Uses a default track selection. Edit `DEFAULT_TRACKS` in the script to customise.

### 3. Play and capture

Play `reference.wav` through your pipeline. Record or download the audio from the output (YouTube stream download, loopback recording, file from CDN). Any format ffmpeg can decode is supported.

### 4. Measure

**Web GUI:**

```bash
python web.py
# Open http://localhost:5000
```

Upload the captured audio, add an optional label (e.g. "YouTube 1080p AAC"), and results appear automatically with interactive zoomable diagrams.

**Command line:**

```bash
python measure.py captured_audio.wav
```

Outputs metrics to stdout and generates a PDF report.

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

- **Reference builder** — browse and select SQAM tracks by category, see duration, build with one click
- **Upload and auto-analysis** with background processing
- **Color-coded metrics** (green/amber/red) at a glance
- **Full-screen zoomable diagrams** with keyboard navigation (arrow keys, Escape)
- **Collapsible plot sections** — waveform, null test, spectral difference, spectrograms, segmental SNR, PESQ per chunk, magnitude spectrum
- **PDF report download** per measurement
- **Persistent results** in SQLite

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
├── generate_reference.py     # Reference builder (CLI + library for web UI)
├── measure.py                # Measurement pipeline (CLI + library)
├── report.py                 # PDF report and web plot generation
├── web.py                    # Flask web GUI with reference builder + results dashboard
├── requirements.txt
└── SQAM_FLAC_00s9l4/         # EBU SQAM source files (download separately for custom references)
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
