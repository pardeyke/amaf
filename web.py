"""
AMAF Web GUI — upload captured audio, run measurements, browse results.

    python web.py          # starts on http://localhost:5000
"""

import os
import json
import sqlite3
import uuid
import threading
from datetime import datetime

from flask import (
    Flask, request, redirect, url_for,
    render_template_string, send_file, jsonify,
)

from measure import run_measurement
from report import generate_plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "amaf.db")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            label TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'processing',
            error TEXT,
            snr_db REAL,
            null_depth_db REAL,
            thd_n_pct REAL,
            pesq_mos REAL,
            peaq_odg REAL,
            visqol_mos REAL,
            latency_s REAL,
            duration_s REAL,
            results_json TEXT
        )
    """)
    db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def process_measurement(measurement_id, audio_path, filename):
    db = get_db()
    try:
        report_path = os.path.join(DATA_DIR, f"{measurement_id}.pdf")
        plots_dir = os.path.join(DATA_DIR, f"{measurement_id}_plots")
        results = run_measurement(audio_path, report_path, plots_dir=plots_dir)

        spec = results["spectral"]
        snr = results["snr"]
        pesq_r = results["pesq"]
        peaq_r = results["peaq"]
        visqol_r = results["visqol"]
        align = results["alignment"]

        db.execute("""
            UPDATE measurements SET
                status = 'done',
                snr_db = ?, null_depth_db = ?, thd_n_pct = ?,
                pesq_mos = ?, peaq_odg = ?, visqol_mos = ?,
                latency_s = ?, duration_s = ?,
                results_json = ?
            WHERE id = ?
        """, (
            snr["snr_db"],
            spec["null_depth_db"],
            snr["thd_n_pct"],
            pesq_r["mos_lqo_mean"] if pesq_r else None,
            peaq_r["odg"] if peaq_r and peaq_r.get("odg") is not None else None,
            visqol_r["moslqo"] if visqol_r and "moslqo" in visqol_r else None,
            align["offset_samples"] / 44100,
            results["duration_s"],
            json.dumps(results, default=str),
            measurement_id,
        ))
    except Exception as e:
        db.execute("UPDATE measurements SET status = 'error', error = ? WHERE id = ?",
                   (str(e), measurement_id))
    finally:
        db.commit()
        db.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM measurements ORDER BY created_at DESC"
    ).fetchall()
    db.close()
    return render_template_string(PAGE_INDEX, measurements=rows)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("audio")
    if not f or not f.filename:
        return redirect(url_for("index"))

    label = request.form.get("label", "").strip()
    mid = uuid.uuid4().hex[:12]
    ext = os.path.splitext(f.filename)[1] or ".wav"
    audio_path = os.path.join(DATA_DIR, f"{mid}{ext}")
    f.save(audio_path)

    db = get_db()
    db.execute(
        "INSERT INTO measurements (id, filename, label, created_at, status) VALUES (?, ?, ?, ?, ?)",
        (mid, f.filename, label or None, datetime.now().isoformat(), "processing"),
    )
    db.commit()
    db.close()

    thread = threading.Thread(
        target=process_measurement, args=(mid, audio_path, f.filename), daemon=True
    )
    thread.start()

    return redirect(url_for("index"))


@app.route("/result/<mid>")
def result(mid):
    db = get_db()
    row = db.execute("SELECT * FROM measurements WHERE id = ?", (mid,)).fetchone()
    db.close()
    if not row:
        return "Not found", 404

    detail = json.loads(row["results_json"]) if row["results_json"] else None
    return render_template_string(PAGE_RESULT, m=row, detail=detail)


@app.route("/plot/<mid>/<name>")
def plot(mid, name):
    if not name.endswith(".png"):
        return "Not found", 404
    path = os.path.join(DATA_DIR, f"{mid}_plots", name)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/png")


@app.route("/report/<mid>")
def report(mid):
    path = os.path.join(DATA_DIR, f"{mid}.pdf")
    if not os.path.exists(path):
        return "Report not ready", 404
    return send_file(path, mimetype="application/pdf")


@app.route("/status/<mid>")
def status(mid):
    db = get_db()
    row = db.execute("SELECT status FROM measurements WHERE id = ?", (mid,)).fetchone()
    db.close()
    if not row:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": row["status"]})


@app.route("/delete/<mid>", methods=["POST"])
def delete(mid):
    db = get_db()
    db.execute("DELETE FROM measurements WHERE id = ?", (mid,))
    db.commit()
    db.close()
    # Clean up files
    for ext in (".pdf", ".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus", ".aac"):
        p = os.path.join(DATA_DIR, f"{mid}{ext}")
        if os.path.exists(p):
            os.remove(p)
    import shutil
    plots_dir = os.path.join(DATA_DIR, f"{mid}_plots")
    if os.path.isdir(plots_dir):
        shutil.rmtree(plots_dir, ignore_errors=True)
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

PAGE_INDEX = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AMAF</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2e3345; --text: #e4e6f0; --muted: #8b8fa3;
    --accent: #6c8aff; --accent2: #4ecdc4; --green: #4caf50;
    --orange: #ff9800; --red: #f44336;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; }
  h1 span { color: var(--accent); }
  .subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }

  /* Upload card */
  .upload-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;
  }
  .upload-card h2 { font-size: 1rem; margin-bottom: 1rem; color: var(--accent2); }
  .upload-form { display: flex; gap: 0.75rem; align-items: end; flex-wrap: wrap; }
  .field { display: flex; flex-direction: column; gap: 0.3rem; }
  .field label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  input[type="text"], input[type="file"] {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.6rem 0.8rem; color: var(--text);
    font-size: 0.9rem;
  }
  input[type="file"] { cursor: pointer; }
  input[type="file"]::file-selector-button {
    background: var(--accent); color: white; border: none;
    border-radius: 6px; padding: 0.4rem 0.8rem; cursor: pointer;
    margin-right: 0.5rem; font-size: 0.8rem;
  }
  .btn {
    background: var(--accent); color: white; border: none; border-radius: 8px;
    padding: 0.6rem 1.5rem; font-size: 0.9rem; cursor: pointer;
    font-weight: 600; transition: opacity 0.2s;
  }
  .btn:hover { opacity: 0.85; }
  .btn-sm { padding: 0.35rem 0.8rem; font-size: 0.8rem; }
  .btn-red { background: var(--red); }

  /* Results table */
  .results h2 { font-size: 1rem; margin-bottom: 1rem; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left; padding: 0.6rem 0.8rem; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); border-bottom: 1px solid var(--border);
  }
  td {
    padding: 0.7rem 0.8rem; border-bottom: 1px solid var(--border);
    font-size: 0.9rem; font-variant-numeric: tabular-nums;
  }
  tr:hover td { background: var(--surface2); }
  .mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem; }

  /* Status badges */
  .badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 99px;
    font-size: 0.75rem; font-weight: 600;
  }
  .badge-processing { background: rgba(108,138,255,0.15); color: var(--accent); }
  .badge-done { background: rgba(76,175,80,0.15); color: var(--green); }
  .badge-error { background: rgba(244,67,54,0.15); color: var(--red); }

  /* Metric cells */
  .good { color: var(--green); }
  .warn { color: var(--orange); }
  .bad { color: var(--red); }

  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .actions { display: flex; gap: 0.4rem; }
  .empty { text-align: center; padding: 3rem; color: var(--muted); }
</style>
</head>
<body>
<div class="container">
  <h1><span>AMAF</span></h1>
  <p class="subtitle">Audio Multi-Method Assessment Fusion</p>

  <div class="upload-card">
    <h2>New Measurement</h2>
    <form class="upload-form" method="post" action="/upload" enctype="multipart/form-data">
      <div class="field">
        <label>Audio file</label>
        <input type="file" name="audio" accept="audio/*,.wav,.flac,.m4a,.mp3,.ogg,.opus,.aac" required>
      </div>
      <div class="field">
        <label>Label (optional)</label>
        <input type="text" name="label" placeholder="e.g. YouTube 1080p AAC">
      </div>
      <button class="btn" type="submit">Analyse</button>
    </form>
  </div>

  <div class="results">
    <h2>Measurements ({{ measurements|length }})</h2>
    {% if measurements %}
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>File</th>
          <th>Label</th>
          <th>Status</th>
          <th>SNR</th>
          <th>Null</th>
          <th>THD+N</th>
          <th>PESQ</th>
          <th>Latency</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {% for m in measurements %}
        <tr data-id="{{ m.id }}" data-status="{{ m.status }}">
          <td>{{ m.created_at[:16] | replace('T', ' ') }}</td>
          <td class="mono">{{ m.filename[:30] }}</td>
          <td>{{ m.label or '' }}</td>
          <td>
            <span class="badge badge-{{ m.status }}">{{ m.status }}</span>
          </td>
          {% if m.status == 'done' %}
          <td class="mono {{ 'good' if m.snr_db and m.snr_db >= 30 else ('warn' if m.snr_db and m.snr_db >= 20 else 'bad') }}">
            {{ '%.1f'|format(m.snr_db) if m.snr_db is not none else '-' }} dB
          </td>
          <td class="mono">{{ '%.1f'|format(m.null_depth_db) if m.null_depth_db is not none else '-' }} dB</td>
          <td class="mono {{ 'good' if m.thd_n_pct is not none and m.thd_n_pct < 3 else ('warn' if m.thd_n_pct is not none and m.thd_n_pct < 10 else 'bad') }}">
            {{ '%.2f'|format(m.thd_n_pct) if m.thd_n_pct is not none else '-' }}%
          </td>
          <td class="mono {{ 'good' if m.pesq_mos and m.pesq_mos >= 4.0 else ('warn' if m.pesq_mos and m.pesq_mos >= 3.5 else 'bad') }}">
            {{ '%.2f'|format(m.pesq_mos) if m.pesq_mos is not none else '-' }}
          </td>
          <td class="mono">{{ '%.2f'|format(m.latency_s) if m.latency_s is not none else '-' }}s</td>
          <td>
            <div class="actions">
              <a href="/result/{{ m.id }}" class="btn btn-sm">Details</a>
              <a href="/report/{{ m.id }}" class="btn btn-sm" style="background:var(--accent2)">PDF</a>
              <form method="post" action="/delete/{{ m.id }}" style="display:inline"
                    onsubmit="return confirm('Delete this measurement?')">
                <button class="btn btn-sm btn-red" type="submit">Del</button>
              </form>
            </div>
          </td>
          {% elif m.status == 'error' %}
          <td colspan="5" style="color:var(--red)">{{ m.error[:80] if m.error else 'Unknown error' }}</td>
          <td>
            <form method="post" action="/delete/{{ m.id }}" style="display:inline">
              <button class="btn btn-sm btn-red" type="submit">Del</button>
            </form>
          </td>
          {% else %}
          <td colspan="5" style="color:var(--muted)">Processing...</td>
          <td></td>
          {% endif %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <div class="empty">No measurements yet. Upload a captured audio file to get started.</div>
    {% endif %}
  </div>
</div>

<script>
// Auto-refresh rows that are still processing
(function poll() {
  const rows = document.querySelectorAll('tr[data-status="processing"]');
  if (!rows.length) return;
  setTimeout(() => {
    rows.forEach(row => {
      fetch('/status/' + row.dataset.id)
        .then(r => r.json())
        .then(d => { if (d.status !== 'processing') location.reload(); });
    });
    poll();
  }, 3000);
})();
</script>
</body>
</html>
"""


PAGE_RESULT = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AMAF — {{ m.filename }}</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2e3345; --text: #e4e6f0; --muted: #8b8fa3;
    --accent: #6c8aff; --accent2: #4ecdc4; --green: #4caf50;
    --orange: #ff9800; --red: #f44336;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .back { font-size: 0.85rem; margin-bottom: 1.5rem; display: inline-block; }
  h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
  .meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }

  /* Metric cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 2rem; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem; text-align: center;
  }
  .card .value { font-size: 1.5rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .card .label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; margin-top: 0.2rem; }
  .good { color: var(--green); }
  .warn { color: var(--orange); }
  .bad { color: var(--red); }

  /* Plot sections */
  .plot-section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; margin-bottom: 1rem; overflow: hidden;
  }
  .plot-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.8rem 1.2rem; border-bottom: 1px solid var(--border); cursor: pointer;
  }
  .plot-header h2 { font-size: 0.9rem; color: var(--accent2); margin: 0; }
  .plot-header .toggle { color: var(--muted); font-size: 1.2rem; transition: transform 0.2s; }
  .plot-header.collapsed .toggle { transform: rotate(-90deg); }
  .plot-body { padding: 0.5rem; }
  .plot-body.hidden { display: none; }
  .plot-img {
    width: 100%; border-radius: 8px; cursor: zoom-in;
    transition: opacity 0.2s;
  }
  .plot-img:hover { opacity: 0.9; }

  /* Side-by-side plots */
  .plot-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
  .plot-row .plot-section { margin-bottom: 0; }

  /* Detail data sections */
  .data-section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
  }
  .data-section h2 { font-size: 0.9rem; color: var(--accent2); margin-bottom: 0.8rem; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem 2rem; }
  .kv { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid var(--border); }
  .kv:last-child { border: none; }
  .k { color: var(--muted); font-size: 0.85rem; }
  .v { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem; }

  .btn {
    display: inline-block; background: var(--accent); color: white; border: none;
    border-radius: 8px; padding: 0.5rem 1.2rem; font-size: 0.85rem; cursor: pointer;
    text-decoration: none; font-weight: 600; margin-right: 0.5rem;
  }
  .btn:hover { opacity: 0.85; text-decoration: none; }
  .btn-accent2 { background: var(--accent2); }

  /* Lightbox */
  .lightbox {
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.92); justify-content: center; align-items: center;
    cursor: zoom-out;
  }
  .lightbox.active { display: flex; }
  .lightbox img {
    max-width: 95vw; max-height: 95vh; border-radius: 8px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  .lightbox-close {
    position: fixed; top: 1rem; right: 1.5rem; color: var(--muted);
    font-size: 2rem; cursor: pointer; z-index: 1001; line-height: 1;
  }
  .lightbox-close:hover { color: var(--text); }
  .lightbox-nav {
    position: fixed; top: 50%; transform: translateY(-50%);
    color: var(--muted); font-size: 2.5rem; cursor: pointer; z-index: 1001;
    padding: 1rem; user-select: none;
  }
  .lightbox-nav:hover { color: var(--text); }
  .lightbox-prev { left: 0.5rem; }
  .lightbox-next { right: 0.5rem; }
  .lightbox-caption {
    position: fixed; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
    color: var(--muted); font-size: 0.85rem; z-index: 1001;
    background: rgba(0,0,0,0.6); padding: 0.3rem 1rem; border-radius: 6px;
  }
</style>
</head>
<body>

<!-- Lightbox -->
<div class="lightbox" id="lightbox">
  <span class="lightbox-close" id="lb-close">&times;</span>
  <span class="lightbox-nav lightbox-prev" id="lb-prev">&#8249;</span>
  <span class="lightbox-nav lightbox-next" id="lb-next">&#8250;</span>
  <img id="lb-img" src="">
  <div class="lightbox-caption" id="lb-caption"></div>
</div>

<div class="container">
  <a class="back" href="/">&larr; All measurements</a>
  <h1>{{ m.filename }}</h1>
  <p class="meta">
    {{ m.created_at[:16] | replace('T', ' ') }}
    {% if m.label %} &mdash; {{ m.label }}{% endif %}
    &mdash; {{ '%.1f'|format(m.duration_s) if m.duration_s else '?' }}s analysed
  </p>

  <!-- Metric cards -->
  <div class="cards">
    <div class="card">
      <div class="value {{ 'good' if m.snr_db and m.snr_db >= 30 else ('warn' if m.snr_db and m.snr_db >= 20 else 'bad') }}">
        {{ '%.1f'|format(m.snr_db) if m.snr_db is not none else '-' }}
      </div>
      <div class="label">SNR (dB)</div>
    </div>
    <div class="card">
      <div class="value">{{ '%.1f'|format(m.null_depth_db) if m.null_depth_db is not none else '-' }}</div>
      <div class="label">Null Depth (dB)</div>
    </div>
    <div class="card">
      <div class="value {{ 'good' if m.thd_n_pct is not none and m.thd_n_pct < 3 else ('warn' if m.thd_n_pct is not none and m.thd_n_pct < 10 else 'bad') }}">
        {{ '%.2f'|format(m.thd_n_pct) if m.thd_n_pct is not none else '-' }}%
      </div>
      <div class="label">THD+N</div>
    </div>
    <div class="card">
      <div class="value {{ 'good' if m.pesq_mos and m.pesq_mos >= 4.0 else ('warn' if m.pesq_mos and m.pesq_mos >= 3.5 else 'bad') }}">
        {{ '%.2f'|format(m.pesq_mos) if m.pesq_mos is not none else '-' }}
      </div>
      <div class="label">PESQ MOS</div>
    </div>
    <div class="card">
      <div class="value">{{ '%.2f'|format(m.latency_s) if m.latency_s is not none else '-' }}s</div>
      <div class="label">Latency</div>
    </div>
    {% if m.peaq_odg is not none %}
    <div class="card">
      <div class="value">{{ '%.2f'|format(m.peaq_odg) }}</div>
      <div class="label">PEAQ ODG</div>
    </div>
    {% endif %}
    {% if m.visqol_mos is not none %}
    <div class="card">
      <div class="value">{{ '%.2f'|format(m.visqol_mos) }}</div>
      <div class="label">ViSQOL MOS</div>
    </div>
    {% endif %}
  </div>

  <!-- Diagrams -->
  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Waveform Comparison</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <img class="plot-img" data-title="Waveform Comparison"
           src="/plot/{{ m.id }}/waveform.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Null Test &mdash; Difference Signal Envelope</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <img class="plot-img" data-title="Null Test"
           src="/plot/{{ m.id }}/null_test.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Spectral Difference</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <img class="plot-img" data-title="Spectral Difference"
           src="/plot/{{ m.id }}/spectral_diff.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Average Magnitude Spectrum</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <img class="plot-img" data-title="Average Magnitude Spectrum"
           src="/plot/{{ m.id }}/magnitude_spectrum.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Spectrograms (Reference vs Processed)</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <img class="plot-img" data-title="Spectrograms"
           src="/plot/{{ m.id }}/spectrograms.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Segmental SNR (1s windows)</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <img class="plot-img" data-title="Segmental SNR"
           src="/plot/{{ m.id }}/segmental_snr.png" loading="lazy">
    </div>
  </div>

  <div class="plot-row">
    <div class="plot-section">
      <div class="plot-header" onclick="togglePlot(this)">
        <h2>PESQ per Chunk</h2><span class="toggle">&#9660;</span>
      </div>
      <div class="plot-body">
        <img class="plot-img" data-title="PESQ per Chunk"
             src="/plot/{{ m.id }}/pesq_chunks.png" loading="lazy">
      </div>
    </div>
    <div class="plot-section">
      <div class="plot-header" onclick="togglePlot(this)">
        <h2>Spectral Difference by Band</h2><span class="toggle">&#9660;</span>
      </div>
      <div class="plot-body">
        <img class="plot-img" data-title="Spectral Difference by Band"
             src="/plot/{{ m.id }}/band_diff.png" loading="lazy">
      </div>
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Difference Signal Spectrogram</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <img class="plot-img" data-title="Difference Signal Spectrogram"
           src="/plot/{{ m.id }}/diff_spectrogram.png" loading="lazy">
    </div>
  </div>

  <!-- Detailed metrics -->
  {% if detail %}
  <div class="data-section">
    <h2>Spectral Difference</h2>
    <div class="grid2">
      <div class="kv"><span class="k">Mean difference</span><span class="v">{{ '%.2f'|format(detail.spectral.mean_spectral_diff_db) }} dB</span></div>
      <div class="kv"><span class="k">Max difference</span><span class="v">{{ '%.2f'|format(detail.spectral.max_spectral_diff_db) }} dB</span></div>
      {% for band, val in detail.spectral.band_diffs_db.items() %}
      <div class="kv"><span class="k">{{ band }}</span><span class="v">{{ '%.2f'|format(val) }} dB</span></div>
      {% endfor %}
    </div>
  </div>

  <div class="data-section">
    <h2>SNR / THD+N</h2>
    <div class="grid2">
      <div class="kv"><span class="k">Overall SNR</span><span class="v">{{ '%.1f'|format(detail.snr.snr_db) }} dB</span></div>
      <div class="kv"><span class="k">Segmental SNR (mean)</span><span class="v">{{ '%.1f'|format(detail.snr.segmental_snr_db_mean) }} dB</span></div>
      <div class="kv"><span class="k">Segmental SNR (min)</span><span class="v">{{ '%.1f'|format(detail.snr.segmental_snr_db_min) }} dB</span></div>
      <div class="kv"><span class="k">THD+N</span><span class="v">{{ '%.1f'|format(detail.snr.thd_n_db) }} dB ({{ '%.2f'|format(detail.snr.thd_n_pct) }}%)</span></div>
    </div>
  </div>

  {% if detail.pesq %}
  <div class="data-section">
    <h2>PESQ (ITU-T P.862)</h2>
    <div class="grid2">
      <div class="kv"><span class="k">MOS-LQO mean</span><span class="v">{{ '%.3f'|format(detail.pesq.mos_lqo_mean) }}</span></div>
      <div class="kv"><span class="k">MOS-LQO min</span><span class="v">{{ '%.3f'|format(detail.pesq.mos_lqo_min) }}</span></div>
      <div class="kv"><span class="k">MOS-LQO max</span><span class="v">{{ '%.3f'|format(detail.pesq.mos_lqo_max) }}</span></div>
      <div class="kv"><span class="k">Chunks</span><span class="v">{{ detail.pesq.n_chunks }}</span></div>
    </div>
  </div>
  {% endif %}

  <div class="data-section">
    <h2>Alignment</h2>
    <div class="grid2">
      <div class="kv"><span class="k">Offset</span><span class="v">{{ detail.alignment.offset_samples }} samples</span></div>
      <div class="kv"><span class="k">Confidence</span><span class="v">{{ '%.0f'|format(detail.alignment.confidence) }}x</span></div>
    </div>
  </div>
  {% endif %}

  <div style="margin-top: 1.5rem;">
    <a class="btn btn-accent2" href="/report/{{ m.id }}">Download PDF Report</a>
    <a class="btn" href="/">Back to Dashboard</a>
  </div>
</div>

<script>
// Collapsible sections
function togglePlot(header) {
  const body = header.nextElementSibling;
  body.classList.toggle('hidden');
  header.classList.toggle('collapsed');
}

// Lightbox
const images = Array.from(document.querySelectorAll('.plot-img'));
const lb = document.getElementById('lightbox');
const lbImg = document.getElementById('lb-img');
const lbCaption = document.getElementById('lb-caption');
let currentIdx = 0;

function openLightbox(idx) {
  currentIdx = idx;
  lbImg.src = images[idx].src;
  lbCaption.textContent = images[idx].dataset.title || '';
  lb.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  lb.classList.remove('active');
  document.body.style.overflow = '';
}

function navigate(delta) {
  currentIdx = (currentIdx + delta + images.length) % images.length;
  lbImg.src = images[currentIdx].src;
  lbCaption.textContent = images[currentIdx].dataset.title || '';
}

images.forEach((img, i) => img.addEventListener('click', () => openLightbox(i)));
document.getElementById('lb-close').addEventListener('click', closeLightbox);
document.getElementById('lb-prev').addEventListener('click', (e) => { e.stopPropagation(); navigate(-1); });
document.getElementById('lb-next').addEventListener('click', (e) => { e.stopPropagation(); navigate(1); });
lb.addEventListener('click', (e) => { if (e.target === lb) closeLightbox(); });
document.addEventListener('keydown', (e) => {
  if (!lb.classList.contains('active')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    init_db()
    print("AMAF Web GUI — http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
