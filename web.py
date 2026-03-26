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
from generate_reference import get_sqam_tracks, build_reference

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
# Reference builder routes
# ---------------------------------------------------------------------------

@app.route("/reference")
def reference_builder():
    tracks = get_sqam_tracks()
    ref_path = os.path.join(BASE_DIR, "reference.wav")
    ref_exists = os.path.exists(ref_path)
    ref_size = os.path.getsize(ref_path) if ref_exists else 0
    return render_template_string(
        PAGE_REFERENCE, tracks=tracks,
        ref_exists=ref_exists, ref_size=ref_size,
    )


@app.route("/reference/build", methods=["POST"])
def reference_build():
    data = request.get_json()
    track_nums = data.get("tracks", [])
    if not track_nums:
        return jsonify({"error": "No tracks selected"}), 400
    try:
        track_nums = [int(t) for t in track_nums]
        path, duration = build_reference(track_nums)
        size = os.path.getsize(path)
        return jsonify({
            "ok": True,
            "duration": round(duration, 1),
            "size": size,
            "tracks": len(track_nums),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reference/download")
def reference_download():
    path = os.path.join(BASE_DIR, "reference.wav")
    if not os.path.exists(path):
        return "No reference file yet", 404
    return send_file(path, mimetype="audio/wav", as_attachment=True,
                     download_name="reference.wav")


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

  <div class="upload-card" style="display:flex; gap:1.5rem; flex-wrap:wrap; align-items:start;">
    <div style="flex:1; min-width:300px;">
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
    <div style="border-left:1px solid var(--border); padding-left:1.5rem; min-width:200px;">
      <h2 style="color:var(--accent);">Reference Track</h2>
      <p style="font-size:0.85rem; color:var(--muted); margin-bottom:0.75rem;">
        Build or download the reference WAV to play through your pipeline.
      </p>
      <a href="/reference" class="btn" style="background:var(--accent);">Build Reference</a>
      <a href="/reference/download" class="btn" style="background:var(--accent2); margin-top:0.5rem;">Download Current</a>
    </div>
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


PAGE_REFERENCE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AMAF — Build Reference</title>
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
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .back { font-size: 0.85rem; margin-bottom: 1.5rem; display: inline-block; }
  h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
  h1 span { color: var(--accent); }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }

  /* Sticky bar */
  .sticky-bar {
    position: sticky; top: 0; z-index: 100;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;
    backdrop-filter: blur(12px);
  }
  .sticky-bar .stat { text-align: center; }
  .sticky-bar .stat .num {
    font-size: 1.4rem; font-weight: 700; color: var(--accent2);
    font-variant-numeric: tabular-nums;
  }
  .sticky-bar .stat .lbl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; }
  .btn {
    display: inline-block; background: var(--accent); color: white; border: none;
    border-radius: 8px; padding: 0.55rem 1.4rem; font-size: 0.9rem; cursor: pointer;
    text-decoration: none; font-weight: 600; transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; text-decoration: none; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-accent2 { background: var(--accent2); }
  .btn-sm { padding: 0.35rem 0.8rem; font-size: 0.8rem; }
  .btn-outline {
    background: transparent; border: 1px solid var(--border); color: var(--muted);
  }
  .btn-outline:hover { border-color: var(--accent); color: var(--accent); }

  .spacer { flex: 1; }
  .status-msg {
    font-size: 0.85rem; padding: 0.4rem 0.8rem; border-radius: 6px;
    display: none;
  }
  .status-msg.success { display: inline-block; background: rgba(76,175,80,0.15); color: var(--green); }
  .status-msg.error { display: inline-block; background: rgba(244,67,54,0.15); color: var(--red); }
  .status-msg.building { display: inline-block; background: rgba(108,138,255,0.15); color: var(--accent); }

  /* Category headers */
  .category {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--accent2); margin: 1.5rem 0 0.5rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
  }
  .category:first-of-type { margin-top: 0; }

  /* Track grid */
  .tracks { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 0.5rem; }
  .track {
    display: flex; align-items: center; gap: 0.75rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.6rem 0.8rem; cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    user-select: none;
  }
  .track:hover { border-color: var(--accent); }
  .track.selected { border-color: var(--accent2); background: rgba(78,205,196,0.08); }
  .track .num {
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem;
    color: var(--muted); min-width: 1.8rem;
  }
  .track .title { font-size: 0.85rem; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .track .dur {
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem;
    color: var(--muted); white-space: nowrap;
  }
  .track .order-badge {
    background: var(--accent2); color: var(--bg); font-size: 0.7rem;
    font-weight: 700; width: 1.4rem; height: 1.4rem; border-radius: 50%;
    display: none; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .track.selected .order-badge { display: flex; }
</style>
</head>
<body>
<div class="container">
  <a class="back" href="/">&larr; Dashboard</a>
  <h1><span>Reference Track Builder</span></h1>
  <p class="subtitle">
    Select tracks from the EBU SQAM library to build your reference file.
    A 2s sync chirp is automatically prepended for alignment.
    Click tracks in the order you want them.
  </p>

  <div class="sticky-bar">
    <div class="stat">
      <div class="num" id="sel-count">0</div>
      <div class="lbl">Tracks</div>
    </div>
    <div class="stat">
      <div class="num" id="sel-duration">0:03</div>
      <div class="lbl">Duration</div>
    </div>
    <div class="spacer"></div>
    <button class="btn btn-outline btn-sm" onclick="selectAll()">Select All</button>
    <button class="btn btn-outline btn-sm" onclick="clearAll()">Clear</button>
    <button class="btn btn-accent2" id="build-btn" onclick="buildReference()" disabled>
      Build Reference
    </button>
    <span class="status-msg" id="status"></span>
  </div>

  {% set categories = [
    ("Test Signals", [1,2,3,4,5,6,7]),
    ("Strings", [8,9,10,11]),
    ("Woodwinds", [12,13,14,15,16,17,18,19,20]),
    ("Brass", [21,22,23,24]),
    ("Plucked / Keys", [25,39,40,41,42,43,58]),
    ("Percussion", [26,27,28,29,30,31,32,33,34,35,36,37,38]),
    ("Vocals", [44,45,46,47,48]),
    ("Speech", [49,50,51,52,53,54]),
    ("Classical", [55,56,57,59,60,61,62,63,64,65,66,67,68]),
    ("Pop", [69,70]),
  ] %}

  {% for cat_name, cat_nums in categories %}
  <div class="category">{{ cat_name }}</div>
  <div class="tracks">
    {% for t in tracks if t.num in cat_nums %}
    <div class="track" data-num="{{ t.num }}" data-dur="{{ t.duration }}"
         onclick="toggleTrack(this)">
      <span class="order-badge"></span>
      <span class="num">{{ '%02d'|format(t.num) }}</span>
      <span class="title" title="{{ t.title }}">{{ t.title }}</span>
      <span class="dur">{{ '%d:%02d'|format(t.duration // 60, t.duration % 60) }}</span>
    </div>
    {% endfor %}
  </div>
  {% endfor %}
</div>

<script>
const CHIRP_DUR = 3; // 2s chirp + 1s silence
let selected = [];

function updateUI() {
  const totalDur = selected.reduce((sum, el) => sum + parseFloat(el.dataset.dur), 0)
    + selected.length  // 1s silence per track
    + CHIRP_DUR;
  const mins = Math.floor(totalDur / 60);
  const secs = Math.floor(totalDur % 60);
  document.getElementById('sel-count').textContent = selected.length;
  document.getElementById('sel-duration').textContent = mins + ':' + String(secs).padStart(2, '0');
  document.getElementById('build-btn').disabled = selected.length === 0;

  // Update order badges
  document.querySelectorAll('.track').forEach(t => {
    const idx = selected.indexOf(t);
    const badge = t.querySelector('.order-badge');
    if (idx >= 0) {
      badge.textContent = idx + 1;
      t.classList.add('selected');
    } else {
      t.classList.remove('selected');
    }
  });
}

function toggleTrack(el) {
  const idx = selected.indexOf(el);
  if (idx >= 0) {
    selected.splice(idx, 1);
  } else {
    selected.push(el);
  }
  updateUI();
}

function selectAll() {
  selected = Array.from(document.querySelectorAll('.track'));
  updateUI();
}

function clearAll() {
  selected = [];
  updateUI();
}

function buildReference() {
  const btn = document.getElementById('build-btn');
  const status = document.getElementById('status');
  const nums = selected.map(el => parseInt(el.dataset.num));

  btn.disabled = true;
  btn.textContent = 'Building...';
  status.className = 'status-msg building';
  status.textContent = 'Generating reference file...';

  fetch('/reference/build', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tracks: nums}),
  })
  .then(r => r.json())
  .then(data => {
    if (data.ok) {
      const mb = (data.size / 1048576).toFixed(1);
      status.className = 'status-msg success';
      status.textContent = 'reference.wav built (' + data.duration + 's, ' + mb + ' MB)';
    } else {
      status.className = 'status-msg error';
      status.textContent = 'Error: ' + data.error;
    }
  })
  .catch(e => {
    status.className = 'status-msg error';
    status.textContent = 'Error: ' + e.message;
  })
  .finally(() => {
    btn.disabled = false;
    btn.textContent = 'Build Reference';
  });
}

updateUI();
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
  .card .hint { font-size: 0.7rem; color: var(--muted); margin-top: 0.35rem; line-height: 1.3; }
  .good { color: var(--green); }
  .warn { color: var(--orange); }
  .bad { color: var(--red); }

  /* Explanations */
  .explain {
    font-size: 0.8rem; color: var(--muted); line-height: 1.5;
    padding: 0.5rem 0.8rem; margin-bottom: 0.5rem;
  }
  .explain strong { color: var(--text); font-weight: 600; }
  .scale { display: flex; gap: 0.6rem; margin-top: 0.3rem; flex-wrap: wrap; }
  .scale span { font-size: 0.75rem; }
  .scale .g { color: var(--green); } .scale .w { color: var(--orange); } .scale .r { color: var(--red); }

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
      <div class="hint">Signal vs. added noise. Higher = cleaner.</div>
    </div>
    <div class="card">
      <div class="value">{{ '%.1f'|format(m.null_depth_db) if m.null_depth_db is not none else '-' }}</div>
      <div class="label">Null Depth (dB)</div>
      <div class="hint">How quiet the difference is. More negative = closer to original.</div>
    </div>
    <div class="card">
      <div class="value {{ 'good' if m.thd_n_pct is not none and m.thd_n_pct < 3 else ('warn' if m.thd_n_pct is not none and m.thd_n_pct < 10 else 'bad') }}">
        {{ '%.2f'|format(m.thd_n_pct) if m.thd_n_pct is not none else '-' }}%
      </div>
      <div class="label">THD+N</div>
      <div class="hint">Total distortion as % of output. Lower = better.</div>
    </div>
    <div class="card">
      <div class="value {{ 'good' if m.pesq_mos and m.pesq_mos >= 4.0 else ('warn' if m.pesq_mos and m.pesq_mos >= 3.5 else 'bad') }}">
        {{ '%.2f'|format(m.pesq_mos) if m.pesq_mos is not none else '-' }}
      </div>
      <div class="label">PESQ MOS</div>
      <div class="hint">Perceptual quality score. 4.64 = perfect, &gt;4.0 = very good.</div>
    </div>
    <div class="card">
      <div class="value">{{ '%.2f'|format(m.latency_s) if m.latency_s is not none else '-' }}s</div>
      <div class="label">Latency</div>
      <div class="hint">Delay from playout to capture point.</div>
    </div>
    {% if m.peaq_odg is not none %}
    <div class="card">
      <div class="value">{{ '%.2f'|format(m.peaq_odg) }}</div>
      <div class="label">PEAQ ODG</div>
      <div class="hint">Perceptual audio quality. 0 = transparent, &minus;4 = very annoying.</div>
    </div>
    {% endif %}
    {% if m.visqol_mos is not none %}
    <div class="card">
      <div class="value">{{ '%.2f'|format(m.visqol_mos) }}</div>
      <div class="label">ViSQOL MOS</div>
      <div class="hint">Perceptual similarity. 5 = identical, 1 = bad.</div>
    </div>
    {% endif %}
  </div>

  <!-- Diagrams -->
  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Waveform Comparison</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <div class="explain">
        A 2-second excerpt from the middle of the audio showing original (blue) and processed (red) overlaid.
        If the pipeline is transparent, the two waveforms sit exactly on top of each other.
        Visible differences indicate level changes, clipping, or timing shifts.
      </div>
      <img class="plot-img" data-title="Waveform Comparison"
           src="/plot/{{ m.id }}/waveform.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Null Test &mdash; Difference Signal Envelope</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <div class="explain">
        Subtracting the original from the processed audio leaves only what the pipeline changed.
        In a perfect system this would be silence (&minus;&#8734; dB).
        The <strong>dashed line</strong> shows the overall average.
        Louder passages in this plot reveal where the codec introduces the most artifacts &mdash;
        typically on transients (drums, consonants) and high-frequency content.
      </div>
      <img class="plot-img" data-title="Null Test"
           src="/plot/{{ m.id }}/null_test.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Spectral Difference</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <div class="explain">
        Frequency-by-frequency comparison of the processed audio against the original.
        The line at <strong>0 dB</strong> means no change.
        Values above 0 mean the pipeline added energy (boosted); below 0 means it removed energy (attenuated).
        A sharp drop-off at high frequencies (e.g. above 16 kHz) is the codec&rsquo;s low-pass filter &mdash;
        this is normal for AAC/Opus and is usually inaudible to most listeners.
      </div>
      <img class="plot-img" data-title="Spectral Difference"
           src="/plot/{{ m.id }}/spectral_diff.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Average Magnitude Spectrum</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <div class="explain">
        The overall frequency &ldquo;fingerprint&rdquo; of both signals.
        <strong>Blue</strong> = original, <strong>red</strong> = processed.
        Where the two lines overlap the pipeline has preserved the audio faithfully.
        Divergence (especially at the high end) shows where the codec discards information to save bandwidth.
      </div>
      <img class="plot-img" data-title="Average Magnitude Spectrum"
           src="/plot/{{ m.id }}/magnitude_spectrum.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Spectrograms (Reference vs Processed)</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <div class="explain">
        A visual &ldquo;heat map&rdquo; of the audio: time runs left to right, frequency runs bottom to top,
        and brightness shows loudness. Comparing the two side by side reveals differences in spectral detail.
        Look for missing high-frequency content (dark areas at the top of the processed spectrogram)
        or smeared transients (blurry vertical lines where the original has sharp ones).
      </div>
      <img class="plot-img" data-title="Spectrograms"
           src="/plot/{{ m.id }}/spectrograms.png" loading="lazy">
    </div>
  </div>

  <div class="plot-section">
    <div class="plot-header" onclick="togglePlot(this)">
      <h2>Segmental SNR (1s windows)</h2><span class="toggle">&#9660;</span>
    </div>
    <div class="plot-body">
      <div class="explain">
        Signal-to-noise ratio measured every second across the full duration.
        <strong>Taller bars = better quality.</strong>
        <div class="scale">
          <span class="g">Green (&ge;30 dB): excellent</span>
          <span class="w">Amber (20&ndash;30 dB): acceptable</span>
          <span class="r">Red (&lt;20 dB): degraded</span>
        </div>
        Dips often correspond to quiet passages, speech sibilants, or sharp transients where lossy codecs struggle most.
      </div>
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
        <div class="explain">
          Perceptual quality scored in 30-second chunks on the MOS (Mean Opinion Score) scale, which predicts
          how a panel of human listeners would rate the quality.
          <div class="scale">
            <span class="g">&ge;4.0: very good (most listeners can&rsquo;t tell the difference)</span>
            <span class="w">3.5&ndash;4.0: noticeable but not annoying</span>
            <span class="r">&lt;3.5: clearly degraded</span>
          </div>
        </div>
        <img class="plot-img" data-title="PESQ per Chunk"
             src="/plot/{{ m.id }}/pesq_chunks.png" loading="lazy">
      </div>
    </div>
    <div class="plot-section">
      <div class="plot-header" onclick="togglePlot(this)">
        <h2>Spectral Difference by Band</h2><span class="toggle">&#9660;</span>
      </div>
      <div class="plot-body">
        <div class="explain">
          Average difference split into three ranges:
          <strong>Low</strong> (bass, 20&ndash;200 Hz),
          <strong>Mid</strong> (voice/instruments, 200&ndash;4k Hz), and
          <strong>High</strong> (detail/air, 4k&ndash;20k Hz).
          Low/mid should be near zero. High is where most codec loss shows up.
        </div>
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
      <div class="explain">
        Spectrogram of <em>only the difference</em> between original and processed &mdash; i.e. everything the pipeline changed.
        In a perfect system this would be completely black (silent).
        Bright areas show where and when artifacts were introduced.
        A persistent bright band at the top indicates the codec&rsquo;s high-frequency cutoff.
        Scattered bright spots in the mid-range reveal pre-echo or quantisation noise on transients.
      </div>
      <img class="plot-img" data-title="Difference Signal Spectrogram"
           src="/plot/{{ m.id }}/diff_spectrogram.png" loading="lazy">
    </div>
  </div>

  <!-- Detailed metrics -->
  {% if detail %}
  <div class="data-section">
    <h2>Spectral Difference</h2>
    <div class="explain">
      How much the frequency content changed. <strong>Mean</strong> is the average deviation across all audible
      frequencies; <strong>max</strong> is the worst single point (usually at the codec&rsquo;s cutoff frequency).
      The per-band breakdown shows where the damage is &mdash; low and mid should be near zero for a good codec.
    </div>
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
    <div class="explain">
      <strong>SNR</strong> (Signal-to-Noise Ratio): how much louder the wanted signal is compared to everything the
      pipeline added. Higher is better. <strong>Segmental SNR</strong> breaks this down per second &mdash; the
      minimum value shows the worst moment.<br>
      <strong>THD+N</strong> (Total Harmonic Distortion + Noise): the total amount of unwanted signal as a
      percentage of the output. For lossy codecs, values under 3% are typical of a well-configured pipeline.
    </div>
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
    <div class="explain">
      An ITU standard that predicts how human listeners would rate audio quality on a 1&ndash;4.64 scale
      (MOS &mdash; Mean Opinion Score). The audio is split into 30-second chunks and scored individually.
      <strong>Mean</strong> is the overall quality; <strong>min</strong> highlights the worst section.
      Scores above 4.0 are considered very good &mdash; most people cannot hear the difference from the original.
    </div>
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
    <div class="explain">
      Before comparing, the two audio files must be lined up sample-by-sample. AMAF uses a sync chirp
      (a rising tone at the start of the reference) to find the exact offset.
      <strong>Offset</strong> is how far into the captured file the reference audio starts &mdash;
      this equals the pipeline latency. <strong>Confidence</strong> is how clear the match was;
      values above 100x are reliable.
    </div>
    <div class="grid2">
      <div class="kv"><span class="k">Offset</span><span class="v">{{ detail.alignment.offset_samples }} samples ({{ '%.2f'|format(m.latency_s) }}s)</span></div>
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


def main():
    init_db()
    print("AMAF — Audio Multi-Method Assessment Fusion")
    print("http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
