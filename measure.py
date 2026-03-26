"""
AMAF — Audio Multi-Method Assessment Fusion

Usage:
    python measure.py <captured.wav>

Aligns the captured (degraded) audio to reference.wav using the sync chirp,
then runs all available quality metrics:
  - Spectral difference / null test
  - SNR, THD+N
  - PESQ (ITU-T P.862)
  - PEAQ (ITU-R BS.1387) — requires gstpeaq
  - ViSQOL — requires visqol binary
"""

import sys
import os
import subprocess
import tempfile
import shutil
import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq
import soundfile as sf

SR = 44100
CHIRP_DURATION = 2.0
CHIRP_F0 = 20.0
CHIRP_F1 = 20000.0
CHIRP_AMPLITUDE = 10 ** (-3.0 / 20)

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.wav")


# ---------------------------------------------------------------------------
# Chirp generation (must match generate_reference.py exactly)
# ---------------------------------------------------------------------------

def generate_chirp_mono(duration=CHIRP_DURATION, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    phase = 2 * np.pi * (CHIRP_F0 * t + (CHIRP_F1 - CHIRP_F0) / (2 * duration) * t ** 2)
    return (CHIRP_AMPLITUDE * np.sin(phase)).astype(np.float64)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align(ref, deg, sr):
    """Align degraded to reference using cross-correlation of the sync chirp."""
    chirp_mono = generate_chirp_mono()
    chirp_len = len(chirp_mono)

    # Work with mono for correlation
    ref_mono = ref.mean(axis=1) if ref.ndim > 1 else ref
    deg_mono = deg.mean(axis=1) if deg.ndim > 1 else deg

    # Search for chirp in degraded audio — only scan first 60s to limit computation
    search_len = min(len(deg_mono), sr * 60)
    deg_search = deg_mono[:search_len]

    corr = sig.fftconvolve(deg_search, chirp_mono[::-1], mode="full")
    peak_idx = np.argmax(np.abs(corr))
    deg_chirp_start = peak_idx - chirp_len + 1

    # The chirp starts at sample 0 in the reference
    ref_chirp_start = 0

    # Offset: how many samples into degraded corresponds to sample 0 of reference
    offset = deg_chirp_start - ref_chirp_start

    print(f"Alignment: chirp found at sample {deg_chirp_start} in degraded "
          f"({deg_chirp_start / sr:.3f}s)")
    print(f"Offset: {offset} samples ({offset / sr:.4f}s)")

    # Correlation confidence: peak vs median
    peak_val = np.abs(corr[peak_idx])
    median_val = np.median(np.abs(corr))
    confidence = peak_val / median_val if median_val > 0 else float("inf")
    print(f"Correlation confidence: {confidence:.1f}x above median")
    if confidence < 10:
        print("WARNING: Low correlation confidence — alignment may be inaccurate")

    # Apply offset
    if offset >= 0:
        deg_aligned = deg[offset:]
    else:
        pad = [(abs(offset), 0)] + ([(0, 0)] if deg.ndim > 1 else [])
        deg_aligned = np.pad(deg, pad)

    # Trim to same length
    min_len = min(len(ref), len(deg_aligned))
    info = {
        "offset_samples": offset,
        "confidence": confidence,
    }
    return ref[:min_len], deg_aligned[:min_len], info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def to_mono(audio):
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def spectral_difference(ref, deg, sr):
    """Spectral difference and null test between reference and degraded."""
    ref_m = to_mono(ref)
    deg_m = to_mono(deg)

    # Null test: difference signal
    diff = ref_m - deg_m

    # RMS of difference vs reference
    rms_ref = np.sqrt(np.mean(ref_m ** 2))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    if rms_ref > 0:
        null_depth_db = 20 * np.log10(rms_diff / rms_ref) if rms_diff > 0 else -np.inf
    else:
        null_depth_db = 0.0

    # Spectral difference (magnitude spectrum comparison)
    n_fft = 8192
    freqs = rfftfreq(n_fft, 1.0 / sr)

    # Compute average magnitude spectra using Welch-like windowed segments
    hop = n_fft // 2
    n_segments = max(1, (len(ref_m) - n_fft) // hop)
    window = np.hanning(n_fft)

    ref_mag_acc = np.zeros(n_fft // 2 + 1)
    deg_mag_acc = np.zeros(n_fft // 2 + 1)

    for i in range(n_segments):
        start = i * hop
        ref_seg = ref_m[start:start + n_fft] * window
        deg_seg = deg_m[start:start + n_fft] * window
        ref_mag_acc += np.abs(rfft(ref_seg))
        deg_mag_acc += np.abs(rfft(deg_seg))

    ref_mag_avg = ref_mag_acc / n_segments
    deg_mag_avg = deg_mag_acc / n_segments

    # Spectral difference in dB per frequency bin
    with np.errstate(divide="ignore", invalid="ignore"):
        spec_diff_db = 20 * np.log10(
            np.where(ref_mag_avg > 0, deg_mag_avg / ref_mag_avg, 1.0)
        )

    # Summary stats over audible range
    audible = (freqs >= 20) & (freqs <= 20000)
    mean_spec_diff = np.mean(np.abs(spec_diff_db[audible]))
    max_spec_diff = np.max(np.abs(spec_diff_db[audible]))

    # Band-limited spectral differences
    bands = [
        ("Low (20-200 Hz)", 20, 200),
        ("Mid (200-4000 Hz)", 200, 4000),
        ("High (4000-20000 Hz)", 4000, 20000),
    ]
    band_diffs = {}
    for name, f_lo, f_hi in bands:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        band_diffs[name] = np.mean(np.abs(spec_diff_db[mask]))

    return {
        "null_depth_db": null_depth_db,
        "rms_diff": rms_diff,
        "mean_spectral_diff_db": mean_spec_diff,
        "max_spectral_diff_db": max_spec_diff,
        "band_diffs_db": band_diffs,
    }


def snr_thd_n(ref, deg, sr):
    """SNR and THD+N measurements."""
    ref_m = to_mono(ref)
    deg_m = to_mono(deg)

    noise = deg_m - ref_m

    # Overall SNR
    power_signal = np.mean(ref_m ** 2)
    power_noise = np.mean(noise ** 2)

    if power_noise > 0:
        snr_db = 10 * np.log10(power_signal / power_noise)
    else:
        snr_db = float("inf")

    # Segmental SNR (1-second segments, excluding silence)
    seg_len = sr
    n_segs = len(ref_m) // seg_len
    seg_snrs = []
    for i in range(n_segs):
        s = i * seg_len
        e = s + seg_len
        p_sig = np.mean(ref_m[s:e] ** 2)
        p_noise = np.mean(noise[s:e] ** 2)
        # Skip near-silent segments
        if p_sig < 1e-8:
            continue
        if p_noise > 0:
            seg_snrs.append(10 * np.log10(p_sig / p_noise))

    # THD+N: ratio of (noise) to (signal + noise) power in degraded
    power_deg = np.mean(deg_m ** 2)
    if power_deg > 0:
        thd_n_ratio = np.sqrt(power_noise / power_deg)
        thd_n_db = 20 * np.log10(thd_n_ratio) if thd_n_ratio > 0 else -np.inf
        thd_n_pct = thd_n_ratio * 100
    else:
        thd_n_db = 0.0
        thd_n_pct = 0.0

    return {
        "snr_db": snr_db,
        "segmental_snr_db_mean": np.mean(seg_snrs) if seg_snrs else float("nan"),
        "segmental_snr_db_min": np.min(seg_snrs) if seg_snrs else float("nan"),
        "thd_n_db": thd_n_db,
        "thd_n_pct": thd_n_pct,
    }


def run_pesq(ref, deg, sr):
    """PESQ (ITU-T P.862) wideband MOS."""
    try:
        from pesq import pesq
    except ImportError:
        print("  PESQ: not available (pip install pesq)")
        return None

    ref_m = to_mono(ref)
    deg_m = to_mono(deg)

    # PESQ requires 8000 or 16000 Hz
    from scipy.signal import resample_poly
    from math import gcd

    target_sr = 16000
    if sr != target_sr:
        g = gcd(sr, target_sr)
        ref_16k = resample_poly(ref_m, target_sr // g, sr // g)
        deg_16k = resample_poly(deg_m, target_sr // g, sr // g)
    else:
        ref_16k, deg_16k = ref_m, deg_m

    min_len = min(len(ref_16k), len(deg_16k))
    ref_16k = ref_16k[:min_len]
    deg_16k = deg_16k[:min_len]

    # Process in 30s chunks to avoid PESQ internal limits
    chunk_dur = 30
    chunk_len = chunk_dur * target_sr
    n_chunks = int(np.ceil(min_len / chunk_len))

    scores = []
    for i in range(n_chunks):
        start = i * chunk_len
        end = min(start + chunk_len, min_len)
        if end - start < target_sr:
            continue
        try:
            s = pesq(target_sr, ref_16k[start:end], deg_16k[start:end], "wb")
            scores.append(s)
        except Exception as e:
            print(f"  PESQ chunk {i + 1}/{n_chunks}: skipped ({e})")

    if not scores:
        return None

    return {
        "mos_lqo_mean": float(np.mean(scores)),
        "mos_lqo_min": float(np.min(scores)),
        "mos_lqo_max": float(np.max(scores)),
        "n_chunks": len(scores),
        "chunk_scores": scores,
    }


def run_peaq(ref_path, deg_path):
    """PEAQ via gstpeaq (GStreamer plugin)."""
    if not shutil.which("gstpeaq"):
        print("  PEAQ: not available (install gstpeaq)")
        return None

    try:
        result = subprocess.run(
            ["gstpeaq", "--basic", ref_path, deg_path],
            capture_output=True, text=True, timeout=120,
        )
        # Parse ODG and DI from output
        odg = None
        di = None
        for line in result.stdout.splitlines():
            if "ODG" in line:
                odg = float(line.split(":")[-1].strip())
            if "DI" in line:
                di = float(line.split(":")[-1].strip())
        return {"odg": odg, "di": di}
    except Exception as e:
        print(f"  PEAQ: error ({e})")
        return None


def run_visqol(ref_path, deg_path):
    """ViSQOL via command-line binary."""
    visqol_bin = shutil.which("visqol")
    if not visqol_bin:
        print("  ViSQOL: not available (install visqol)")
        return None

    try:
        result = subprocess.run(
            [visqol_bin,
             "--reference_file", ref_path,
             "--degraded_file", deg_path,
             "--use_speech_mode", "false"],
            capture_output=True, text=True, timeout=300,
        )
        # Parse MOS-LQO from output
        for line in result.stdout.splitlines():
            if "MOS-LQO" in line.upper():
                score = float(line.split(":")[-1].strip())
                return {"moslqo": score}
        return {"raw_output": result.stdout}
    except Exception as e:
        print(f"  ViSQOL: error ({e})")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_measurement(captured_path, report_path=None, plots_dir=None):
    """Run full measurement pipeline. Returns dict of all results."""
    from report import generate_report, generate_plots as _generate_plots

    ref, sr_ref = sf.read(REFERENCE_PATH, dtype="float64", always_2d=True)
    try:
        deg, sr_deg = sf.read(captured_path, dtype="float64", always_2d=True)
    except Exception:
        tmp_wav = os.path.join(tempfile.gettempdir(), "amaf_decoded.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", captured_path, "-acodec", "pcm_s24le", tmp_wav],
            capture_output=True, check=True,
        )
        deg, sr_deg = sf.read(tmp_wav, dtype="float64", always_2d=True)
        os.remove(tmp_wav)

    if sr_ref != sr_deg:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_deg, sr_ref)
        channels = []
        for ch in range(deg.shape[1]):
            channels.append(resample_poly(deg[:, ch], sr_ref // g, sr_deg // g))
        deg = np.column_stack(channels)

    ref_aligned, deg_aligned, alignment_info = align(ref, deg, sr_ref)

    # Skip the chirp + silence region for quality metrics (first 3 seconds)
    skip_samples = int(3.0 * sr_ref)
    ref_content = ref_aligned[skip_samples:]
    deg_content = deg_aligned[skip_samples:]

    spec = spectral_difference(ref_content, deg_content, sr_ref)
    snr = snr_thd_n(ref_content, deg_content, sr_ref)
    pesq_result = run_pesq(ref_content, deg_content, sr_ref)

    # PEAQ / ViSQOL need temp files
    tmpdir = tempfile.mkdtemp(prefix="amaf_")
    ref_content_path = os.path.join(tmpdir, "ref_content.wav")
    deg_content_path = os.path.join(tmpdir, "deg_content.wav")
    sf.write(ref_content_path, ref_content, sr_ref, subtype="PCM_24")
    sf.write(deg_content_path, deg_content, sr_ref, subtype="PCM_24")
    peaq_result = run_peaq(ref_content_path, deg_content_path)
    visqol_result = run_visqol(ref_content_path, deg_content_path)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Generate PDF report
    if report_path:
        generate_report(
            ref=ref_content, deg=deg_content, sr=sr_ref,
            spec_results=spec, snr_results=snr,
            pesq_results=pesq_result, peaq_results=peaq_result,
            visqol_results=visqol_result,
            captured_path=captured_path, alignment_info=alignment_info,
            output_path=report_path,
        )

    # Generate web plots
    if plots_dir:
        _generate_plots(
            ref=ref_content, deg=deg_content, sr=sr_ref,
            spec_results=spec, snr_results=snr,
            pesq_results=pesq_result,
            output_dir=plots_dir,
        )

    return {
        "alignment": alignment_info,
        "duration_s": len(ref_content) / sr_ref,
        "spectral": spec,
        "snr": snr,
        "pesq": pesq_result,
        "peaq": peaq_result,
        "visqol": visqol_result,
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <captured.wav>")
        sys.exit(1)

    captured_path = sys.argv[1]

    print("=" * 60)
    print("AMAF — Audio Multi-Method Assessment Fusion")
    print("=" * 60)
    print(f"\nReference: {REFERENCE_PATH}")
    print(f"Captured:  {captured_path}")

    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"amaf_report_{os.path.splitext(os.path.basename(captured_path))[0]}.pdf",
    )

    results = run_measurement(captured_path, report_path)

    spec = results["spectral"]
    snr = results["snr"]
    pesq_result = results["pesq"]
    peaq_result = results["peaq"]
    visqol_result = results["visqol"]
    alignment_info = results["alignment"]

    print(f"\nAlignment: offset {alignment_info['offset_samples']} samples "
          f"({alignment_info['offset_samples']/SR:.3f}s), "
          f"confidence {alignment_info['confidence']:.0f}x")
    print(f"Analysed duration: {results['duration_s']:.1f}s")

    print(f"\n  Null depth:        {spec['null_depth_db']:.1f} dB")
    print(f"  SNR:               {snr['snr_db']:.1f} dB")
    print(f"  THD+N:             {snr['thd_n_pct']:.4f}%")
    if pesq_result:
        print(f"  PESQ MOS-LQO:      {pesq_result['mos_lqo_mean']:.3f}")
    if peaq_result and peaq_result.get("odg") is not None:
        print(f"  PEAQ ODG:          {peaq_result['odg']}")
    if visqol_result and "moslqo" in visqol_result:
        print(f"  ViSQOL MOS-LQO:    {visqol_result['moslqo']:.3f}")

    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
