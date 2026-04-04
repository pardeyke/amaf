"""
AMAF — Audio/Video Multi-Method Assessment Fusion

Usage:
    python measure.py <captured.wav|mp4>

Aligns the captured (degraded) media to the reference using sync chirp
(audio) and luminance chirp (video), then runs all available quality metrics:
  Audio: Spectral difference / null test, SNR, THD+N, PESQ, POLQA, PEAQ, ViSQOL
  Video: VMAF, PSNR, SSIM (via ffmpeg)
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
REFERENCE_VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_video.mp4")


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


def align_signals(ref, deg, sr):
    """Align two arbitrary signals via cross-correlation (no sync chirp needed)."""
    ref_mono = ref.mean(axis=1) if ref.ndim > 1 else ref
    deg_mono = deg.mean(axis=1) if deg.ndim > 1 else deg

    # Use first 60s of reference as correlation template
    template_len = min(len(ref_mono), sr * 60)
    template = ref_mono[:template_len]

    search_len = min(len(deg_mono), sr * 60 + template_len)
    deg_search = deg_mono[:search_len]

    corr = sig.fftconvolve(deg_search, template[::-1], mode="full")
    peak_idx = np.argmax(np.abs(corr))
    offset = peak_idx - template_len + 1

    print(f"Alignment: reference found at sample {offset} in processed "
          f"({offset / sr:.3f}s)")

    peak_val = np.abs(corr[peak_idx])
    median_val = np.median(np.abs(corr))
    confidence = peak_val / median_val if median_val > 0 else float("inf")
    print(f"Correlation confidence: {confidence:.1f}x above median")
    if confidence < 5:
        print("WARNING: Low correlation confidence — alignment may be inaccurate")

    # Apply offset
    if offset >= 0:
        deg_aligned = deg[offset:]
    else:
        ref = ref[abs(offset):]
        deg_aligned = deg

    min_len = min(len(ref), len(deg_aligned))
    info = {
        "offset_samples": offset,
        "confidence": confidence,
    }
    return ref[:min_len], deg_aligned[:min_len], info


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_rms(ref, deg):
    """Scale degraded signal to match reference RMS level. Returns (deg_scaled, gain_db)."""
    ref_mono = ref.mean(axis=1) if ref.ndim > 1 else ref
    deg_mono = deg.mean(axis=1) if deg.ndim > 1 else deg
    rms_ref = np.sqrt(np.mean(ref_mono ** 2))
    rms_deg = np.sqrt(np.mean(deg_mono ** 2))
    if rms_deg > 0:
        gain = rms_ref / rms_deg
        gain_db = 20 * np.log10(gain)
        return deg * gain, gain_db
    return deg, 0.0


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


def run_polqa(ref_path, deg_path, sr=44100):
    """POLQA (ITU-T P.863) via command-line binary."""
    polqa_bin = shutil.which("polqa")
    if not polqa_bin:
        print("  POLQA: not available (install polqa)")
        return None

    # POLQA SWB/fullband expects 48 kHz; resample if needed
    target_sr = 48000
    tmpdir_resamp = None
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd

        tmpdir_resamp = tempfile.mkdtemp(prefix="amaf_polqa_")
        g = gcd(sr, target_sr)

        ref_48k_path = os.path.join(tmpdir_resamp, "ref_48k.wav")
        deg_48k_path = os.path.join(tmpdir_resamp, "deg_48k.wav")

        ref_data, _ = sf.read(ref_path, dtype="float64")
        deg_data, _ = sf.read(deg_path, dtype="float64")

        if ref_data.ndim > 1:
            ref_48k = np.column_stack([
                resample_poly(ref_data[:, ch], target_sr // g, sr // g)
                for ch in range(ref_data.shape[1])
            ])
        else:
            ref_48k = resample_poly(ref_data, target_sr // g, sr // g)

        if deg_data.ndim > 1:
            deg_48k = np.column_stack([
                resample_poly(deg_data[:, ch], target_sr // g, sr // g)
                for ch in range(deg_data.shape[1])
            ])
        else:
            deg_48k = resample_poly(deg_data, target_sr // g, sr // g)

        sf.write(ref_48k_path, ref_48k, target_sr, subtype="PCM_24")
        sf.write(deg_48k_path, deg_48k, target_sr, subtype="PCM_24")
        ref_path = ref_48k_path
        deg_path = deg_48k_path

    try:
        result = subprocess.run(
            [polqa_bin, "-ref", ref_path, "-deg", deg_path, "-mode", "SWB"],
            capture_output=True, text=True, timeout=300,
        )
        output = result.stdout + result.stderr
        mos = None
        for line in output.splitlines():
            low = line.lower()
            # Common output formats: "MOS-LQO: 4.23" or "Score: 4.23"
            if "mos" in low and ":" in line:
                try:
                    mos = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "score" in low and ":" in line and mos is None:
                try:
                    mos = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        if mos is not None:
            return {"mos_lqo": mos}
        return {"raw_output": output}
    except Exception as e:
        print(f"  POLQA: error ({e})")
        return None
    finally:
        if tmpdir_resamp:
            shutil.rmtree(tmpdir_resamp, ignore_errors=True)


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

def run_measurement(captured_path, report_path=None, plots_dir=None, audio_dir=None, normalize=False, label=None):
    """Run full measurement pipeline. Returns dict of all results."""
    from report import generate_report, generate_plots as _generate_plots
    from video import (is_video_file, has_video_stream, has_audio_stream,
                       extract_audio, get_video_info,
                       align_video_chirp, trim_video, run_video_metrics)

    input_is_video = is_video_file(captured_path) and has_video_stream(captured_path)

    # For video input, extract the audio track
    audio_tmp = None
    if input_is_video:
        if has_audio_stream(captured_path):
            audio_tmp = os.path.join(tempfile.gettempdir(), f"amaf_audio_{os.getpid()}.wav")
            extract_audio(captured_path, audio_tmp)
            captured_audio = audio_tmp
        else:
            captured_audio = None
    else:
        captured_audio = captured_path

    ref, sr_ref = sf.read(REFERENCE_PATH, dtype="float64", always_2d=True)

    # Load audio (from extracted track or directly)
    if captured_audio:
        try:
            deg, sr_deg = sf.read(captured_audio, dtype="float64", always_2d=True)
        except Exception:
            tmp_wav = os.path.join(tempfile.gettempdir(), "amaf_decoded.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", captured_audio, "-acodec", "pcm_s24le", tmp_wav],
                capture_output=True, check=True,
            )
            deg, sr_deg = sf.read(tmp_wav, dtype="float64", always_2d=True)
            os.remove(tmp_wav)
    else:
        # Video with no audio track — create silent placeholder
        deg = np.zeros_like(ref)
        sr_deg = sr_ref

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

    # Normalize level if requested
    gain_db = 0.0
    if normalize:
        deg_content, gain_db = normalize_rms(ref_content, deg_content)
        print(f"Normalization: applied {gain_db:+.2f} dB gain to processed signal")

    # Save aligned audio for A/B playback
    if audio_dir:
        os.makedirs(audio_dir, exist_ok=True)
        sf.write(os.path.join(audio_dir, "reference.wav"), ref_content, sr_ref, subtype="PCM_16")
        sf.write(os.path.join(audio_dir, "processed.wav"), deg_content, sr_ref, subtype="PCM_16")
        diff_signal = ref_content - deg_content
        sf.write(os.path.join(audio_dir, "difference.wav"), diff_signal, sr_ref, subtype="PCM_16")

    spec = spectral_difference(ref_content, deg_content, sr_ref)
    snr = snr_thd_n(ref_content, deg_content, sr_ref)
    pesq_result = run_pesq(ref_content, deg_content, sr_ref)

    # PEAQ / ViSQOL need temp files
    tmpdir = tempfile.mkdtemp(prefix="amaf_")
    ref_content_path = os.path.join(tmpdir, "ref_content.wav")
    deg_content_path = os.path.join(tmpdir, "deg_content.wav")
    sf.write(ref_content_path, ref_content, sr_ref, subtype="PCM_24")
    sf.write(deg_content_path, deg_content, sr_ref, subtype="PCM_24")
    polqa_result = run_polqa(ref_content_path, deg_content_path, sr=sr_ref)
    peaq_result = run_peaq(ref_content_path, deg_content_path)
    visqol_result = run_visqol(ref_content_path, deg_content_path)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # --- Video analysis ---
    video_results = None
    video_info = None
    if input_is_video and os.path.exists(REFERENCE_VIDEO_PATH):
        print("\n--- Video analysis ---")
        video_info = get_video_info(captured_path)
        vtmpdir = tempfile.mkdtemp(prefix="amaf_vtrim_")
        try:
            valign = align_video_chirp(captured_path)
            skip_s = 3.0  # luma chirp duration
            ref_trimmed = os.path.join(vtmpdir, "ref_trimmed.mp4")
            deg_trimmed = os.path.join(vtmpdir, "deg_trimmed.mp4")

            ref_info = get_video_info(REFERENCE_VIDEO_PATH)
            ref_dur = ref_info["duration"] - skip_s if ref_info else 60
            deg_start = skip_s + valign["offset_seconds"]

            trim_video(REFERENCE_VIDEO_PATH, skip_s, ref_dur, ref_trimmed)
            trim_video(captured_path, max(0, deg_start), ref_dur, deg_trimmed)

            video_results = run_video_metrics(ref_trimmed, deg_trimmed)
            if video_results:
                video_results["alignment"] = valign
                video_results["info"] = video_info
        except Exception as e:
            print(f"  Video analysis failed: {e}")
        finally:
            shutil.rmtree(vtmpdir, ignore_errors=True)
    elif input_is_video:
        video_info = get_video_info(captured_path)

    # Clean up extracted audio
    if audio_tmp and os.path.exists(audio_tmp):
        os.remove(audio_tmp)

    # Generate PDF report
    if report_path:
        generate_report(
            ref=ref_content, deg=deg_content, sr=sr_ref,
            spec_results=spec, snr_results=snr,
            pesq_results=pesq_result, polqa_results=polqa_result,
            peaq_results=peaq_result,
            visqol_results=visqol_result,
            captured_path=captured_path, alignment_info=alignment_info,
            output_path=report_path,
            video_results=video_results,
            label=label,
        )

    # Generate web plots
    if plots_dir:
        _generate_plots(
            ref=ref_content, deg=deg_content, sr=sr_ref,
            spec_results=spec, snr_results=snr,
            pesq_results=pesq_result,
            output_dir=plots_dir,
            video_results=video_results,
        )

    return {
        "alignment": alignment_info,
        "duration_s": len(ref_content) / sr_ref,
        "normalized": normalize,
        "gain_db": gain_db,
        "spectral": spec,
        "snr": snr,
        "pesq": pesq_result,
        "polqa": polqa_result,
        "peaq": peaq_result,
        "visqol": visqol_result,
        "video": video_results,
        "video_info": video_info,
        "has_video": input_is_video,
    }


def run_comparison(ref_path, deg_path, report_path=None, plots_dir=None, audio_dir=None, normalize=False, label=None):
    """Run measurement comparing a user-supplied reference to a processed file."""
    from report import generate_report, generate_plots as _generate_plots
    from video import (is_video_file, has_video_stream, has_audio_stream,
                       extract_audio, get_video_info,
                       align_video_signals, trim_video, run_video_metrics)

    ref_is_video = is_video_file(ref_path) and has_video_stream(ref_path)
    deg_is_video = is_video_file(deg_path) and has_video_stream(deg_path)
    input_is_video = ref_is_video and deg_is_video

    # Extract audio from video files
    audio_tmps = []

    def _load(path):
        if is_video_file(path) and has_audio_stream(path):
            tmp = os.path.join(tempfile.gettempdir(), f"amaf_cmp_{os.getpid()}_{len(audio_tmps)}.wav")
            extract_audio(path, tmp)
            audio_tmps.append(tmp)
            return sf.read(tmp, dtype="float64", always_2d=True)
        try:
            return sf.read(path, dtype="float64", always_2d=True)
        except Exception:
            tmp_wav = os.path.join(tempfile.gettempdir(), "amaf_decoded.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-acodec", "pcm_s24le", tmp_wav],
                capture_output=True, check=True,
            )
            data, sr = sf.read(tmp_wav, dtype="float64", always_2d=True)
            os.remove(tmp_wav)
            return data, sr

    ref, sr_ref = _load(ref_path)
    deg, sr_deg = _load(deg_path)

    # Resample degraded to match reference if needed
    if sr_ref != sr_deg:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_deg, sr_ref)
        channels = []
        for ch in range(deg.shape[1]):
            channels.append(resample_poly(deg[:, ch], sr_ref // g, sr_deg // g))
        deg = np.column_stack(channels)

    # Match channel count
    if ref.shape[1] != deg.shape[1]:
        if ref.shape[1] == 1:
            ref = np.column_stack([ref[:, 0], ref[:, 0]])
        elif deg.shape[1] == 1:
            deg = np.column_stack([deg[:, 0], deg[:, 0]])
        else:
            ref = ref[:, :1].mean(axis=1, keepdims=True)
            deg = deg[:, :1].mean(axis=1, keepdims=True)

    ref_aligned, deg_aligned, alignment_info = align_signals(ref, deg, sr_ref)

    # Normalize level if requested
    gain_db = 0.0
    if normalize:
        deg_aligned, gain_db = normalize_rms(ref_aligned, deg_aligned)
        print(f"Normalization: applied {gain_db:+.2f} dB gain to processed signal")

    if audio_dir:
        os.makedirs(audio_dir, exist_ok=True)
        sf.write(os.path.join(audio_dir, "reference.wav"), ref_aligned, sr_ref, subtype="PCM_16")
        sf.write(os.path.join(audio_dir, "processed.wav"), deg_aligned, sr_ref, subtype="PCM_16")
        diff_signal = ref_aligned - deg_aligned
        sf.write(os.path.join(audio_dir, "difference.wav"), diff_signal, sr_ref, subtype="PCM_16")

    spec = spectral_difference(ref_aligned, deg_aligned, sr_ref)
    snr = snr_thd_n(ref_aligned, deg_aligned, sr_ref)
    pesq_result = run_pesq(ref_aligned, deg_aligned, sr_ref)

    tmpdir = tempfile.mkdtemp(prefix="amaf_")
    ref_content_path = os.path.join(tmpdir, "ref_content.wav")
    deg_content_path = os.path.join(tmpdir, "deg_content.wav")
    sf.write(ref_content_path, ref_aligned, sr_ref, subtype="PCM_24")
    sf.write(deg_content_path, deg_aligned, sr_ref, subtype="PCM_24")
    polqa_result = run_polqa(ref_content_path, deg_content_path, sr=sr_ref)
    peaq_result = run_peaq(ref_content_path, deg_content_path)
    visqol_result = run_visqol(ref_content_path, deg_content_path)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # --- Video analysis (compare mode) ---
    video_results = None
    video_info = None
    if input_is_video:
        print("\n--- Video analysis ---")
        video_info = get_video_info(deg_path)
        vtmpdir = tempfile.mkdtemp(prefix="amaf_vtrim_")
        try:
            valign = align_video_signals(ref_path, deg_path)
            ref_info = get_video_info(ref_path)
            ref_dur = ref_info["duration"] if ref_info else 60
            deg_start = max(0, valign["offset_seconds"])
            ref_start = max(0, -valign["offset_seconds"])

            ref_trimmed = os.path.join(vtmpdir, "ref_trimmed.mp4")
            deg_trimmed = os.path.join(vtmpdir, "deg_trimmed.mp4")
            trim_video(ref_path, ref_start, ref_dur, ref_trimmed)
            trim_video(deg_path, deg_start, ref_dur, deg_trimmed)

            video_results = run_video_metrics(ref_trimmed, deg_trimmed)
            if video_results:
                video_results["alignment"] = valign
                video_results["info"] = video_info
        except Exception as e:
            print(f"  Video analysis failed: {e}")
        finally:
            shutil.rmtree(vtmpdir, ignore_errors=True)

    # Clean up extracted audio temps
    for tmp in audio_tmps:
        if os.path.exists(tmp):
            os.remove(tmp)

    # Generate PDF report
    if report_path:
        generate_report(
            ref=ref_aligned, deg=deg_aligned, sr=sr_ref,
            spec_results=spec, snr_results=snr,
            pesq_results=pesq_result, polqa_results=polqa_result,
            peaq_results=peaq_result,
            visqol_results=visqol_result,
            captured_path=deg_path, alignment_info=alignment_info,
            output_path=report_path,
            video_results=video_results,
            label=label,
        )

    # Generate web plots
    if plots_dir:
        _generate_plots(
            ref=ref_aligned, deg=deg_aligned, sr=sr_ref,
            spec_results=spec, snr_results=snr,
            pesq_results=pesq_result,
            output_dir=plots_dir,
            video_results=video_results,
        )

    return {
        "alignment": alignment_info,
        "duration_s": len(ref_aligned) / sr_ref,
        "normalized": normalize,
        "gain_db": gain_db,
        "spectral": spec,
        "snr": snr,
        "pesq": pesq_result,
        "polqa": polqa_result,
        "peaq": peaq_result,
        "visqol": visqol_result,
        "video": video_results,
        "video_info": video_info,
        "has_video": input_is_video,
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
    polqa_result = results["polqa"]
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
    if polqa_result and "mos_lqo" in polqa_result:
        print(f"  POLQA MOS-LQO:     {polqa_result['mos_lqo']:.3f}")
    if peaq_result and peaq_result.get("odg") is not None:
        print(f"  PEAQ ODG:          {peaq_result['odg']}")
    if visqol_result and "moslqo" in visqol_result:
        print(f"  ViSQOL MOS-LQO:    {visqol_result['moslqo']:.3f}")

    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
