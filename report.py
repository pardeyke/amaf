"""
PDF report generation for AMAF measurements.
"""

import os
from datetime import datetime
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


# Consistent style
COLORS = {
    "ref": "#2196F3",
    "deg": "#FF5722",
    "diff": "#9C27B0",
    "good": "#4CAF50",
    "warn": "#FF9800",
    "bad": "#F44336",
    "bg": "#FAFAFA",
    "grid": "#E0E0E0",
    "text": "#212121",
    "muted": "#757575",
}

def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["grid"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.5,
        "figure.facecolor": "white",
        "text.color": COLORS["text"],
    })


def _score_color(value, good, bad):
    """Return color based on value quality."""
    if value >= good:
        return COLORS["good"]
    elif value >= bad:
        return COLORS["warn"]
    return COLORS["bad"]


def to_mono(audio):
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def generate_report(
    ref, deg, sr,
    spec_results, snr_results, pesq_results,
    polqa_results=None,
    peaq_results=None, visqol_results=None,
    captured_path="", alignment_info=None,
    output_path="report.pdf",
    video_results=None,
    label=None,
):
    _setup_style()

    ref_m = to_mono(ref)
    deg_m = to_mono(deg)
    diff_m = ref_m - deg_m
    t = np.arange(len(ref_m)) / sr

    with PdfPages(output_path) as pdf:
        # ===================================================================
        # PAGE 1: Title + Summary
        # ===================================================================
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3,
                      left=0.08, right=0.92, top=0.92, bottom=0.06)

        # Title
        fig.text(0.5, 0.97, "AMAF — Audio Quality Report",
                 ha="center", va="top", fontsize=18, fontweight="bold",
                 color=COLORS["text"])
        if label:
            fig.text(0.5, 0.945, label,
                     ha="center", va="top", fontsize=11, color=COLORS["text"])
            fig.text(0.5, 0.925,
                     f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                     ha="center", va="top", fontsize=9, color=COLORS["muted"])
        else:
            fig.text(0.5, 0.945,
                     f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                     ha="center", va="top", fontsize=9, color=COLORS["muted"])

        # --- Summary metrics panel ---
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.set_xlim(0, 10)
        ax_summary.set_ylim(0, 3.5)
        ax_summary.axis("off")

        # File info
        ax_summary.text(0, 3.2, "Source:", fontsize=8, color=COLORS["muted"])
        ax_summary.text(1.5, 3.2, os.path.basename(captured_path),
                        fontsize=8, fontweight="bold")
        ax_summary.text(0, 2.7, "Duration:", fontsize=8, color=COLORS["muted"])
        ax_summary.text(1.5, 2.7, f"{len(ref_m)/sr:.1f}s analysed", fontsize=8)
        ax_summary.text(5, 2.7, "Alignment confidence:", fontsize=8,
                        color=COLORS["muted"])
        ax_summary.text(7, 2.7, f"{alignment_info['confidence']:.0f}x",
                        fontsize=8, fontweight="bold")

        # Big metric boxes
        metrics_y = 0.3
        box_w = 1.8
        metrics = []

        metrics.append(("SNR", f"{snr_results['snr_db']:.1f} dB",
                        _score_color(snr_results["snr_db"], 30, 20)))
        metrics.append(("Null Depth", f"{spec_results['null_depth_db']:.1f} dB",
                        _score_color(-spec_results["null_depth_db"], 30, 20)))
        metrics.append(("THD+N", f"{snr_results['thd_n_pct']:.2f}%",
                        _score_color(-snr_results["thd_n_pct"], -3, -10)))
        if pesq_results:
            metrics.append(("PESQ MOS", f"{pesq_results['mos_lqo_mean']:.2f}",
                            _score_color(pesq_results["mos_lqo_mean"], 4.0, 3.5)))
        if polqa_results and "mos_lqo" in polqa_results:
            metrics.append(("POLQA MOS", f"{polqa_results['mos_lqo']:.2f}",
                            _score_color(polqa_results["mos_lqo"], 4.0, 3.5)))
        if peaq_results and peaq_results.get("odg") is not None:
            metrics.append(("PEAQ ODG", f"{peaq_results['odg']:.2f}",
                            _score_color(peaq_results["odg"], -1, -2)))
        if visqol_results and "moslqo" in visqol_results:
            metrics.append(("ViSQOL", f"{visqol_results['moslqo']:.2f}",
                            _score_color(visqol_results["moslqo"], 4.0, 3.5)))

        n_metrics = len(metrics)
        total_w = n_metrics * box_w + (n_metrics - 1) * 0.2
        start_x = (10 - total_w) / 2

        for i, (label, value, color) in enumerate(metrics):
            x = start_x + i * (box_w + 0.2)
            rect = plt.Rectangle((x, metrics_y), box_w, 1.8,
                                  facecolor=color, alpha=0.12,
                                  edgecolor=color, linewidth=1.5,
                                  transform=ax_summary.transData)
            ax_summary.add_patch(rect)
            ax_summary.text(x + box_w / 2, metrics_y + 1.25, value,
                           ha="center", va="center", fontsize=14,
                           fontweight="bold", color=color)
            ax_summary.text(x + box_w / 2, metrics_y + 0.35, label,
                           ha="center", va="center", fontsize=8,
                           color=COLORS["muted"])

        # --- Waveform comparison ---
        ax_wave = fig.add_subplot(gs[1, :])
        # Show a 2-second excerpt from the middle of the audio
        mid = len(ref_m) // 2
        excerpt = int(2.0 * sr)
        s, e = mid - excerpt // 2, mid + excerpt // 2
        t_ex = t[s:e] - t[s]
        ax_wave.plot(t_ex, ref_m[s:e], color=COLORS["ref"], alpha=0.7,
                     linewidth=0.4, label="Reference")
        ax_wave.plot(t_ex, deg_m[s:e], color=COLORS["deg"], alpha=0.7,
                     linewidth=0.4, label="Processed")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_title("Waveform Comparison (2s excerpt)", fontsize=10,
                          fontweight="bold")
        ax_wave.legend(loc="upper right", fontsize=7)
        ax_wave.set_xlim(t_ex[0], t_ex[-1])

        # --- Null test waveform ---
        ax_null = fig.add_subplot(gs[2, :])
        # Full difference signal envelope
        env_sr = 100  # envelope at 100 Hz
        env_hop = sr // env_sr
        n_env = len(diff_m) // env_hop
        envelope = np.array([
            np.sqrt(np.mean(diff_m[i * env_hop:(i + 1) * env_hop] ** 2))
            for i in range(n_env)
        ])
        t_env = np.arange(n_env) / env_sr
        with np.errstate(divide="ignore"):
            envelope_db = 20 * np.log10(np.maximum(envelope, 1e-10))
        ax_null.plot(t_env, envelope_db, color=COLORS["diff"], linewidth=0.6)
        ax_null.set_xlabel("Time (s)")
        ax_null.set_ylabel("Level (dBFS)")
        ax_null.set_title("Null Test — Difference Signal Envelope", fontsize=10,
                          fontweight="bold")
        ax_null.set_xlim(0, t_env[-1])
        ax_null.set_ylim(-80, 0)
        ax_null.axhline(y=spec_results["null_depth_db"], color=COLORS["muted"],
                        linestyle="--", linewidth=0.8, alpha=0.7,
                        label=f"RMS null depth: {spec_results['null_depth_db']:.1f} dB")
        ax_null.legend(loc="upper right", fontsize=7)

        # --- Spectral difference ---
        ax_spec = fig.add_subplot(gs[3, :])
        n_fft = 8192
        freqs = rfftfreq(n_fft, 1.0 / sr)
        hop = n_fft // 2
        n_segments = max(1, (len(ref_m) - n_fft) // hop)
        window = np.hanning(n_fft)
        ref_mag = np.zeros(n_fft // 2 + 1)
        deg_mag = np.zeros(n_fft // 2 + 1)
        for i in range(n_segments):
            start = i * hop
            ref_mag += np.abs(rfft(ref_m[start:start + n_fft] * window))
            deg_mag += np.abs(rfft(deg_m[start:start + n_fft] * window))
        ref_mag /= n_segments
        deg_mag /= n_segments
        with np.errstate(divide="ignore", invalid="ignore"):
            diff_db = 20 * np.log10(
                np.where(ref_mag > 0, deg_mag / ref_mag, 1.0))
        audible = freqs >= 20
        ax_spec.semilogx(freqs[audible], diff_db[audible],
                         color=COLORS["diff"], linewidth=0.6)
        ax_spec.axhline(y=0, color=COLORS["muted"], linewidth=0.5)
        ax_spec.fill_between(freqs[audible], 0, diff_db[audible],
                             alpha=0.15, color=COLORS["diff"])
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("Difference (dB)")
        ax_spec.set_title("Spectral Difference (Processed vs Reference)",
                          fontsize=10, fontweight="bold")
        ax_spec.set_xlim(20, 22050)
        ax_spec.set_ylim(-40, 20)
        # Band annotations
        for label, lo, hi, val in [
            ("Low", 20, 200, spec_results["band_diffs_db"]["Low (20-200 Hz)"]),
            ("Mid", 200, 4000, spec_results["band_diffs_db"]["Mid (200-4000 Hz)"]),
            ("High", 4000, 20000, spec_results["band_diffs_db"]["High (4000-20000 Hz)"]),
        ]:
            cx = np.sqrt(lo * hi)
            ax_spec.text(cx, -37, f"{label}\n{val:.1f} dB",
                        ha="center", fontsize=7, color=COLORS["muted"])

        # --- Spectrograms ---
        ax_sg_ref = fig.add_subplot(gs[4, 0])
        ax_sg_deg = fig.add_subplot(gs[4, 1])
        sg_nfft = 4096
        for ax, data, title, cmap in [
            (ax_sg_ref, ref_m, "Reference", "inferno"),
            (ax_sg_deg, deg_m, "Processed", "inferno"),
        ]:
            ax.specgram(data, NFFT=sg_nfft, Fs=sr, noverlap=sg_nfft // 2,
                       cmap=cmap, vmin=-90, vmax=0, scale="dB")
            ax.set_ylabel("Freq (Hz)")
            ax.set_xlabel("Time (s)")
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_ylim(0, 22050)

        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================================
        # PAGE 2: Detailed metrics
        # ===================================================================
        fig2 = plt.figure(figsize=(8.27, 11.69))
        gs2 = GridSpec(4, 2, figure=fig2, hspace=0.45, wspace=0.35,
                       left=0.1, right=0.92, top=0.94, bottom=0.06)

        fig2.text(0.5, 0.97, "Detailed Analysis",
                  ha="center", va="top", fontsize=14, fontweight="bold",
                  color=COLORS["text"])

        # --- Segmental SNR over time ---
        ax_seg = fig2.add_subplot(gs2[0, :])
        seg_len = sr
        n_segs = len(ref_m) // seg_len
        seg_snrs = []
        seg_times = []
        for i in range(n_segs):
            s = i * seg_len
            e = s + seg_len
            p_sig = np.mean(ref_m[s:e] ** 2)
            p_noise = np.mean(diff_m[s:e] ** 2)
            if p_sig < 1e-8:
                seg_snrs.append(np.nan)
            elif p_noise > 0:
                seg_snrs.append(10 * np.log10(p_sig / p_noise))
            else:
                seg_snrs.append(60)
            seg_times.append((s + seg_len / 2) / sr)

        seg_snrs = np.array(seg_snrs)
        seg_times = np.array(seg_times)
        valid = ~np.isnan(seg_snrs)
        colors_seg = [_score_color(v, 30, 20) if not np.isnan(v)
                      else COLORS["grid"] for v in seg_snrs]
        ax_seg.bar(seg_times, np.where(valid, seg_snrs, 0),
                   width=0.9, color=colors_seg, alpha=0.8)
        ax_seg.axhline(y=snr_results["segmental_snr_db_mean"],
                       color=COLORS["muted"], linestyle="--", linewidth=0.8,
                       label=f"Mean: {snr_results['segmental_snr_db_mean']:.1f} dB")
        ax_seg.set_xlabel("Time (s)")
        ax_seg.set_ylabel("SNR (dB)")
        ax_seg.set_title("Segmental SNR (1s windows)", fontsize=10,
                          fontweight="bold")
        ax_seg.legend(loc="upper right", fontsize=7)
        ax_seg.set_xlim(0, len(ref_m) / sr)

        # --- PESQ per chunk ---
        if pesq_results and pesq_results.get("chunk_scores"):
            ax_pesq = fig2.add_subplot(gs2[1, 0])
            chunks = pesq_results["chunk_scores"]
            chunk_x = np.arange(1, len(chunks) + 1)
            chunk_colors = [_score_color(v, 4.0, 3.5) for v in chunks]
            ax_pesq.bar(chunk_x, chunks, color=chunk_colors, alpha=0.8)
            ax_pesq.axhline(y=pesq_results["mos_lqo_mean"],
                           color=COLORS["muted"], linestyle="--", linewidth=0.8)
            ax_pesq.set_xlabel("Chunk (30s)")
            ax_pesq.set_ylabel("MOS-LQO")
            ax_pesq.set_title("PESQ per Chunk", fontsize=10, fontweight="bold")
            ax_pesq.set_ylim(1, 5)
        else:
            ax_pesq = fig2.add_subplot(gs2[1, 0])
            ax_pesq.text(0.5, 0.5, "PESQ chunk data\nnot available",
                        ha="center", va="center", transform=ax_pesq.transAxes,
                        fontsize=10, color=COLORS["muted"])
            ax_pesq.set_title("PESQ per Chunk", fontsize=10, fontweight="bold")

        # --- Band difference bar chart ---
        ax_band = fig2.add_subplot(gs2[1, 1])
        bands = spec_results["band_diffs_db"]
        band_labels = ["Low\n20-200", "Mid\n200-4k", "High\n4k-20k"]
        band_vals = list(bands.values())
        band_colors = [_score_color(-v, -1, -5) for v in band_vals]
        ax_band.bar(band_labels, band_vals, color=band_colors, alpha=0.8,
                    width=0.5)
        ax_band.set_ylabel("Mean |diff| (dB)")
        ax_band.set_title("Spectral Difference by Band", fontsize=10,
                          fontweight="bold")
        for i, v in enumerate(band_vals):
            ax_band.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=8)

        # --- Magnitude spectra overlay ---
        ax_mag = fig2.add_subplot(gs2[2, :])
        with np.errstate(divide="ignore"):
            ref_mag_db = 20 * np.log10(np.maximum(ref_mag, 1e-10))
            deg_mag_db = 20 * np.log10(np.maximum(deg_mag, 1e-10))
        ax_mag.semilogx(freqs[audible], ref_mag_db[audible],
                        color=COLORS["ref"], linewidth=0.6, alpha=0.8,
                        label="Reference")
        ax_mag.semilogx(freqs[audible], deg_mag_db[audible],
                        color=COLORS["deg"], linewidth=0.6, alpha=0.8,
                        label="Processed")
        ax_mag.set_xlabel("Frequency (Hz)")
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title("Average Magnitude Spectrum", fontsize=10,
                          fontweight="bold")
        ax_mag.legend(loc="upper right", fontsize=7)
        ax_mag.set_xlim(20, 22050)

        # --- Difference spectrogram ---
        ax_sg_diff = fig2.add_subplot(gs2[3, :])
        ax_sg_diff.specgram(diff_m, NFFT=sg_nfft, Fs=sr,
                            noverlap=sg_nfft // 2,
                            cmap="magma", vmin=-90, vmax=-20, scale="dB")
        ax_sg_diff.set_ylabel("Freq (Hz)")
        ax_sg_diff.set_xlabel("Time (s)")
        ax_sg_diff.set_title("Difference Signal Spectrogram", fontsize=10,
                              fontweight="bold")
        ax_sg_diff.set_ylim(0, 22050)

        pdf.savefig(fig2)
        plt.close(fig2)

        # ===================================================================
        # PAGE 3: Pipeline info + metrics table
        # ===================================================================
        fig3 = plt.figure(figsize=(8.27, 11.69))
        fig3.text(0.5, 0.97, "Measurement Summary",
                  ha="center", va="top", fontsize=14, fontweight="bold",
                  color=COLORS["text"])

        ax_table = fig3.add_axes([0.1, 0.15, 0.8, 0.75])
        ax_table.axis("off")

        rows = [
            ["Alignment Confidence", f"{alignment_info['confidence']:.0f}x"],
            ["Analysed Duration", f"{len(ref_m)/sr:.1f}s"],
            ["", ""],
            ["SNR (overall)", f"{snr_results['snr_db']:.1f} dB"],
            ["Segmental SNR (mean)", f"{snr_results['segmental_snr_db_mean']:.1f} dB"],
            ["Segmental SNR (min)", f"{snr_results['segmental_snr_db_min']:.1f} dB"],
            ["THD+N", f"{snr_results['thd_n_db']:.1f} dB ({snr_results['thd_n_pct']:.2f}%)"],
            ["", ""],
            ["Null Depth (RMS)", f"{spec_results['null_depth_db']:.1f} dB"],
            ["Spectral Diff (mean)", f"{spec_results['mean_spectral_diff_db']:.2f} dB"],
            ["Spectral Diff (max)", f"{spec_results['max_spectral_diff_db']:.2f} dB"],
            ["  Low (20-200 Hz)", f"{spec_results['band_diffs_db']['Low (20-200 Hz)']:.2f} dB"],
            ["  Mid (200-4k Hz)", f"{spec_results['band_diffs_db']['Mid (200-4000 Hz)']:.2f} dB"],
            ["  High (4k-20k Hz)", f"{spec_results['band_diffs_db']['High (4000-20000 Hz)']:.2f} dB"],
        ]
        if pesq_results:
            rows.append(["", ""])
            rows.append(["PESQ MOS-LQO (mean)", f"{pesq_results['mos_lqo_mean']:.3f}"])
            rows.append(["PESQ MOS-LQO (min)", f"{pesq_results['mos_lqo_min']:.3f}"])
            rows.append(["PESQ MOS-LQO (max)", f"{pesq_results['mos_lqo_max']:.3f}"])
        if polqa_results and "mos_lqo" in polqa_results:
            rows.append(["", ""])
            rows.append(["POLQA MOS-LQO (P.863)", f"{polqa_results['mos_lqo']:.3f}"])
        if peaq_results and peaq_results.get("odg") is not None:
            rows.append(["", ""])
            rows.append(["PEAQ ODG", f"{peaq_results['odg']:.2f}"])
            rows.append(["PEAQ DI", f"{peaq_results['di']:.2f}"])
        if visqol_results and "moslqo" in visqol_results:
            rows.append(["", ""])
            rows.append(["ViSQOL MOS-LQO", f"{visqol_results['moslqo']:.3f}"])

        table = ax_table.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            colWidths=[0.55, 0.45],
            loc="upper center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        # Style the table
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(COLORS["grid"])
            if row == 0:
                cell.set_facecolor(COLORS["text"])
                cell.set_text_props(color="white", fontweight="bold")
            elif rows[row - 1][0] == "":
                cell.set_facecolor("white")
                cell.set_edgecolor("white")
            elif row % 2 == 0:
                cell.set_facecolor("#F5F5F5")
            else:
                cell.set_facecolor("white")

        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Web plots — individual high-res PNGs with dark theme
# ---------------------------------------------------------------------------

DARK = {
    "bg": "#1a1d27", "surface": "#242836", "border": "#2e3345",
    "text": "#e4e6f0", "muted": "#8b8fa3", "grid": "#2e3345",
    "ref": "#6c8aff", "deg": "#ff6b6b", "diff": "#c084fc",
    "good": "#4caf50", "warn": "#ff9800", "bad": "#f44336",
}


def _dark_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.facecolor": DARK["surface"],
        "axes.edgecolor": DARK["border"],
        "axes.grid": True,
        "axes.labelcolor": DARK["text"],
        "grid.color": DARK["grid"],
        "grid.alpha": 0.4,
        "figure.facecolor": DARK["bg"],
        "text.color": DARK["text"],
        "xtick.color": DARK["muted"],
        "ytick.color": DARK["muted"],
        "legend.facecolor": DARK["surface"],
        "legend.edgecolor": DARK["border"],
        "legend.labelcolor": DARK["text"],
    })


def _dark_score_color(value, good, bad):
    if value >= good:
        return DARK["good"]
    elif value >= bad:
        return DARK["warn"]
    return DARK["bad"]


def generate_plots(
    ref, deg, sr,
    spec_results, snr_results, pesq_results,
    output_dir,
    video_results=None,
):
    """Generate individual plot PNGs for the web detail view."""
    _dark_style()
    os.makedirs(output_dir, exist_ok=True)

    ref_m = to_mono(ref)
    deg_m = to_mono(deg)
    diff_m = ref_m - deg_m
    t = np.arange(len(ref_m)) / sr
    DPI = 150

    # --- Waveform comparison ---
    fig, ax = plt.subplots(figsize=(12, 3.5))
    mid = len(ref_m) // 2
    excerpt = int(2.0 * sr)
    s, e = mid - excerpt // 2, mid + excerpt // 2
    t_ex = t[s:e] - t[s]
    ax.plot(t_ex, ref_m[s:e], color=DARK["ref"], alpha=0.8, linewidth=0.4, label="Reference")
    ax.plot(t_ex, deg_m[s:e], color=DARK["deg"], alpha=0.8, linewidth=0.4, label="Processed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform Comparison (2s excerpt)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(t_ex[0], t_ex[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "waveform.png"), dpi=DPI)
    plt.close(fig)

    # --- Null test envelope ---
    fig, ax = plt.subplots(figsize=(12, 3.5))
    env_sr = 100
    env_hop = sr // env_sr
    n_env = len(diff_m) // env_hop
    envelope = np.array([
        np.sqrt(np.mean(diff_m[i * env_hop:(i + 1) * env_hop] ** 2))
        for i in range(n_env)
    ])
    t_env = np.arange(n_env) / env_sr
    with np.errstate(divide="ignore"):
        envelope_db = 20 * np.log10(np.maximum(envelope, 1e-10))
    ax.plot(t_env, envelope_db, color=DARK["diff"], linewidth=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Level (dBFS)")
    ax.set_title("Null Test — Difference Signal Envelope", fontweight="bold")
    ax.set_xlim(0, t_env[-1])
    ax.set_ylim(-80, 0)
    ax.axhline(y=spec_results["null_depth_db"], color=DARK["muted"],
               linestyle="--", linewidth=1, alpha=0.8,
               label=f"RMS null depth: {spec_results['null_depth_db']:.1f} dB")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "null_test.png"), dpi=DPI)
    plt.close(fig)

    # --- Spectral difference ---
    n_fft = 8192
    freqs = rfftfreq(n_fft, 1.0 / sr)
    hop = n_fft // 2
    n_segments = max(1, (len(ref_m) - n_fft) // hop)
    window = np.hanning(n_fft)
    ref_mag = np.zeros(n_fft // 2 + 1)
    deg_mag = np.zeros(n_fft // 2 + 1)
    for i in range(n_segments):
        start = i * hop
        ref_mag += np.abs(rfft(ref_m[start:start + n_fft] * window))
        deg_mag += np.abs(rfft(deg_m[start:start + n_fft] * window))
    ref_mag /= n_segments
    deg_mag /= n_segments
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_db = 20 * np.log10(np.where(ref_mag > 0, deg_mag / ref_mag, 1.0))
    audible = freqs >= 20

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogx(freqs[audible], diff_db[audible], color=DARK["diff"], linewidth=0.7)
    ax.axhline(y=0, color=DARK["muted"], linewidth=0.5)
    ax.fill_between(freqs[audible], 0, diff_db[audible], alpha=0.2, color=DARK["diff"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Difference (dB)")
    ax.set_title("Spectral Difference (Processed vs Reference)", fontweight="bold")
    ax.set_xlim(20, 22050)
    ax.set_ylim(-40, 20)
    for label, lo, hi, val in [
        ("Low", 20, 200, spec_results["band_diffs_db"]["Low (20-200 Hz)"]),
        ("Mid", 200, 4000, spec_results["band_diffs_db"]["Mid (200-4000 Hz)"]),
        ("High", 4000, 20000, spec_results["band_diffs_db"]["High (4000-20000 Hz)"]),
    ]:
        cx = np.sqrt(lo * hi)
        ax.text(cx, -37, f"{label}\n{val:.1f} dB", ha="center", fontsize=9, color=DARK["muted"])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectral_diff.png"), dpi=DPI)
    plt.close(fig)

    # --- Spectrograms (ref + deg side by side) ---
    sg_nfft = 4096
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in [(ax1, ref_m, "Reference"), (ax2, deg_m, "Processed")]:
        ax.specgram(data, NFFT=sg_nfft, Fs=sr, noverlap=sg_nfft // 2,
                    cmap="inferno", vmin=-90, vmax=0, scale="dB")
        ax.set_ylabel("Freq (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 22050)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrograms.png"), dpi=DPI)
    plt.close(fig)

    # --- Segmental SNR ---
    fig, ax = plt.subplots(figsize=(12, 3.5))
    seg_len = sr
    n_segs = len(ref_m) // seg_len
    seg_snrs = []
    seg_times = []
    for i in range(n_segs):
        ss = i * seg_len
        ee = ss + seg_len
        p_sig = np.mean(ref_m[ss:ee] ** 2)
        p_noise = np.mean(diff_m[ss:ee] ** 2)
        if p_sig < 1e-8:
            seg_snrs.append(np.nan)
        elif p_noise > 0:
            seg_snrs.append(10 * np.log10(p_sig / p_noise))
        else:
            seg_snrs.append(60)
        seg_times.append((ss + seg_len / 2) / sr)
    seg_snrs = np.array(seg_snrs)
    seg_times = np.array(seg_times)
    valid = ~np.isnan(seg_snrs)
    colors_seg = [_dark_score_color(v, 30, 20) if not np.isnan(v) else DARK["grid"] for v in seg_snrs]
    ax.bar(seg_times, np.where(valid, seg_snrs, 0), width=0.9, color=colors_seg, alpha=0.85)
    ax.axhline(y=snr_results["segmental_snr_db_mean"], color=DARK["muted"],
               linestyle="--", linewidth=1,
               label=f"Mean: {snr_results['segmental_snr_db_mean']:.1f} dB")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Segmental SNR (1s windows)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, len(ref_m) / sr)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "segmental_snr.png"), dpi=DPI)
    plt.close(fig)

    # --- PESQ per chunk ---
    if pesq_results and pesq_results.get("chunk_scores"):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        chunks = pesq_results["chunk_scores"]
        chunk_x = np.arange(1, len(chunks) + 1)
        chunk_colors = [_dark_score_color(v, 4.0, 3.5) for v in chunks]
        ax.bar(chunk_x, chunks, color=chunk_colors, alpha=0.85)
        ax.axhline(y=pesq_results["mos_lqo_mean"], color=DARK["muted"],
                   linestyle="--", linewidth=1,
                   label=f"Mean: {pesq_results['mos_lqo_mean']:.2f}")
        ax.set_xlabel("Chunk (30s)")
        ax.set_ylabel("MOS-LQO")
        ax.set_title("PESQ per Chunk", fontweight="bold")
        ax.set_ylim(1, 5)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "pesq_chunks.png"), dpi=DPI)
        plt.close(fig)

    # --- Band difference ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bands = spec_results["band_diffs_db"]
    band_labels = ["Low\n20-200", "Mid\n200-4k", "High\n4k-20k"]
    band_vals = list(bands.values())
    band_colors = [_dark_score_color(-v, -1, -5) for v in band_vals]
    ax.bar(band_labels, band_vals, color=band_colors, alpha=0.85, width=0.5)
    ax.set_ylabel("Mean |diff| (dB)")
    ax.set_title("Spectral Difference by Band", fontweight="bold")
    for i, v in enumerate(band_vals):
        ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=10, color=DARK["text"])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "band_diff.png"), dpi=DPI)
    plt.close(fig)

    # --- Magnitude spectrum overlay ---
    fig, ax = plt.subplots(figsize=(12, 4))
    with np.errstate(divide="ignore"):
        ref_mag_db = 20 * np.log10(np.maximum(ref_mag, 1e-10))
        deg_mag_db = 20 * np.log10(np.maximum(deg_mag, 1e-10))
    ax.semilogx(freqs[audible], ref_mag_db[audible], color=DARK["ref"],
                linewidth=0.7, alpha=0.85, label="Reference")
    ax.semilogx(freqs[audible], deg_mag_db[audible], color=DARK["deg"],
                linewidth=0.7, alpha=0.85, label="Processed")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Average Magnitude Spectrum", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(20, 22050)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "magnitude_spectrum.png"), dpi=DPI)
    plt.close(fig)

    # --- Difference spectrogram ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.specgram(diff_m, NFFT=sg_nfft, Fs=sr, noverlap=sg_nfft // 2,
                cmap="magma", vmin=-90, vmax=-20, scale="dB")
    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Difference Signal Spectrogram", fontweight="bold")
    ax.set_ylim(0, 22050)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "diff_spectrogram.png"), dpi=DPI)
    plt.close(fig)

    # --- Per-frame VMAF (if video results available) ---
    if video_results and video_results.get("per_frame"):
        per_frame = video_results["per_frame"]
        vmaf_vals = [f.get("vmaf") for f in per_frame if f.get("vmaf") is not None]
        if vmaf_vals:
            fps = video_results.get("info", {}).get("fps", 30)
            t_vid = np.arange(len(vmaf_vals)) / fps

            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.plot(t_vid, vmaf_vals, color=DARK["ref"], linewidth=0.6)
            ax.axhline(y=90, color=DARK["good"], linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axhline(y=70, color=DARK["warn"], linestyle="--", alpha=0.5, linewidth=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("VMAF Score")
            ax.set_title("Per-Frame VMAF", fontweight="bold")
            ax.set_ylim(0, 100)
            ax.set_xlim(0, t_vid[-1] if len(t_vid) > 0 else 1)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "vmaf_per_frame.png"), dpi=DPI)
            plt.close(fig)
