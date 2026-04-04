"""
AMAF — Video analysis module.

Provides video detection, audio extraction, luminance-chirp sync,
frame-accurate alignment, and quality metrics (VMAF, PSNR, SSIM)
via ffmpeg.
"""

import json
import os
import subprocess
import tempfile
import shutil

import numpy as np
from scipy import signal as sig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts"}
AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus", ".aac"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

LUMA_CHIRP_DURATION = 3.0  # seconds
LUMA_CHIRP_F0 = 0.5        # Hz start
LUMA_CHIRP_F1 = 4.0        # Hz end


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def is_video_file(path):
    """Check if file extension is a known video format."""
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def has_video_stream(path):
    """Check whether the file actually contains a video stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return "video" in r.stdout
    except Exception:
        return False


def has_audio_stream(path):
    """Check whether the file contains an audio stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in r.stdout
    except Exception:
        return False


def get_video_info(path):
    """Get video stream metadata: width, height, fps, codec, duration."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries",
             "stream=width,height,r_frame_rate,codec_name,duration:"
             "format=duration",
             "-select_streams", "v:0", path],
            capture_output=True, text=True, timeout=10,
        )
        info = json.loads(r.stdout)
        stream = info.get("streams", [{}])[0]
        fmt = info.get("format", {})
        # Parse r_frame_rate "30/1" or "30000/1001"
        rfr = stream.get("r_frame_rate", "30/1")
        num, den = (int(x) for x in rfr.split("/"))
        fps = num / den if den else 30.0
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": fps,
            "codec_name": stream.get("codec_name", ""),
            "duration": float(stream.get("duration", 0)
                              or fmt.get("duration", 0)),
        }
    except Exception as e:
        print(f"  get_video_info: error ({e})")
        return None


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path, output_wav, sr=44100):
    """Extract audio track from a video file to WAV."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-acodec", "pcm_s24le", "-ar", str(sr), output_wav],
        capture_output=True, check=True, timeout=300,
    )
    return output_wav


# ---------------------------------------------------------------------------
# Luminance chirp — generation and extraction
# ---------------------------------------------------------------------------

def generate_luma_chirp(duration=LUMA_CHIRP_DURATION, fps=30,
                        f0=LUMA_CHIRP_F0, f1=LUMA_CHIRP_F1):
    """Generate per-frame luminance values following a chirp pattern.

    Returns uint8 array of shape (n_frames,) with values 0-255.
    """
    n_frames = int(duration * fps)
    t = np.arange(n_frames) / fps
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    luma = (128 + 127 * np.sin(phase)).astype(np.uint8)
    return luma


def extract_frame_luminance(video_path, max_seconds=None):
    """Extract per-frame average luminance by scaling to 1x1 gray.

    Returns (luminance_array_uint8, fps).
    """
    info = get_video_info(video_path)
    if not info:
        raise RuntimeError(f"Cannot read video info from {video_path}")
    fps = info["fps"]

    cmd = ["ffmpeg", "-i", video_path]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-vf", "scale=1:1,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]

    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg luminance extraction failed: {r.stderr[:200]}")

    luma = np.frombuffer(r.stdout, dtype=np.uint8)
    return luma, fps


# ---------------------------------------------------------------------------
# Video alignment
# ---------------------------------------------------------------------------

def align_video_chirp(video_path, chirp_duration=LUMA_CHIRP_DURATION,
                      f0=LUMA_CHIRP_F0, f1=LUMA_CHIRP_F1):
    """Align a processed video to the reference using the luminance chirp.

    Returns dict with offset_frames, offset_seconds, confidence, fps.
    """
    # Extract luminance (first 60s max)
    luma, fps = extract_frame_luminance(video_path, max_seconds=60)
    chirp = generate_luma_chirp(chirp_duration, fps, f0, f1)

    # Remove DC for better correlation
    luma_f = luma.astype(np.float64) - luma.mean()
    chirp_f = chirp.astype(np.float64) - chirp.mean()

    corr = sig.fftconvolve(luma_f, chirp_f[::-1], mode="full")
    peak_idx = np.argmax(np.abs(corr))
    chirp_len = len(chirp_f)
    offset_frames = peak_idx - chirp_len + 1

    peak_val = np.abs(corr[peak_idx])
    median_val = np.median(np.abs(corr))
    confidence = peak_val / median_val if median_val > 0 else float("inf")

    print(f"Video alignment: chirp found at frame {offset_frames} "
          f"({offset_frames / fps:.3f}s), confidence {confidence:.1f}x")
    if confidence < 5:
        print("WARNING: Low video alignment confidence")

    return {
        "offset_frames": int(offset_frames),
        "offset_seconds": offset_frames / fps,
        "confidence": float(confidence),
        "fps": fps,
    }


def align_video_signals(ref_path, deg_path):
    """Align two arbitrary videos via luminance cross-correlation (compare mode).

    Returns dict with offset_frames, offset_seconds, confidence, fps.
    """
    ref_luma, ref_fps = extract_frame_luminance(ref_path, max_seconds=60)
    deg_luma, deg_fps = extract_frame_luminance(deg_path, max_seconds=60)

    # Use the reference fps as the canonical one
    fps = ref_fps

    # Remove DC
    ref_f = ref_luma.astype(np.float64) - ref_luma.mean()
    deg_f = deg_luma.astype(np.float64) - deg_luma.mean()

    # Use first 60s of ref as template
    template_len = min(len(ref_f), int(fps * 60))
    template = ref_f[:template_len]
    search_len = min(len(deg_f), int(fps * 60) + template_len)
    deg_search = deg_f[:search_len]

    corr = sig.fftconvolve(deg_search, template[::-1], mode="full")
    peak_idx = np.argmax(np.abs(corr))
    offset_frames = peak_idx - template_len + 1

    peak_val = np.abs(corr[peak_idx])
    median_val = np.median(np.abs(corr))
    confidence = peak_val / median_val if median_val > 0 else float("inf")

    print(f"Video alignment: offset {offset_frames} frames "
          f"({offset_frames / fps:.3f}s), confidence {confidence:.1f}x")

    return {
        "offset_frames": int(offset_frames),
        "offset_seconds": offset_frames / fps,
        "confidence": float(confidence),
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Video trimming
# ---------------------------------------------------------------------------

def trim_video(input_path, start_s, duration_s, output_path):
    """Frame-accurate video trim using re-encode."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"trim=start={start_s}:duration={duration_s},setpts=PTS-STARTPTS",
        "-an", "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=600)
    return output_path


# ---------------------------------------------------------------------------
# Video quality metrics via ffmpeg
# ---------------------------------------------------------------------------

_libvmaf_available = None


def check_libvmaf():
    """Check if ffmpeg has the libvmaf filter."""
    global _libvmaf_available
    if _libvmaf_available is not None:
        return _libvmaf_available
    try:
        r = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10,
        )
        _libvmaf_available = "libvmaf" in r.stdout and "vmafmotion" not in r.stdout.split("libvmaf")[0][-20:]
        # More reliable: just check for the word "libvmaf" as a filter name
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "libvmaf":
                _libvmaf_available = True
                return True
        _libvmaf_available = False
    except Exception:
        _libvmaf_available = False
    return _libvmaf_available


def run_vmaf(ref_path, deg_path):
    """Run VMAF via ffmpeg's libvmaf filter. Returns dict or None."""
    if not check_libvmaf():
        print("  VMAF: not available (ffmpeg not built with --enable-libvmaf)")
        return None

    ref_info = get_video_info(ref_path)
    if not ref_info:
        return None
    w, h = ref_info["width"], ref_info["height"]

    tmpdir = tempfile.mkdtemp(prefix="amaf_vmaf_")
    log_path = os.path.join(tmpdir, "vmaf.json")

    try:
        filtergraph = (
            f"[0:v]scale={w}:{h}:flags=bicubic[deg];"
            f"[1:v]scale={w}:{h}:flags=bicubic[ref];"
            f"[deg][ref]libvmaf=log_fmt=json:log_path={log_path}"
            f":feature=name=psnr:feature=name=float_ssim"
        )
        cmd = [
            "ffmpeg", "-i", deg_path, "-i", ref_path,
            "-lavfi", filtergraph, "-f", "null", "-",
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=1800)

        with open(log_path) as f:
            data = json.load(f)

        pooled = data.get("pooled_metrics", {})
        vmaf_mean = pooled.get("vmaf", {}).get("mean")
        vmaf_hm = pooled.get("vmaf", {}).get("harmonic_mean")
        psnr_y = pooled.get("psnr_y", {}).get("mean")
        ssim = pooled.get("float_ssim", {}).get("mean")

        per_frame = []
        for frame in data.get("frames", []):
            m = frame.get("metrics", {})
            per_frame.append({
                "vmaf": m.get("vmaf"),
                "psnr_y": m.get("psnr_y"),
                "ssim": m.get("float_ssim"),
            })

        return {
            "vmaf_mean": vmaf_mean,
            "vmaf_harmonic_mean": vmaf_hm,
            "psnr_avg": psnr_y,
            "ssim_avg": ssim,
            "per_frame": per_frame,
        }
    except subprocess.CalledProcessError as e:
        print(f"  VMAF: ffmpeg error ({e.stderr[:200] if e.stderr else e})")
        return None
    except Exception as e:
        print(f"  VMAF: error ({e})")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_psnr(ref_path, deg_path):
    """Run PSNR via ffmpeg's psnr filter (fallback)."""
    ref_info = get_video_info(ref_path)
    if not ref_info:
        return None
    w, h = ref_info["width"], ref_info["height"]

    tmpdir = tempfile.mkdtemp(prefix="amaf_psnr_")
    stats_path = os.path.join(tmpdir, "psnr.log")

    try:
        filtergraph = (
            f"[0:v]scale={w}:{h}:flags=bicubic[deg];"
            f"[1:v]scale={w}:{h}:flags=bicubic[ref];"
            f"[deg][ref]psnr=stats_file={stats_path}"
        )
        r = subprocess.run(
            ["ffmpeg", "-i", deg_path, "-i", ref_path,
             "-lavfi", filtergraph, "-f", "null", "-"],
            capture_output=True, text=True, timeout=600,
        )
        # Parse average from stderr: "PSNR ... average:XX.XX"
        psnr_avg = None
        for line in r.stderr.splitlines():
            if "average:" in line and "PSNR" in line:
                for part in line.split():
                    if part.startswith("average:"):
                        psnr_avg = float(part.split(":")[1])

        # Parse per-frame from stats file
        per_frame = []
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                for line in f:
                    parts = dict(p.split(":") for p in line.strip().split()
                                 if ":" in p)
                    if "psnr_avg" in parts:
                        per_frame.append(float(parts["psnr_avg"]))

        if psnr_avg is None and per_frame:
            psnr_avg = np.mean(per_frame)

        return {"psnr_avg": psnr_avg, "per_frame": per_frame}
    except Exception as e:
        print(f"  PSNR: error ({e})")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_ssim(ref_path, deg_path):
    """Run SSIM via ffmpeg's ssim filter (fallback)."""
    ref_info = get_video_info(ref_path)
    if not ref_info:
        return None
    w, h = ref_info["width"], ref_info["height"]

    tmpdir = tempfile.mkdtemp(prefix="amaf_ssim_")
    stats_path = os.path.join(tmpdir, "ssim.log")

    try:
        filtergraph = (
            f"[0:v]scale={w}:{h}:flags=bicubic[deg];"
            f"[1:v]scale={w}:{h}:flags=bicubic[ref];"
            f"[deg][ref]ssim=stats_file={stats_path}"
        )
        r = subprocess.run(
            ["ffmpeg", "-i", deg_path, "-i", ref_path,
             "-lavfi", filtergraph, "-f", "null", "-"],
            capture_output=True, text=True, timeout=600,
        )
        # Parse from stderr: "SSIM ... All:X.XXXX"
        ssim_avg = None
        for line in r.stderr.splitlines():
            if "All:" in line and "SSIM" in line:
                for part in line.split():
                    if part.startswith("All:"):
                        ssim_avg = float(part.split(":")[1])

        per_frame = []
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                for line in f:
                    parts = dict(p.split(":") for p in line.strip().split()
                                 if ":" in p)
                    if "All" in parts:
                        per_frame.append(float(parts["All"]))

        if ssim_avg is None and per_frame:
            ssim_avg = np.mean(per_frame)

        return {"ssim_avg": ssim_avg, "per_frame": per_frame}
    except Exception as e:
        print(f"  SSIM: error ({e})")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_video_metrics(ref_path, deg_path):
    """Run video quality metrics. Tries VMAF first, falls back to PSNR+SSIM."""
    vmaf = run_vmaf(ref_path, deg_path)
    if vmaf:
        return {
            "vmaf_score": vmaf["vmaf_mean"],
            "vmaf_harmonic_mean": vmaf.get("vmaf_harmonic_mean"),
            "psnr_avg": vmaf.get("psnr_avg"),
            "ssim_avg": vmaf.get("ssim_avg"),
            "per_frame": vmaf.get("per_frame"),
            "vmaf_available": True,
        }

    # Fallback: separate PSNR and SSIM
    print("  VMAF unavailable, falling back to PSNR + SSIM")
    psnr = run_psnr(ref_path, deg_path)
    ssim = run_ssim(ref_path, deg_path)

    return {
        "vmaf_score": None,
        "vmaf_harmonic_mean": None,
        "psnr_avg": psnr["psnr_avg"] if psnr else None,
        "ssim_avg": ssim["ssim_avg"] if ssim else None,
        "per_frame": None,
        "vmaf_available": False,
    }


# ---------------------------------------------------------------------------
# Reference video generation
# ---------------------------------------------------------------------------

def build_reference_video(track_nums, resolution="1920x1080", fps=30,
                          output_path=None):
    """Build a reference video with luminance chirp sync + test pattern.

    Audio track is the standard audio reference (chirp + SQAM tracks).
    Returns (output_path, total_duration).
    """
    from generate_reference import build_reference

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if output_path is None:
        output_path = os.path.join(base_dir, "reference_video.mp4")

    w, h = resolution.split("x")
    tmpdir = tempfile.mkdtemp(prefix="amaf_vidref_")

    try:
        # 1. Build audio reference
        audio_path = os.path.join(tmpdir, "audio.wav")
        build_reference(track_nums, output_path=audio_path)

        # Get audio duration to size the video
        import soundfile as sf
        audio_data, sr = sf.read(audio_path, dtype="float64")
        audio_duration = len(audio_data) / sr

        content_duration = audio_duration - LUMA_CHIRP_DURATION
        if content_duration < 1:
            content_duration = 10  # minimum

        # 2. Generate luminance chirp video segment
        chirp_luma = generate_luma_chirp(LUMA_CHIRP_DURATION, fps)
        chirp_raw = chirp_luma.tobytes()

        chirp_path = os.path.join(tmpdir, "chirp.mp4")
        proc = subprocess.Popen(
            ["ffmpeg", "-y",
             "-f", "rawvideo", "-pix_fmt", "gray", "-s", "1x1",
             "-r", str(fps), "-i", "pipe:0",
             "-vf", f"scale={w}:{h}:flags=neighbor,format=yuv420p",
             "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
             chirp_path],
            stdin=subprocess.PIPE, capture_output=True, timeout=60,
        )
        proc.stdin.write(chirp_raw)
        proc.stdin.close()
        proc.wait(timeout=60)

        # 3. Generate test pattern content
        content_path = os.path.join(tmpdir, "content.mp4")
        subprocess.run(
            ["ffmpeg", "-y",
             "-f", "lavfi", "-i",
             f"testsrc2=size={w}x{h}:rate={fps}:duration={content_duration}",
             "-c:v", "libx264", "-crf", "18", "-preset", "medium",
             content_path],
            capture_output=True, check=True, timeout=300,
        )

        # 4. Concatenate chirp + content
        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            f.write(f"file '{chirp_path}'\n")
            f.write(f"file '{content_path}'\n")

        video_only = os.path.join(tmpdir, "video_only.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list,
             "-c:v", "libx264", "-crf", "18", "-preset", "medium",
             video_only],
            capture_output=True, check=True, timeout=300,
        )

        # 5. Mux video + audio
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", video_only, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", output_path],
            capture_output=True, check=True, timeout=300,
        )

        total_duration = min(audio_duration,
                             LUMA_CHIRP_DURATION + content_duration)
        return output_path, total_duration

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
