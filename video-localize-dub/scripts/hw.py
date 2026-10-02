"""Hardware detection + per-machine defaults (GPU / CPU / RAM / video encoder).

Every heavy step asks this module which device to use, so the same job runs on any PC:
  NVIDIA (CUDA)        -> LaMa, demucs, whisper, OCR on GPU; NVENC video encoding
  Apple Silicon (MPS)  -> LaMa, demucs on GPU; VideoToolbox encoding
  AMD / Intel on Windows -> LaMa via torch-directml if installed (experimental); AMF / QSV encoding
  anything else        -> CPU (always works)
job.json "hardware" overrides: device auto|cpu|cuda|mps|directml, encoder auto|libx264|h264_nvenc|...,
threads auto|N, remotion_concurrency auto|N.

  py -3.12 hw.py            # print what this machine has and what will be used
"""

import ctypes
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

CACHE_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "video-localize-dub"

# quality-first settings: visually equal to libx264 crf 17 at the cost of a bigger file
HW_ENCODERS = {
    "h264_nvenc": ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "{q}", "-b:v", "0"],
    "h264_videotoolbox": ["-q:v", "{vt}"],
    "h264_qsv": ["-preset", "veryslow", "-global_quality", "{q}"],
    "h264_amf": ["-quality", "quality", "-rc", "cqp", "-qp_i", "{q}", "-qp_p", "{q}", "-qp_b", "{q}"],
}


def cpu_threads():
    return os.cpu_count() or 4


def ram_gb():
    try:
        if sys.platform == "win32":
            class M(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            m = M()
            m.dwLength = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2 ** 30, m.ullAvailPhys / 2 ** 30
        tot = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        try:
            av = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 2 ** 30
        except (ValueError, OSError):
            av = tot / 2
        return tot, av
    except Exception:
        return 8.0, 4.0


@lru_cache(None)
def torch_devices():
    """Usable torch devices in preference order, e.g. ['cuda', 'cpu']."""
    out = []
    try:
        import torch
        if torch.cuda.is_available():
            out.append("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            out.append("mps")
    except Exception:
        return ["cpu"]
    try:
        import torch_directml  # noqa: F401
        out.append("directml")
    except Exception:
        pass
    return out + ["cpu"]


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name} ({p.total_memory / 2 ** 30:.0f} GB)"
        if "mps" in torch_devices():
            return "Apple GPU (MPS)"
    except Exception:
        pass
    return None


def _cfg_hw(cfg):
    return (cfg or {}).get("hardware", {}) or {}


def torch_device(cfg=None):
    """'cuda' | 'mps' | 'directml' | 'cpu' for torch models (LaMa, demucs)."""
    want = _cfg_hw(cfg).get("device", "auto")
    have = torch_devices()
    if want != "auto":
        if want not in have:
            print(f"WARNING: hardware.device={want} not available here ({have}); using {have[0]}")
            return have[0]
        return want
    return have[0]


def torch_device_obj(name):
    import torch
    if name == "directml":
        import torch_directml
        return torch_directml.device()
    return torch.device(name)


@lru_cache(None)
def whisper_device():
    """(device, compute_type) for faster-whisper / ctranslate2."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


@lru_cache(None)
def ort_providers():
    try:
        import onnxruntime
        return onnxruntime.get_available_providers()
    except Exception:
        return []


def _cache_load():
    try:
        return json.loads((CACHE_DIR / "hw_cache.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_save(d):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "hw_cache.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


@lru_cache(None)
def working_hw_encoders():
    """Hardware H.264 encoders that really encode on this machine (listed != working: test-encode each)."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return []
    key = f"{ff}|{os.path.getmtime(ff):.0f}|{gpu_name()}"
    cache = _cache_load()
    if cache.get("encoders_key") == key:
        return cache["encoders"]
    listed = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    ok = []
    for enc in HW_ENCODERS:
        if f" {enc} " not in listed:
            continue
        r = subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:d=0.3", "-c:v", enc,
                            "-pix_fmt", "yuv420p" if enc != "h264_qsv" else "nv12", "-f", "null", "-"],
                           capture_output=True, timeout=60)
        if r.returncode == 0:
            ok.append(enc)
    cache.update({"encoders_key": key, "encoders": ok})
    _cache_save(cache)
    return ok


def encoder_args(cfg, crf):
    """ffmpeg video-encoder args. Default libx264 (best quality / size, works everywhere);
    hardware.encoder 'auto' -> first working GPU encoder, else libx264."""
    want = _cfg_hw(cfg).get("encoder", "libx264")
    if want == "auto":
        found = working_hw_encoders()
        want = found[0] if found else "libx264"
    if want == "libx264":
        return ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p"]
    if want not in HW_ENCODERS or want not in working_hw_encoders():
        print(f"WARNING: encoder {want} not working here; using libx264")
        return encoder_args({"hardware": {"encoder": "libx264"}}, crf)
    vt = max(1, min(100, round(100 - 2.2 * crf)))  # videotoolbox 0-100 quality scale
    args = [a.format(q=crf, vt=vt) for a in HW_ENCODERS[want]]
    return ["-c:v", want, *args, "-pix_fmt", "nv12" if want == "h264_qsv" else "yuv420p"]


def threads(cfg=None):
    t = _cfg_hw(cfg).get("threads", "auto")
    return cpu_threads() if t == "auto" else int(t)


def remotion_concurrency(cfg=None):
    """Each Remotion tab is a Chrome renderer (~0.5-1 GB). auto = min(cores/2, free RAM GB / 1.5), 1..8."""
    r = (cfg or {}).get("render", {}).get("concurrency", "auto")
    r = _cfg_hw(cfg).get("remotion_concurrency", r)
    if r not in (None, "auto"):
        return int(r)
    _, avail = ram_gb()
    return max(1, min(8, cpu_threads() // 2, int(avail / 1.5)))


# ---- run-time estimate for caption removal (seconds per frame that has text) ----
LAMA_SPF = {"cpu": 2.9, "cuda": 0.15, "mps": 0.6, "directml": 0.8}


def removal_spf(engine, device):
    if engine == "lama":
        return LAMA_SPF.get(device, 2.9)
    return 0.1


def ocr_engine():
    """RapidOCR; uses CUDA when onnxruntime-gpu is installed (CUDAExecutionProvider)."""
    from rapidocr_onnxruntime import RapidOCR
    if "CUDAExecutionProvider" in ort_providers():
        try:
            return RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        except Exception as e:
            print(f"WARNING: OCR on CUDA failed ({type(e).__name__}); using CPU")
    return RapidOCR()


FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
             Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts",
             Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
             Path.home() / "Library/Fonts", Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
             Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
FONT_FALLBACKS = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
                  "Helvetica.ttc", "arial.ttf", "DejaVuSans.ttf"]


def find_font(*names):
    """First existing font file among names (file names, searched in the OS font folders), then bold fallbacks."""
    for n in [*names, *FONT_FALLBACKS]:
        if not n:
            continue
        if Path(n).is_file():
            return str(n)
        for d in FONT_DIRS:
            if (d / n).is_file():
                return str(d / n)
            if d.is_dir() and d.name == "fonts":  # linux: nested folders
                hit = next(d.rglob(n), None)
                if hit:
                    return str(hit)
    return None


def report(cfg=None):
    tot, av = ram_gb()
    dev = torch_device(cfg)
    wd, wc = whisper_device()
    lines = [
        f"CPU threads      {cpu_threads()}",
        f"RAM              {tot:.1f} GB total, {av:.1f} GB free",
        f"GPU (torch)      {gpu_name() or 'none usable'}   devices: {', '.join(torch_devices())}",
        f"LaMa / demucs    -> {dev}   (LaMa ~{LAMA_SPF.get(dev, 2.9)} s per frame with text, 544x960)",
        f"whisper          -> {wd} / {wc}",
        f"OCR onnxruntime  {', '.join(p.replace('ExecutionProvider', '') for p in ort_providers()) or 'missing'}",
        f"GPU encoders     {', '.join(working_hw_encoders()) or 'none working'}   "
        f"(hardware.encoder=auto to use; default libx264)",
        f"Remotion tabs    {remotion_concurrency(cfg)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for s in (sys.stdout,):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(report())
