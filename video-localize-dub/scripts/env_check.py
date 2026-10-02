"""Check everything the pipeline needs. Never prints the API key.

  py -3.12 env_check.py
  py -3.12 env_check.py --install-required     # pip install the required Python packages
Full automatic deployment (tools, optional packages, GPU, Remotion, models, key): setup.py
Exit 1 if something REQUIRED is missing. Optional items only change which choices you can offer.
"""

import argparse
import importlib
import shutil
import subprocess
import sys

REQUIRED_PIP = {"numpy": "numpy", "cv2": "opencv-python", "PIL": "pillow", "requests": "requests",
                "faster_whisper": "faster-whisper", "rapidocr_onnxruntime": "rapidocr_onnxruntime"}
OPTIONAL_PIP = {"demucs": ("demucs", "voice/music separation (dub.background = separate)"),
                "simple_lama_inpainting": ("simple-lama-inpainting", "LaMa inpainting (removal.engine = lama)")}


def ok(flag, name, note=""):
    print(f"  [{'OK' if flag else '--'}] {name}{('  ' + note) if note else ''}")
    return flag


def pillow_text_ok():
    # Pillow 9.x crashes in getlength() on Python 3.12 (simple-lama-inpainting pins it); test in a subprocess
    import hw
    f = hw.find_font()
    if not f:
        return False
    code = f"from PIL import ImageFont; print(ImageFont.truetype({f!r}, 20).getlength('Ab'))"
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def gpu_hints():
    """What to install so this machine's GPU is used (never installs anything itself)."""
    import hw
    out = []
    nvidia = shutil.which("nvidia-smi") is not None
    devs = hw.torch_devices()
    if nvidia and "cuda" not in devs:
        out.append("NVIDIA GPU found but torch is CPU-only -> LaMa/demucs on GPU (~20x faster): "
                   f"{sys.executable} -m pip install --force-reinstall torch torchaudio "
                   "--index-url https://download.pytorch.org/whl/cu124   (~2.5 GB; then re-fix pillow, help.md)")
    if nvidia and hw.whisper_device()[0] == "cpu":
        out.append("whisper on GPU needs the CUDA 12 + cuDNN 9 runtime: "
                   f"{sys.executable} -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*")
    if nvidia and "CUDAExecutionProvider" not in hw.ort_providers():
        out.append("OCR on GPU (optional, OCR is fast anyway): pip install onnxruntime-gpu")
    if sys.platform == "win32" and not nvidia and devs == ["cpu"] and hw.working_hw_encoders():
        out.append("AMD/Intel GPU: LaMa can try DirectML (experimental; needs a torch version that torch-directml "
                   f"supports): {sys.executable} -m pip install torch-directml ; on failure it falls back to CPU")
    if devs == ["cpu"]:
        out.append("no usable GPU for LaMa: plan ~2.9 s per frame with text, or use engine opencv / blur / color")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-required", action="store_true")
    a = ap.parse_args()
    bad = []

    print("Tools")
    ok(sys.version_info[:2] >= (3, 10), f"Python {sys.version.split()[0]}") or bad.append("python>=3.10")
    for t in ("ffmpeg", "ffprobe"):
        ok(bool(shutil.which(t)), t) or bad.append(t)
    node = shutil.which("node")
    ok(bool(node), "node", "(needed for Remotion captions)")
    ok(bool(shutil.which("npm")), "npm")

    print("Python packages (required)")
    missing = []
    for mod, pkg in REQUIRED_PIP.items():
        try:
            importlib.import_module(mod)
            ok(True, pkg)
        except Exception:
            ok(False, pkg)
            missing.append(pkg)
    if "pillow" not in missing:
        ok(pillow_text_ok(), "pillow text measuring",
           "" if pillow_text_ok() else "BROKEN -> py -3.12 -m pip install --force-reinstall --no-deps \"pillow>=10\"") or bad.append("pillow>=10")
    if missing and a.install_required:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=False)
    elif missing:
        bad += missing
        print(f"     install: {sys.executable} -m pip install {' '.join(missing)}")

    print("Python packages (optional)")
    for mod, (pkg, why) in OPTIONAL_PIP.items():
        try:
            importlib.import_module(mod)
            ok(True, pkg, why)
        except Exception:
            ok(False, pkg, f"{why}; install: py -3.12 -m pip install {pkg}")

    print("Whisper models (cached)")
    try:
        from huggingface_hub import scan_cache_dir
        sizes = sorted(r.repo_id.split("faster-whisper-")[-1] for r in scan_cache_dir().repos
                       if "faster-whisper" in r.repo_id)
        ok(bool(sizes), ", ".join(sizes) or "none",
           "" if sizes else "first run downloads one (China: set HF_ENDPOINT=https://hf-mirror.com)")
    except Exception:
        ok(False, "could not scan HF cache")

    print("MiMo API key")
    try:
        from mimo_key import key_source
        src = key_source()
        ok(bool(src), src or "not found",
           "" if src else "create a key at https://platform.xiaomimimo.com/ then "
                          "the AI stores it: SKILL.md Phase 0, key piped into mimo_key.py --store") or bad.append("mimo key")
    except Exception as e:
        ok(False, f"key lookup failed: {type(e).__name__}")
        bad.append("mimo key")

    print("Remotion")
    from common import load_json
    from render import RUNTIME, find_remotion
    nm, ver = find_remotion({"render": {}})
    ok(bool(nm), f"{nm} (v{ver})" if nm else "not found",
       "" if nm else f"setup.py / render.py install it into {RUNTIME}")

    print("Hardware (GPU acceleration is automatic; details: py -3.12 hw.py)")
    import hw
    print("  " + hw.report().replace("\n", "\n  "))
    for line in gpu_hints():
        print("  hint: " + line)

    print("\nRESULT: " + ("missing required: " + ", ".join(bad) if bad else "ready"))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
