"""One-command deployment. The AI runs this; it installs whatever this machine is missing, then runs env_check.

  py -3.12 setup.py                 # everything: tools, Python packages, LaMa + demucs, GPU torch, Remotion, models
  py -3.12 setup.py --minimal       # required only (no LaMa / demucs / GPU torch / model pre-download)
  py -3.12 setup.py --no-gpu        # skip the CUDA torch download (~2.5 GB) even when an NVIDIA GPU is present
  py -3.12 setup.py --dry-run       # only print what would be installed

Steps (each is skipped when already satisfied; a failed optional step is reported and the rest continues):
  1 ffmpeg + ffprobe   winget (Windows) / brew (macOS) / apt-get or dnf (Linux, needs sudo)
  2 Node.js 18+        winget / brew / apt-get
  3 required pip       numpy opencv-python pillow requests faster-whisper rapidocr_onnxruntime (+ keyring off Windows)
  4 optional pip       simple-lama-inpainting (then Pillow is re-fixed: it pins Pillow 9.x) , demucs
  5 GPU               NVIDIA + CPU-only torch -> CUDA torch (cu124) + cuBLAS/cuDNN for whisper; Pillow re-fixed
  6 Remotion runtime   npm install into the shared cache folder (~300 MB) unless a usable install is found
  7 models             whisper 'small', LaMa, htdemucs pre-downloaded (HF mirror used when huggingface.co is blocked)
  8 MiMo key           only checked; if an env var holds it, it is copied into the credential store.
                       Missing -> the AI stores it (SKILL.md Phase 0: key piped into mimo_key.py --store)
  9 env_check.py
Never prints the API key.
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import common  # noqa: F401  (utf-8 console + PATH refresh)
from common import refresh_path

HERE = Path(__file__).resolve().parent
PY = sys.executable
REQUIRED = {"numpy": "numpy", "cv2": "opencv-python", "PIL": "pillow", "requests": "requests",
            "faster_whisper": "faster-whisper", "rapidocr_onnxruntime": "rapidocr_onnxruntime"}
if sys.platform != "win32":
    REQUIRED["keyring"] = "keyring"
PILLOW_FIX = [PY, "-m", "pip", "install", "--force-reinstall", "--no-deps", "pillow>=10"]
DRY = False
REPORT = []


def say(msg):
    print(f"\n== {msg}", flush=True)


def note(step, ok, detail=""):
    REPORT.append((step, ok, detail))


def sh(cmd, **kw):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    if DRY:
        return True
    try:
        return subprocess.run([str(c) for c in cmd], **kw).returncode == 0
    except FileNotFoundError:
        return False


def has_mod(m):
    importlib.invalidate_caches()
    return subprocess.run([PY, "-c", f"import {m}"], capture_output=True).returncode == 0


def pip(*pkgs, extra=()):
    return sh([PY, "-m", "pip", "install", "--disable-pip-version-check", *extra, *pkgs])


def sys_install(winget_id, brew, apt):
    """Install a system tool with the platform's package manager."""
    if sys.platform == "win32":
        if shutil.which("winget"):
            return sh(["winget", "install", "-e", "--id", winget_id, "--silent",
                       "--accept-package-agreements", "--accept-source-agreements"])
        return False
    if sys.platform == "darwin":
        return bool(shutil.which("brew")) and sh(["brew", "install", brew])
    for mgr in (["apt-get", "install", "-y"], ["dnf", "install", "-y"]):
        if shutil.which(mgr[0]):
            pre = [] if getattr(os, "geteuid", lambda: 1)() == 0 else ["sudo", "-n"]
            if mgr[0] == "apt-get":
                sh([*pre, "apt-get", "update"])
            return sh([*pre, *mgr, apt])
    return False


def node_ok():
    n = shutil.which("node")
    if not n:
        return False
    try:
        v = subprocess.run([n, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
        return int(v.split(".")[0]) >= 18
    except Exception:
        return False


def pillow_fix():
    from env_check import pillow_text_ok
    if DRY or pillow_text_ok():
        return True
    sh(PILLOW_FIX)
    return pillow_text_ok()


def hf_reachable():
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=8)
        return True
    except Exception:
        return False


def model_env():
    env = dict(os.environ)
    if not env.get("HF_ENDPOINT") and not hf_reachable():
        print("huggingface.co not reachable -> using mirror https://hf-mirror.com")
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
    return env


def step_tools():
    say("1-2 system tools")
    for tool, wid, brew, apt in (("ffmpeg", "Gyan.FFmpeg", "ffmpeg", "ffmpeg"),):
        if shutil.which(tool) and shutil.which("ffprobe"):
            note("ffmpeg", True, "present")
            continue
        sys_install(wid, brew, apt)
        refresh_path()
        note("ffmpeg", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
             "" if shutil.which("ffmpeg") else "install manually: https://ffmpeg.org/ (then reopen the terminal)")
    if node_ok():
        note("node 18+", True, "present")
    else:
        sys_install("OpenJS.NodeJS.LTS", "node", "nodejs")
        refresh_path()
        note("node 18+", node_ok(), "" if node_ok() else "install manually: https://nodejs.org/ (LTS)")


def step_pip_required():
    say("3 required Python packages")
    miss = [p for m, p in REQUIRED.items() if not has_mod(m)]
    if miss:
        pip(*miss)
    miss = [p for m, p in REQUIRED.items() if not has_mod(m)]
    note("python packages", DRY or not miss, ("still missing: " + ", ".join(miss)) if miss else "")
    note("pillow text", pillow_fix())


def step_pip_optional():
    say("4 optional Python packages (LaMa inpainting, demucs music separation)")
    for mod, pkg in (("simple_lama_inpainting", "simple-lama-inpainting"), ("demucs", "demucs")):
        if not has_mod(mod):
            pip(pkg)
        note(pkg, DRY or has_mod(mod))
    note("pillow after LaMa", pillow_fix(), "simple-lama-inpainting pins Pillow 9.x; re-fixed")


def step_gpu():
    say("5 GPU acceleration")
    if not shutil.which("nvidia-smi"):
        note("gpu", True, "no NVIDIA GPU: CPU / hardware encoder paths used automatically")
        return
    cuda = subprocess.run([PY, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                          capture_output=True).returncode == 0
    if not cuda and has_mod("torch"):
        pip("torch", "torchaudio", extra=("--force-reinstall", "--index-url", "https://download.pytorch.org/whl/cu124"))
        pillow_fix()
        cuda = DRY or subprocess.run([PY, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                                     capture_output=True).returncode == 0
    note("torch CUDA", cuda, "" if cuda else "CUDA torch not usable (driver too old?) - CPU is used")
    if not has_mod("nvidia.cudnn"):
        pip("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")
    note("whisper CUDA libs", DRY or has_mod("nvidia.cudnn"))


def step_remotion():
    say("6 Remotion runtime")
    from render import find_remotion, install_runtime
    nm, ver = find_remotion({"render": {}})
    if nm:
        note("remotion", True, f"v{ver} at {nm}")
        return
    if not node_ok():
        note("remotion", False, "needs Node.js 18+ first")
        return
    if DRY:
        print("would npm install Remotion (~300 MB)")
        return
    try:
        nm, ver = install_runtime()
        note("remotion", True, f"v{ver} installed at {nm}")
    except SystemExit:
        note("remotion", False, "npm install failed (network? try: npm config set registry https://registry.npmmirror.com)")


def step_models():
    say("7 model pre-download (whisper small, LaMa, htdemucs)")
    env = model_env()
    jobs = [("whisper small", "faster_whisper",
             "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"),
            ("LaMa model", "simple_lama_inpainting",
             "import torch; from simple_lama_inpainting import SimpleLama; SimpleLama(device=torch.device('cpu'))"),
            ("htdemucs model", "demucs", "from demucs.pretrained import get_model; get_model('htdemucs')")]
    for name, mod, code in jobs:
        if not has_mod(mod):
            continue
        ok = sh([PY, "-c", code], env=env, timeout=1800)
        note(name, ok, "" if ok else "download failed; it is retried automatically on first use")


def step_key():
    say("8 MiMo API key")
    from mimo_key import key_source, read_credential
    src = key_source()
    if src and not read_credential() and not DRY:  # only in an env var -> persist it in the credential store
        subprocess.run([PY, str(HERE / "mimo_key.py"), "--store", "--from-env"])
    if read_credential() or key_source():
        note("mimo key", True, key_source())
    else:
        note("mimo key", False, "NOT configured -> AI: follow SKILL.md Phase 0 'MiMo key' (pipe it into mimo_key.py --store)")


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimal", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    DRY = a.dry_run
    if sys.version_info[:2] < (3, 10):
        sys.exit("Python 3.10+ needed (3.12 recommended): https://www.python.org/")
    t0 = time.time()
    step_tools()
    step_pip_required()
    if not a.minimal:
        step_pip_optional()
        if not a.no_gpu:
            step_gpu()
    step_remotion()
    if not a.minimal:
        step_models()
    step_key()
    say(f"summary ({time.time() - t0:.0f}s)")
    for s, ok, d in REPORT:
        print(f"  [{'OK' if ok else '!!'}] {s}{('  ' + d) if d else ''}")
    say("9 env_check")
    r = subprocess.run([PY, str(HERE / "env_check.py")])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
