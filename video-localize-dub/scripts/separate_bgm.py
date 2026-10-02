"""Remove the original voice, keep music/effects -> analysis/bgm.wav (used when dub.background = "separate").

Needs demucs (setup.py installs it; installed automatically here when missing)
First run downloads the htdemucs model (~80 MB).

  py -3.12 separate_bgm.py --job <job>
"""

import argparse
import shutil
import subprocess
import sys

from common import die, job_paths, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--model", default="htdemucs")
    a = ap.parse_args()
    P = job_paths(a.job)
    if not P["audio_full"].exists():
        die("analysis/audio_full.wav missing. Run analyze_audio.py first.")
    try:
        import demucs  # noqa: F401
    except ImportError:
        print("demucs not installed - installing it (pip, ~100 MB incl. model on first run)")
        if subprocess.run([sys.executable, "-m", "pip", "install", "demucs"]).returncode:
            die("pip install demucs failed; choose dub.background = none / file instead.")
    import hw
    from common import load_config
    dev = hw.torch_device(load_config(a.job))
    dev = dev if dev in ("cuda", "mps") else "cpu"  # demucs CLI has no directml
    out = P["analysis"] / "demucs"
    print(f"demucs on {dev}")
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", a.model, "-o", out, P["audio_full"]]
    try:
        run(cmd[:3] + ["-d", dev] + cmd[3:])
    except Exception:
        if dev == "cpu":
            raise
        print("WARNING: demucs on GPU failed; retrying on CPU")
        run(cmd[:3] + ["-d", "cpu"] + cmd[3:])
    stem = out / a.model / P["audio_full"].stem / "no_vocals.wav"
    if not stem.exists():
        die(f"demucs output missing: {stem}")
    # back to 48 kHz stereo pcm so the mixer can read it directly
    run(["ffmpeg", "-v", "error", "-y", "-i", stem, "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", P["bgm"]])
    shutil.rmtree(out, ignore_errors=True)
    print(f"Background (voice removed): {P['bgm']}  - listen to it before mixing; separation is not perfect.")


if __name__ == "__main__":
    main()
