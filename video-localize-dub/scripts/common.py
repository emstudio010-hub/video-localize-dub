"""Shared helpers: ffprobe, ffmpeg, JSON IO, job paths."""

import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # Windows consoles default to GBK
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = SKILL_DIR / "remotion-template"


def refresh_path():
    """Pick up tools installed during this process (winget / brew write PATH for new shells only)."""
    import os
    extra = []
    if sys.platform == "win32":
        try:
            import winreg
            for root, sub in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                              (winreg.HKEY_CURRENT_USER, "Environment")):
                with winreg.OpenKey(root, sub) as k:
                    extra += os.path.expandvars(winreg.QueryValueEx(k, "Path")[0]).split(";")
        except OSError:
            pass
        extra.append(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"))
    else:
        extra += ["/opt/homebrew/bin", "/usr/local/bin"]
    have = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(have + [p for p in extra if p and p not in have])


def die(msg, code=1):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(code)


def run(cmd, quiet=False, **kw):
    if not quiet:
        print("$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def need(tool):
    if not shutil.which(tool):
        die(f"{tool} not found on PATH. Run env_check.py first.")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def probe(video):
    """Return width, height, fps (float), fps_str, frames, duration, has_audio."""
    need("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
        check=True, capture_output=True, text=True, encoding="utf-8").stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        die(f"No video stream in {video}")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate")
    if not fps_str or fps_str == "0/0":
        fps_str = v.get("r_frame_rate")
    fps = float(Fraction(fps_str))
    duration = float(v.get("duration") or info["format"]["duration"])
    frames = v.get("nb_frames")
    if frames and frames.isdigit():
        frames = int(frames)
    else:
        frames = count_frames(video)
    r_fps = float(Fraction(v.get("r_frame_rate", fps_str)))
    return {
        "width": int(v["width"]), "height": int(v["height"]),
        "fps": fps, "fps_str": fps_str, "frames": frames, "duration": duration,
        "has_audio": a is not None,
        "vfr_suspect": abs(r_fps - fps) > 0.01,
    }


def count_frames(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True).stdout.strip()
    return int(out)


def job_paths(job_dir):
    j = Path(job_dir).resolve()
    return {
        "job": j,
        "config": j / "job.json",
        "analysis": j / "analysis",
        "audio16k": j / "analysis" / "audio16k.wav",
        "audio_full": j / "analysis" / "audio_full.wav",
        "transcript": j / "analysis" / "transcript.json",
        "speakers": j / "analysis" / "speakers.json",
        "ocr": j / "analysis" / "ocr.json",
        "bgm": j / "analysis" / "bgm.wav",
        "script": j / "script.json",
        "tts_dir": j / "tts",
        "tts_index": j / "tts" / "tts.json",
        "timeline": j / "audio" / "timeline.json",
        "dub": j / "audio" / "dub.wav",
        "captions": j / "captions.json",
        "clean": j / "video" / "clean.mp4",
        "preview": j / "preview",
        "remotion": j / "remotion",
        "captioned": j / "video" / "captioned.mp4",
        "master": j / "video" / "master.mp4",
        "watermarked": j / "video" / "watermarked.mp4",
        "qc": j / "qc",
    }


def load_config(job_dir):
    p = job_paths(job_dir)["config"]
    if not p.exists():
        die(f"{p} missing. Run init_job.py first.")
    return load_json(p)


def hex_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def count_words(text, lang="en"):
    if lang == "zh":
        return sum(1 for ch in text if "一" <= ch <= "鿿")
    return len([w for w in text.replace("—", " ").split() if any(c.isalnum() for c in w)])
