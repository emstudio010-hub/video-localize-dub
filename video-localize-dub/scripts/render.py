"""Render captions over the cleaned video with Remotion, mux the dub, optional watermark, deliver.

  py -3.12 render.py --job <job> --find          # only report which Remotion install would be used
  py -3.12 render.py --job <job>                 # render + mux (+ watermark) + copy to output.dir
  (no usable Remotion found -> it is npm-installed automatically into the shared runtime, ~300 MB)
  py -3.12 render.py --job <job> --frames 0-89   # quick partial render for review (not delivered)

Remotion lookup order: job.json render.remotion_node_modules -> shared runtime
(%LOCALAPPDATA%/video-localize-dub/remotion-runtime) -> search of common folders for a project
with remotion + @remotion/cli + react (same remotion/cli version).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import hw

from common import TEMPLATE_DIR, die, job_paths, load_config, load_json, run

RUNTIME = hw.CACHE_DIR / "remotion-runtime"
SEARCH_ROOTS = [Path.home() / d for d in ("Documents", "Desktop", "Projects", "source", "code", "dev", "work")]
if sys.platform == "win32":
    SEARCH_ROOTS += [Path(f"{d}:/") for d in "CDEF" if d != "C"] + [Path("C:/tmp"), Path("C:/work")]
else:
    SEARCH_ROOTS += [Path("/tmp"), Path("/opt")]


def _ver(nm, pkg):
    try:
        return json.loads((nm / pkg / "package.json").read_text(encoding="utf-8"))["version"]
    except Exception:
        return None


def usable(nm):
    nm = Path(nm)
    r, c = _ver(nm, "remotion"), _ver(nm, "@remotion/cli")
    ok = r and c and r == c and r.startswith("4.") and _ver(nm, "react") and _ver(nm, "react-dom")
    return (r if ok else None)


def find_remotion(cfg, deep=True):
    cands = []
    if cfg.get("render", {}).get("remotion_node_modules"):
        cands.append(Path(cfg["render"]["remotion_node_modules"]))
    cands.append(RUNTIME / "node_modules")
    for p in cands:
        v = usable(p)
        if v:
            return p, v
    if deep:
        for root in SEARCH_ROOTS:
            if not root.exists():
                continue
            for dirpath, dirnames, _ in os.walk(root):
                depth = len(Path(dirpath).relative_to(root).parts)
                if "node_modules" in dirnames:
                    nm = Path(dirpath) / "node_modules"
                    v = usable(nm)
                    if v:
                        return nm, v
                # don't descend into node_modules / hidden / deep trees
                dirnames[:] = [d for d in dirnames if d != "node_modules" and not d.startswith(".")] if depth < 5 else []
    return None, None


def install_runtime():
    shutil.which("npm") or die("npm not found. Install Node.js 18+ (https://nodejs.org) first.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "package.json", RUNTIME / "package.json")
    run(["npm", "install", "--no-audit", "--no-fund"], cwd=RUNTIME, shell=(sys.platform == "win32"))
    v = usable(RUNTIME / "node_modules")
    if not v:
        die("npm install finished but Remotion is still not usable.")
    return RUNTIME / "node_modules", v


def link(src, dst):
    """Hardlink (same drive, instant, no copy) else copy."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_dir(target, link_path):
    """Directory link without admin rights: junction on Windows, symlink elsewhere."""
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link_path), str(target)], check=True, capture_output=True)
    else:
        os.symlink(target, link_path, target_is_directory=True)


def unlink_dir(link_path):
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "rmdir", str(link_path)], check=False)
    else:
        os.unlink(link_path)


def remotion_bin(proj):
    b = proj / "node_modules" / ".bin"
    return str(b / "remotion.cmd") if sys.platform == "win32" else str(b / "remotion")


def prepare_project(P, cfg, nm):
    proj = P["remotion"]
    if proj.exists():
        jn = proj / "node_modules"
        if jn.exists() or jn.is_symlink():
            unlink_dir(jn)  # the link only, never the shared node_modules it points to
        shutil.rmtree(proj, ignore_errors=True)
    shutil.copytree(TEMPLATE_DIR, proj, ignore=shutil.ignore_patterns("node_modules"))
    (proj / "public").mkdir(exist_ok=True)
    link_dir(nm, proj / "node_modules")
    video = P["clean"] if P["clean"].exists() else Path(cfg["source"])
    if not P["clean"].exists():
        print("NOTE: video/clean.mp4 not found - captions go over the ORIGINAL frames.")
    link(video, proj / "public" / "clean.mp4")
    ff = cfg["captions"].get("font_file")
    if ff:
        shutil.copy2(ff, proj / "public" / Path(ff).name)
    return proj


def watermark(cfg, P, src, dst):
    from PIL import Image, ImageDraw, ImageFont
    wm = cfg["watermark"]
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    font = ImageFont.truetype(hw.find_font(wm.get("font_file"), "arialbd.ttf"), int(wm["font_size"] * W / 1080))
    big = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    tw = d.textlength(wm["text"], font=font)
    x, y = W - tw / 2, H - font.size / 2
    a = int(255 * wm["opacity"])
    off = max(2, font.size // 40)
    d.text((x + off, y + off), wm["text"], font=font, fill=(0, 0, 0, int(a * wm.get("shadow_opacity", 0.4))))
    d.text((x, y), wm["text"], font=font, fill=(255, 255, 255, a))
    # angle_deg uses the CSS convention like captions (positive = clockwise, -25 = rising to the right);
    # PIL rotates counter-clockwise for positive angles, hence the minus.
    big = big.rotate(-wm["angle_deg"], resample=Image.BICUBIC)
    big = big.crop((W // 2, H // 2, W // 2 + W, H // 2 + H))
    png = P["qc"] / "watermark_overlay.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    big.save(png)
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", png, "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto",
         "-map", "0:a?", *hw.encoder_args(cfg, cfg["render"]["crf"]), "-c:a", "copy", dst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--find", action="store_true")
    ap.add_argument("--install", action="store_true", help="kept for compatibility; install is automatic")
    ap.add_argument("--frames", default=None, help="e.g. 0-89 : partial review render only")
    a = ap.parse_args()
    P = job_paths(a.job)
    cfg = load_config(a.job)
    r = cfg["render"]

    nm, ver = (None, None)
    if cfg["captions"]["enabled"] or a.find:
        nm, ver = find_remotion(cfg)
        if a.find:
            print(f"Remotion: {nm} (v{ver})" if nm else f"Remotion: NOT FOUND (render.py installs it into {RUNTIME})")
            return
        if not nm:
            print(f"No usable Remotion found - installing it (~300 MB) into {RUNTIME}")
            nm, ver = install_runtime()
        print(f"Using Remotion v{ver}: {nm}")

    P["captioned"].parent.mkdir(parents=True, exist_ok=True)
    if cfg["captions"]["enabled"]:
        if not P["captions"].exists():
            die("captions.json missing. Run build_captions.py first.")
        deps = [P["captions"], P["clean"] if P["clean"].exists() else Path(cfg["source"])]
        fresh = (not a.frames and P["captioned"].exists()
                 and P["captioned"].stat().st_mtime > max(d.stat().st_mtime for d in deps))
        if fresh:
            print("captioned.mp4 is newer than captions.json and clean.mp4 - reusing it (re-mux only).")
        else:
            proj = prepare_project(P, cfg, nm)
            out = P["preview"] / f"render_{a.frames}.mp4" if a.frames else P["captioned"]
            out.parent.mkdir(parents=True, exist_ok=True)
            conc = hw.remotion_concurrency(cfg)
            gl = cfg.get("hardware", {}).get("remotion_gl")  # e.g. angle / egl / swangle; default = Remotion's
            print(f"Remotion: {conc} parallel tabs")
            cmd = [remotion_bin(proj), "render", "src/index.ts", "Localized",
                   str(out), f"--props={P['captions']}", "--codec=h264", f"--crf={r['crf']}",
                   "--pixel-format=yuv420p", f"--concurrency={conc}", "--muted", "--log=warn"]
            if gl:
                cmd.append(f"--gl={gl}")
            if a.frames:
                cmd.append(f"--frames={a.frames}")
            run(cmd, cwd=proj)
            if a.frames:
                print(f"Partial render: {out}")
                return
        video = P["captioned"]
    else:
        video = P["clean"] if P["clean"].exists() else Path(cfg["source"])

    # mux: video stream copied untouched, dub as AAC. No -shortest: dub.wav is already exact length.
    if cfg["dub"]["enabled"]:
        if not P["dub"].exists():
            die("audio/dub.wav missing. Run schedule_audio.py first.")
        run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", P["dub"], "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", P["master"]])
    else:
        run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", cfg["source"], "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", P["master"]])
    outs = [P["master"]]
    if cfg["watermark"]["enabled"]:
        watermark(cfg, P, P["master"], P["watermarked"])
        outs.append(P["watermarked"])

    dest = cfg["output"].get("dir")
    if dest:
        Path(dest).mkdir(parents=True, exist_ok=True)
        base = cfg["output"]["basename"]
        for o, suf in zip(outs, ["", "_watermark"]):
            t = Path(dest) / f"{base}{suf}.mp4"
            shutil.copy2(o, t)
            print(f"Delivered: {t}")
    else:
        print("output.dir empty - files stay in the job folder:\n  " + "\n  ".join(map(str, outs)))
    print("Next: run verify.py and LOOK at the QC sheets before telling the user it is done.")


if __name__ == "__main__":
    main()
