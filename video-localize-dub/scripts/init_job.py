"""Create a job folder with job.json (all user choices live there).

  py -3.12 init_job.py --source <video> --job <job_dir>
"""

import argparse
import shutil
from pathlib import Path

from common import SKILL_DIR, die, load_json, probe, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    src = Path(a.source).resolve()
    if not src.exists():
        die(f"source not found: {src}")
    job = Path(a.job).resolve()
    cfgp = job / "job.json"
    if cfgp.exists():
        die(f"{cfgp} already exists; edit it instead of re-initialising.")
    job.mkdir(parents=True, exist_ok=True)
    cfg = load_json(SKILL_DIR / "config.template.json")
    cfg.pop("_comment", None)
    cfg["source"] = str(src)
    cfg["video"] = probe(src)
    cfg["output"]["basename"] = src.stem + "_" + cfg["target_language"]
    save_json(cfgp, cfg)
    print(f"Job created: {job}\nVideo: {cfg['video']}")
    if cfg["video"]["vfr_suspect"]:
        print("WARNING: variable frame rate suspected. Convert to CFR first "
              "(ffmpeg -i in -vsync cfr -r <fps> -c:v libx264 -crf 14 out) and re-init.")


if __name__ == "__main__":
    main()
