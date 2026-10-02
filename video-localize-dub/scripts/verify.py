"""Final checks + QC images. The agent MUST open the QC images and look at them before reporting.

  py -3.12 verify.py --job <job>

Checks:
  - master: same resolution, fps and frame count as the source; audio length == video length
  - dub timeline: no overlap, tempo inside bounds
  - captions: no overlapping cards, no flicker gaps shorter than bridge_gap_frames
  - residual text: RapidOCR on removed blocks in video/clean.mp4 (should find nothing)
QC images (qc/):
  - removal_*.png : source | clean crops of every removed block at several moments
  - captions_*.png: master frames at caption cards (start + middle), 3x3 grid
Writes qc/report.json. Exit 1 if a hard check fails.
"""

import argparse
import subprocess
from difflib import SequenceMatcher

import cv2
import numpy as np

from common import job_paths, load_config, load_json, probe, save_json


def grab(path, frames):
    cap = cv2.VideoCapture(str(path))
    out = {}
    for f in sorted(set(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            out[f] = img
    cap.release()
    return out


def audio_len(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return None


def label(img, text):
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grid(imgs, cols=3, width=360):
    tiles = []
    for im in imgs:
        h = int(im.shape[0] * width / im.shape[1])
        tiles.append(cv2.resize(im, (width, h)))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], width, 3), np.uint8)]) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    P = job_paths(a.job)
    cfg = load_config(a.job)
    P["qc"].mkdir(parents=True, exist_ok=True)
    errors, warns, info = [], [], {}
    src = cfg["video"]
    fps = src["fps"]

    # 1) master stream checks
    if not P["master"].exists():
        errors.append("video/master.mp4 missing (run render.py)")
    else:
        m = probe(P["master"])
        info["master"] = m
        for k in ("width", "height", "frames"):
            if m[k] != src[k]:
                errors.append(f"master {k} {m[k]} != source {src[k]}")
        if abs(m["fps"] - fps) > 0.01:
            errors.append(f"master fps {m['fps']} != source {fps}")
        al = audio_len(P["master"])
        info["audio_s"] = al
        if al is None:
            errors.append("master has no audio stream")
        elif abs(al - src["duration"]) > max(0.05, 1.5 / fps):
            errors.append(f"audio {al:.3f}s vs video {src['duration']:.3f}s")

    # 2) dub timeline
    if cfg["dub"]["enabled"] and P["timeline"].exists():
        tl = load_json(P["timeline"])["segments"]
        d = cfg["dub"]
        for x, y in zip(tl, tl[1:]):
            if y["start"] < x["end"] - 1e-3:
                errors.append(f"dub overlap seg {x['id']}/{y['id']}")
        for t in tl:
            if not d["min_tempo"] - 1e-3 <= t["tempo"] <= d["max_tempo"] + 1e-3:
                errors.append(f"seg {t['id']} tempo {t['tempo']}")
        info["tempo_range"] = [min(t["tempo"] for t in tl), max(t["tempo"] for t in tl)]

    # 3) captions
    cards = []
    if cfg["captions"]["enabled"] and P["captions"].exists():
        cards = load_json(P["captions"])["cards"]
        br = cfg["captions"]["bridge_gap_frames"]
        for x, y in zip(cards, cards[1:]):
            if y["from"] < x["to"]:
                errors.append(f"caption overlap at frame {y['from']}")
            elif 0 < y["from"] - x["to"] < br:
                warns.append(f"caption flicker gap {y['from'] - x['to']}f at frame {x['to']}")
        info["caption_cards"] = len(cards)

    # 4) residual text in removed blocks + removal sheets
    if P["clean"].exists() and P["ocr"].exists():
        import hw
        ocr_engine = hw.ocr_engine()
        ocr = load_json(P["ocr"])
        blocks_cfg = cfg["removal"].get("blocks", {})
        default_mode = cfg["removal"].get("default", {}).get("mode", "inpaint")
        residual = []
        for b in ocr["blocks"]:
            mode = blocks_cfg.get(b["id"], {}).get("mode", default_mode)
            if mode == "keep":
                continue
            evs = b["events"]
            picks = [(e["first_seen"] + e["last_seen"]) // 2 for e in evs[:: max(1, len(evs) // 6)]][:6]
            S, C = grab(cfg["source"], picks), grab(P["clean"], picks)
            x0, y0, x1, y1 = b["union_bbox"]
            pad = int(b["char_height_px"])
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(src["width"], x1 + pad), min(src["height"], y1 + pad)
            rows = []
            for f in picks:
                if f not in S or f not in C:
                    continue
                cs, cc = S[f][y0:y1, x0:x1], C[f][y0:y1, x0:x1]
                rows.append(np.hstack([label(cs.copy(), f"{b['id']} f{f} src"),
                                       np.full((cs.shape[0], 6, 3), 255, np.uint8),
                                       label(cc.copy(), "clean")]))
                if mode == "inpaint":
                    res, _ = ocr_engine(cc)
                    hits = [r[1] for r in (res or []) if r[2] > 0.6 and len(r[1].strip()) >= 2]
                    orig = next((e["text"] for e in evs if e["first_seen"] <= f <= e["last_seen"]), "")
                    hits = [h for h in hits if SequenceMatcher(None, h, orig).ratio() > 0.3 or not orig]
                    if hits:
                        residual.append({"block": b["id"], "frame": f, "text": hits})
            if rows:
                w = max(r.shape[1] for r in rows)
                rows = [np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)]) for r in rows]
                cv2.imwrite(str(P["qc"] / f"removal_{b['id']}.png"), np.vstack(rows))
        info["residual_text"] = residual
        for r in residual:
            warns.append(f"readable text left in {r['block']} at frame {r['frame']}: {r['text']}")

    # 5) caption sheets from the master
    if P["master"].exists() and cards:
        picks = []
        for c in cards:
            picks += [c["from"] + 1, (c["from"] + c["to"]) // 2]
        frames = grab(P["master"], picks)
        imgs = [label(frames[f].copy(), f"f{f} {f / fps:.2f}s") for f in sorted(frames)]
        for i in range(0, len(imgs), 9):
            cv2.imwrite(str(P["qc"] / f"captions_{i // 9 + 1:02d}.png"), grid(imgs[i:i + 9]))

    save_json(P["qc"] / "report.json", {"errors": errors, "warnings": warns, "info": info})
    for w in warns:
        print("WARN: " + w)
    for e in errors:
        print("ERROR: " + e)
    print(f"\nQC images in {P['qc']} - open and inspect every removal_*.png and captions_*.png now.")
    print("RESULT: " + ("FAIL" if errors else "PASS (hard checks) - visual review still required"))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
