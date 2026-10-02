"""Phase 1 visual analysis: find every burned-in text block anywhere in the frame
(any position, horizontal / vertical / rotated) with local RapidOCR (offline, free).

  py -3.12 ocr_scan.py --job <job_dir> [--step 0.33] [--min-score 0.6]

Writes analysis/ocr.json (blocks with geometry, style, timing) and
analysis/ocr_blocks.png (labelled overview + crops) - LOOK at it and show the user.
"""

import argparse
import math
from difflib import SequenceMatcher

import cv2
import numpy as np

from common import job_paths, load_config, save_json


def poly_geom(poly):
    p = np.array(poly, np.float32)
    top = p[1] - p[0]
    left = p[3] - p[0]
    w = float(np.linalg.norm(top))
    h = float(np.linalg.norm(left))
    angle = math.degrees(math.atan2(top[1], top[0]))
    cx, cy = p.mean(0)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return {"cx": float(cx), "cy": float(cy), "w": w, "h": h, "angle": angle,
            "bbox": [int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))]}


def orientation(g, text):
    n = max(1, len(text.strip()))
    if g["h"] > g["w"] * 1.8 and n >= 2:
        return "vertical"
    if abs(g["angle"]) > 4:
        return "rotated"
    return "horizontal"


def char_height(g, orient):
    return g["w"] if orient == "vertical" else g["h"]


def colors(img, bbox):
    """Return (text_hex, plate_hex_or_None) estimated inside bbox."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    m = 5
    ring = np.concatenate([
        img[max(0, y0 - m):y0, x0:x1].reshape(-1, 3), img[y1:min(H, y1 + m), x0:x1].reshape(-1, 3)])
    bg = ring.mean(0) if len(ring) else crop.reshape(-1, 3).mean(0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a = crop[th > 0].reshape(-1, 3)
    b = crop[th == 0].reshape(-1, 3)
    cands = [c for c in (a, b) if len(c) > 10]
    if not cands:
        return None, None
    text = max(cands, key=lambda c: np.linalg.norm(c.mean(0) - bg))
    t = np.median(text, 0)
    # plate: the inner border band is uniform and differs from the outside ring
    band = np.concatenate([crop[:3].reshape(-1, 3), crop[-3:].reshape(-1, 3)])
    plate = None
    if len(band) and band.std(0).mean() < 18 and np.linalg.norm(band.mean(0) - bg) > 40:
        plate = band.mean(0)

    def hx(c):
        return "#%02X%02X%02X" % (int(c[2]), int(c[1]), int(c[0]))
    return hx(t), (hx(plate) if plate is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--step", type=float, default=0.33, help="seconds between sampled frames")
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--dense-head", type=float, default=1.5,
                    help="seconds at the start sampled every 2 frames (covers / title cards are often short)")
    ap.add_argument("--max-side", type=int, default=1280, help="downscale long side for OCR speed")
    a = ap.parse_args()

    import hw
    eng = hw.ocr_engine()
    P = job_paths(a.job)
    cfg = load_config(a.job)
    v = cfg["video"]
    fps, W, H = v["fps"], v["width"], v["height"]
    step_f = max(1, round(a.step * fps))
    scale = min(1.0, a.max_side / max(W, H))

    cap = cv2.VideoCapture(cfg["source"])
    occ = []
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if n % step_f == 0 or (n < a.dense_head * fps and n % 2 == 0):
            ok, img = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
            res, _ = eng(small)
            for poly, text, score in res or []:
                if float(score) < a.min_score or not text.strip():
                    continue
                poly = (np.array(poly, np.float32) / scale).tolist()
                g = poly_geom(poly)
                o = orientation(g, text)
                tc, pc = colors(img, g["bbox"])
                occ.append({"frame": n, "text": text.strip(), "score": round(float(score), 3),
                            "poly": [[round(x, 1), round(y, 1)] for x, y in poly], **g,
                            "orient": o, "char_h": char_height(g, o), "text_color": tc, "plate_color": pc})
            if n % (step_f * 30) == 0:
                print(f"  frame {n}/{v['frames']}  detections so far {len(occ)}", flush=True)
        n += 1
    cap.release()

    # ---- group occurrences into spatial blocks ----
    blocks = []
    for o in sorted(occ, key=lambda o: o["frame"]):
        best = None
        for b in blocks:
            if b["orient"] != o["orient"] or abs(b["angle"] - o["angle"]) > 8:
                continue
            if abs(b["char_h"] - o["char_h"]) > 0.45 * b["char_h"]:
                continue
            # distance along the axis perpendicular to text flow
            if o["orient"] == "vertical":
                d = abs(b["cx"] - o["cx"])
            else:
                d = abs(b["cy"] - o["cy"])
            if d < 0.7 * b["char_h"] and (best is None or d < best[0]):
                best = (d, b)
        if best is None:
            blocks.append({"orient": o["orient"], "angle": o["angle"], "char_h": o["char_h"],
                           "cx": o["cx"], "cy": o["cy"], "occ": [o]})
        else:
            b = best[1]
            b["occ"].append(o)
            k = len(b["occ"])
            for key in ("angle", "char_h", "cx", "cy"):
                b[key] += (o[key] - b[key]) / k

    # a few detections of a much smaller/larger size inside another block's area with similar text are that
    # block caught mid-animation (pop / scale-in) - fold them in instead of reporting a separate block
    def box(b):
        xs = [p[0] for o in b["occ"] for p in o["poly"]]
        ys = [p[1] for o in b["occ"] for p in o["poly"]]
        return min(xs), min(ys), max(xs), max(ys)

    for small in sorted(blocks, key=lambda b: len(b["occ"])):
        if len(small["occ"]) > 3:
            continue
        for big in blocks:
            if big is small or len(big["occ"]) <= 3 * len(small["occ"]):
                continue
            x0, y0, x1, y1 = box(big)
            inside = all(x0 <= o["cx"] <= x1 and y0 <= o["cy"] <= y1 for o in small["occ"])
            similar = any(SequenceMatcher(None, o["text"], q["text"]).ratio() > 0.6
                          for o in small["occ"] for q in big["occ"] if abs(q["frame"] - o["frame"]) <= 3 * step_f)
            if inside and similar:
                big["occ"] += small["occ"]
                small["occ"] = []
                break
    blocks = [b for b in blocks if b["occ"]]

    # merge several detections of the same block in one frame (OCR may split a line)
    out_blocks = []
    for bi, b in enumerate(sorted(blocks, key=lambda b: (b["cy"], b["cx"]))):
        per_frame = {}
        for o in b["occ"]:
            per_frame.setdefault(o["frame"], []).append(o)
        samples = []
        for f, os_ in sorted(per_frame.items()):
            xs = [p[0] for o in os_ for p in o["poly"]]
            ys = [p[1] for o in os_ for p in o["poly"]]
            os_.sort(key=lambda o: (o["cy"], o["cx"]) if b["orient"] == "vertical" else o["cx"])
            samples.append({"frame": f, "text": " ".join(o["text"] for o in os_),
                            "bbox": [int(min(xs)), int(min(ys)), int(math.ceil(max(xs))), int(math.ceil(max(ys)))],
                            "polys": [o["poly"] for o in os_],
                            "char_h": float(np.median([o["char_h"] for o in os_])),
                            "text_color": os_[0]["text_color"], "plate_color": os_[0]["plate_color"]})
        # events: runs of consecutive samples with similar text
        events = []
        for s in samples:
            e = events[-1] if events else None
            if e and s["frame"] - e["last"] <= 2 * step_f and \
                    SequenceMatcher(None, s["text"], e["text"]).ratio() > 0.6:
                e["last"] = s["frame"]
                e["n"] += 1
                if len(s["text"]) > len(e["text"]):
                    e["text"] = s["text"]
                e["heights"].append(s["char_h"])
            else:
                events.append({"first": s["frame"], "last": s["frame"], "text": s["text"], "n": 1,
                               "heights": [s["char_h"]]})
        if sum(e["n"] for e in events) < 2 and max(o["score"] for o in b["occ"]) < 0.85:
            continue  # one-off low-confidence detection: likely noise
        ev_out = []
        pops = 0
        for e in events:
            med = float(np.median(e["heights"]))
            if len(e["heights"]) >= 3 and e["heights"][0] < 0.9 * med:
                pops += 1
            ev_out.append({"start_frame": max(0, e["first"] - step_f), "end_frame": min(v["frames"], e["last"] + step_f),
                           "first_seen": e["first"], "last_seen": e["last"], "text": e["text"],
                           "cps": round(len(e["text"].replace(" ", "")) / max(1 / fps, (e["last"] - e["first"] + step_f) / fps), 2)})
        has_cjk = any("一" <= c <= "鿿" for e in events for c in e["text"])
        distinct = len({e["text"] for e in events})
        union = [min(s["bbox"][0] for s in samples), min(s["bbox"][1] for s in samples),
                 max(s["bbox"][2] for s in samples), max(s["bbox"][3] for s in samples)]
        tcs = [s["text_color"] for s in samples if s["text_color"]]
        pcs = [s["plate_color"] for s in samples if s["plate_color"]]
        out_blocks.append({
            "id": f"B{len(out_blocks) + 1}",
            "kind": "subtitle" if distinct >= 3 else "static_text",
            "language": "zh" if has_cjk else "latin/other",
            "orient": b["orient"], "angle_deg": round(b["angle"], 1),
            "center": [round(b["cx"]), round(b["cy"])], "union_bbox": union,
            "char_height_px": round(b["char_h"], 1),
            "char_height_pct_of_frame_h": round(100 * b["char_h"] / H, 2),
            "text_color": max(set(tcs), key=tcs.count) if tcs else None,
            "plate_color": (max(set(pcs), key=pcs.count) if len(pcs) > len(samples) / 2 else None),
            "entrance_guess": "pop/scale-in" if pops > len(events) * 0.3 else "cut",
            "median_cps": float(np.median([e["cps"] for e in ev_out])),
            "coverage_s": round(len({f for e in ev_out for f in range(e["start_frame"], e["end_frame"])}) / fps, 1),
            "events": ev_out,
            "samples": samples,
        })

    save_json(P["ocr"], {"step_frames": step_f, "fps": fps, "width": W, "height": H, "blocks": out_blocks})
    overview(cfg["source"], out_blocks, P["analysis"] / "ocr_blocks.png", W, H)
    print(f"\n{len(out_blocks)} text blocks:")
    for b in out_blocks:
        print(f"  {b['id']}: {b['kind']} {b['language']} {b['orient']} angle={b['angle_deg']} "
              f"bbox={b['union_bbox']} char_h={b['char_height_px']}px color={b['text_color']} "
              f"plate={b['plate_color']} entrance={b['entrance_guess']} events={len(b['events'])} "
              f"first='{b['events'][0]['text'][:20]}'")
    print(f"\nWrote {P['ocr']}\nOverview image: {P['analysis'] / 'ocr_blocks.png'}  <- open it and show the user")


def overview(src, blocks, out_path, W, H):
    cap = cv2.VideoCapture(src)
    # pick the sampled frame where the most blocks are visible
    counts = {}
    for b in blocks:
        for s in b["samples"]:
            counts[s["frame"]] = counts.get(s["frame"], 0) + 1
    fidx = max(counts, key=counts.get) if counts else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, base = cap.read()
    if not ok:
        base = np.zeros((H, W, 3), np.uint8)
    pal = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]
    for i, b in enumerate(blocks):
        c = pal[i % len(pal)]
        x0, y0, x1, y1 = b["union_bbox"]
        cv2.rectangle(base, (x0, y0), (x1, y1), c, 2)
        cv2.putText(base, b["id"], (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    # crops: up to 3 different events per block
    rows = []
    for i, b in enumerate(blocks):
        crops = []
        for e in b["events"][:3]:
            s = next(s for s in b["samples"] if s["frame"] >= e["first_seen"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame"])
            ok, im = cap.read()
            if not ok:
                continue
            x0, y0, x1, y1 = s["bbox"]
            cr = im[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6]
            if cr.size:
                crops.append(cv2.resize(cr, (int(cr.shape[1] * 60 / max(cr.shape[0], 1)), 60)))
        if crops:
            row = np.hstack([np.full((60, 50, 3), 40, np.uint8)] + crops)
            cv2.putText(row, b["id"], (3, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pal[i % len(pal)], 2)
            rows.append(row)
    cap.release()
    if rows:
        wmax = max(W, max(r.shape[1] for r in rows))
        pad = lambda im: np.hstack([im, np.zeros((im.shape[0], wmax - im.shape[1], 3), np.uint8)])
        sheet = np.vstack([pad(base)] + [pad(r) for r in rows])
    else:
        sheet = base
    cv2.imwrite(str(out_path), sheet)


if __name__ == "__main__":
    main()
