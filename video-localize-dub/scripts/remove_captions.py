"""Remove / hide original burned-in text blocks, per block, per frame, with time gating.

Modes per block (job.json removal.blocks.<id>.mode, else removal.default):
  inpaint  - per-pixel text mask + inpainting (engine opencv, or lama if installed)
  blur     - blur only inside the block's text area (feathered). keys: strength (px, default 25)
  color    - colored plate over the text. keys: color "#000000", opacity 0..1, radius px
  keep     - leave untouched
Optional per-block keys: padding_px, mask ("stroke" default | "box").

  py -3.12 remove_captions.py --job <job> --preview            # before/after PNGs only
  py -3.12 remove_captions.py --job <job> --preview --at 3.2,10.5
  py -3.12 remove_captions.py --job <job>                      # full video -> video/clean.mp4
                                     (resumable: rerun after a crash continues; --restart discards)
Device: LaMa runs on CUDA / MPS / DirectML when available (hw.py), else CPU.
"""

import argparse
import json
import subprocess
import time

import cv2
import numpy as np

from common import die, hex_to_bgr, job_paths, load_config, load_json

import hw

_LAMA = None
LAMA_DEVICE = "cpu"


def load_lama(cfg):
    """LaMa on the best device of this machine (hardware.device in job.json overrides)."""
    global _LAMA, LAMA_DEVICE
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        die("engine 'lama' needs simple-lama-inpainting: run setup.py (installs it and re-fixes Pillow)")
    import torch
    torch.set_num_threads(hw.threads(cfg))
    dev = hw.torch_device(cfg)
    try:
        _LAMA = SimpleLama(device=hw.torch_device_obj(dev))
        LAMA_DEVICE = dev
    except Exception as e:
        if dev == "cpu":
            raise
        print(f"WARNING: LaMa on {dev} failed ({type(e).__name__}: {e}); falling back to CPU")
        _LAMA = SimpleLama(device=torch.device("cpu"))
        LAMA_DEVICE = "cpu"
    print(f"LaMa device: {LAMA_DEVICE}")


def lama_inpaint(img, mask):
    global _LAMA, LAMA_DEVICE
    if _LAMA is None:
        load_lama(None)
    from PIL import Image
    pi, pm = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), Image.fromarray(mask)
    try:
        res = _LAMA(pi, pm)
    except RuntimeError as e:  # GPU out of memory / unsupported op -> finish the job on CPU
        if LAMA_DEVICE == "cpu":
            raise
        print(f"WARNING: LaMa on {LAMA_DEVICE} failed mid-run ({str(e)[:80]}); continuing on CPU")
        from simple_lama_inpainting import SimpleLama
        import torch
        _LAMA, LAMA_DEVICE = SimpleLama(device=torch.device("cpu")), "cpu"
        res = _LAMA(pi, pm)
    out = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
    return out[: img.shape[0], : img.shape[1]]


class Block:
    def __init__(self, b, opts, fps, W, H):
        self.b = b
        self.o = opts
        self.W, self.H = W, H
        self.pad = int(opts.get("padding_px", 6))
        self.mode = opts.get("mode", "inpaint")
        self.mask_kind = opts.get("mask", "stroke" if b["orient"] != "rotated" else "box")
        # per event: (lo, hi, samples-in-event)
        self.events = []
        for e in b["events"]:
            ss = [s for s in b["samples"] if e["first_seen"] <= s["frame"] <= e["last_seen"]]
            self.events.append((e["start_frame"], e["end_frame"], ss))
        self.ref_density = None

    def sample_for(self, n):
        for lo, hi, ss in self.events:
            if lo <= n <= hi and ss:
                return min(ss, key=lambda s: abs(s["frame"] - n))
        return None

    def region_mask(self, s):
        """Filled (padded) polygons of the detected text lines."""
        m = np.zeros((self.H, self.W), np.uint8)
        for poly in s["polys"]:
            cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        # stroke masks search a wider band (plates, glows, shadows extend past the OCR box);
        # only pixels that differ from the background inside it get touched.
        pad = self.pad if self.mask_kind == "box" else max(self.pad, int(0.3 * self.b["char_height_px"]))
        k = 2 * pad + 1
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    def stroke_mask(self, img, region):
        """Pixels inside region that differ from a background interpolated across the band."""
        ys, xs = np.nonzero(region)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        crop = img[y0:y1, x0:x1].astype(np.float32)
        vertical = self.b["orient"] == "vertical"
        if vertical:
            a = img[y0:y1, max(0, x0 - 4):max(1, x0 - 1)].astype(np.float32).mean(1, keepdims=True) \
                if x0 > 1 else crop[:, :1]
            b = img[y0:y1, min(self.W - 1, x1 + 1):min(self.W, x1 + 4)].astype(np.float32).mean(1, keepdims=True) \
                if x1 < self.W - 1 else crop[:, -1:]
            t = np.linspace(0, 1, x1 - x0)[None, :, None]
        else:
            a = img[max(0, y0 - 4):max(1, y0 - 1), x0:x1].astype(np.float32).mean(0, keepdims=True) \
                if y0 > 1 else crop[:1]
            b = img[min(self.H - 1, y1 + 1):min(self.H, y1 + 4), x0:x1].astype(np.float32).mean(0, keepdims=True) \
                if y1 < self.H - 1 else crop[-1:]
            t = np.linspace(0, 1, y1 - y0)[:, None, None]
        bg = a * (1 - t) + b * t
        diff = np.abs(crop - bg).max(2)
        # hysteresis: faint parts (semi-transparent plates, glow, fade-in edges) count when they touch real text
        thr = float(self.o.get("diff_threshold", 30))
        strong = diff > thr
        weak = (diff > thr * 0.4).astype(np.uint8)
        nlab, lab = cv2.connectedComponents(weak, connectivity=8)
        keep = np.zeros(nlab, bool)
        keep[np.unique(lab[strong])] = True
        keep[0] = False
        m = keep[lab].astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        r = max(2, int(self.b["char_height_px"] / 9))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        full = np.zeros((self.H, self.W), np.uint8)
        full[y0:y1, x0:x1] = m
        return full & region

    def target(self, img, n):
        """(region, mask, box) of this block's text on frame n, or None when it is not on screen."""
        if self.mode == "keep":
            return None
        s = self.sample_for(n)
        if s is None:
            return None
        region = self.region_mask(s)
        if not region.any():
            return None
        if self.mask_kind == "stroke":
            mask = self.stroke_mask(img, region)
            dens = mask.sum() / max(1, region.sum())
            if self.ref_density is None:
                self.ref_density = dens
            # time gate inside the event window: text really present on this frame?
            if dens < 0.25 * self.ref_density or not mask.any():
                return None
            self.ref_density = 0.9 * self.ref_density + 0.1 * dens
            # plus the OCR text boxes themselves: plates / faint fills inside them can look like background when
            # the band edges used for the background estimate are not background (another caption right above)
            if self.mode == "inpaint" and self.o.get("include_boxes", True):
                core = np.zeros_like(mask)
                for poly in s["polys"]:
                    cv2.fillPoly(core, [np.array(poly, np.int32)], 255)
                k = 2 * max(2, int(0.15 * self.b["char_height_px"])) + 1
                mask = mask | cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        else:
            mask = region
        if self.mode == "inpaint":
            # grow the hole past glow / shadow / anti-aliasing: any hairline left on the border is continued INTO the
            # hole by the inpainter (seen as colored streaks / flat patches)
            g = int(self.o.get("grow_px", max(4, round(0.3 * self.b["char_height_px"]))))
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
            region = region | mask
        ys, xs = np.nonzero(region)
        # context around the hole: LaMa needs a lot of real surrounding image to continue textures/edges
        m = 24 if self.o.get("engine") != "lama" else int(self.o.get("context_px", max(96, 0.6 * (ys.max() - ys.min()))))
        box = [max(0, xs.min() - m), max(0, ys.min() - m), min(self.W, xs.max() + 1 + m), min(self.H, ys.max() + 1 + m)]
        return region, mask, box

    def effect(self, img, region, mask, box):
        """blur / color modes (inpaint is batched in process())."""
        x0, y0, x1, y1 = box
        crop, mk = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        if self.mode == "blur":
            k = int(self.o.get("strength", 25)) | 1
            blurred = cv2.GaussianBlur(crop, (k, k), 0)
            soft = cv2.GaussianBlur(region[y0:y1, x0:x1].astype(np.float32) / 255, (9, 9), 0)[..., None]
            out = (blurred * soft + crop * (1 - soft)).astype(np.uint8)
        elif self.mode == "color":
            color = hex_to_bgr(self.o.get("color", "#000000"))
            op = float(self.o.get("opacity", 1.0))
            rad = int(self.o.get("radius", 10))
            rys, rxs = np.nonzero(region[y0:y1, x0:x1])
            plate = np.zeros(mk.shape, np.uint8)
            cv2.rectangle(plate, (int(rxs.min()) + rad, int(rys.min())), (int(rxs.max()) - rad, int(rys.max())), 255, -1)
            cv2.rectangle(plate, (int(rxs.min()), int(rys.min()) + rad), (int(rxs.max()), int(rys.max()) - rad), 255, -1)
            for cx, cy in ((rxs.min() + rad, rys.min() + rad), (rxs.max() - rad, rys.min() + rad),
                           (rxs.min() + rad, rys.max() - rad), (rxs.max() - rad, rys.max() - rad)):
                cv2.circle(plate, (int(cx), int(cy)), rad, 255, -1)
            a = cv2.GaussianBlur(plate.astype(np.float32) / 255, (3, 3), 0)[..., None] * op
            out = (np.array(color, np.float32) * a + crop * (1 - a)).astype(np.uint8)
        else:
            die(f"unknown mode {self.mode} for {self.b['id']}")
        img = img.copy()
        img[y0:y1, x0:x1] = out
        return img


def inpaint(img, box, mask, engine):
    x0, y0, x1, y1 = box
    crop, mk = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    pred = lama_inpaint(crop, mk) if engine == "lama" else cv2.inpaint(crop, mk, 5, cv2.INPAINT_TELEA)
    # only masked pixels change (feathered edge); everything else keeps the original sharpness
    soft = cv2.GaussianBlur(cv2.dilate(mk, np.ones((3, 3), np.uint8)).astype(np.float32) / 255, (5, 5), 0)[..., None]
    img = img.copy()
    img[y0:y1, x0:x1] = (pred * soft + crop * (1 - soft)).round().astype(np.uint8)
    return img


def process(blocks, img, n):
    """All blocks on one frame. Overlapping inpaint areas share one inpaint call (LaMa is the slow part)."""
    hits, groups = [], []
    for b in blocks:
        t = b.target(img, n)
        if t:
            hits.append((b, *t))
    for b, region, mask, box in hits:
        if b.mode != "inpaint":
            continue
        eng = b.o.get("engine", "opencv")
        g = [box, mask.copy(), eng]
        for h in groups[:]:
            hb = h[0]
            if h[2] == eng and not (box[2] < hb[0] or hb[2] < box[0] or box[3] < hb[1] or hb[3] < box[1]):
                g = [[min(g[0][0], hb[0]), min(g[0][1], hb[1]), max(g[0][2], hb[2]), max(g[0][3], hb[3])],
                     g[1] | h[1], eng]
                groups.remove(h)
        groups.append(g)
    for box, mask, eng in groups:
        img = inpaint(img, box, mask, eng)
    for b, region, mask, box in hits:
        if b.mode != "inpaint":
            img = b.effect(img, region, mask, box)
    return img, bool(hits)


def build_blocks(cfg, ocr):
    rem = cfg["removal"]
    blocks = []
    for b in ocr["blocks"]:
        opts = {**rem.get("default", {}), **rem.get("blocks", {}).get(b["id"], {})}
        opts.setdefault("engine", rem.get("engine", "opencv"))
        opts.setdefault("padding_px", rem.get("padding_px", 6))
        blocks.append(Block(b, opts, ocr["fps"], ocr["width"], ocr["height"]))
    return blocks


def estimate(blocks, total, device):
    """Run time ~= frames_with_text x s_per_frame(engine, device) + total_frames x 0.01 (decode/encode).
    s_per_frame (544x960, scales ~ with hole area): lama cpu 2.9 | cuda 0.15 | mps 0.6 | directml 0.8;
    opencv 0.1; blur / color 0.02."""
    per_frame = {}
    for b in blocks:
        cost = hw.removal_spf(b.o.get("engine", "opencv"), device) if b.mode == "inpaint" else 0.02
        for lo, hi, _ in b.events:
            for n in range(lo, min(hi, total - 1) + 1):
                per_frame[n] = max(per_frame.get(n, 0), cost)  # overlapping blocks share one inpaint call
    return len(per_frame), sum(per_frame.values()) + total * 0.01


CHUNK = 150  # frames per resumable segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--at", default=None, help="comma separated seconds for preview")
    ap.add_argument("--restart", action="store_true", help="discard finished chunks of an interrupted run")
    a = ap.parse_args()
    P = job_paths(a.job)
    cfg = load_config(a.job)
    ocr = load_json(P["ocr"])
    fps, W, H = ocr["fps"], ocr["width"], ocr["height"]
    blocks = build_blocks(cfg, ocr)
    active = [b for b in blocks if b.mode != "keep"]
    print("Removal plan: " + ", ".join(f"{b.b['id']}={b.mode}/{b.mask_kind}" for b in blocks))
    uses_lama = any(b.mode == "inpaint" and b.o.get("engine") == "lama" for b in active)
    if uses_lama:
        load_lama(cfg)
    nf, est = estimate(active, cfg["video"]["frames"], LAMA_DEVICE if uses_lama else "cpu")
    print(f"Estimate: {nf} frames with text, full run ~{max(1, round(est / 60))} min on this machine "
          f"(device {LAMA_DEVICE if uses_lama else 'cpu'})")

    if a.preview:
        if a.at:
            frames = [round(float(t) * fps) for t in a.at.split(",")]
        else:  # the middle of up to 3 events per active block + a frame just after an event starts
            frames = []
            for b in active:
                evs = b.b["events"]
                for e in evs[:: max(1, len(evs) // 3)][:3]:
                    frames.append((e["first_seen"] + e["last_seen"]) // 2)
                frames.append(min(evs[0]["first_seen"] + 2, cfg["video"]["frames"] - 1))
            frames = sorted(set(frames))[:8]
        P["preview"].mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(cfg["source"])
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            for b in active:  # density reference: prime from this frame
                b.ref_density = None
            out, _ = process(active, img, f)
            sbs = np.hstack([img, np.full((H, 8, 3), 255, np.uint8), out])
            p = P["preview"] / f"removal_f{f:06d}.png"
            cv2.imwrite(str(p), sbs)
            print(f"preview (left=original, right=processed): {p}")
        return

    # full run in resumable chunks: video/clean_parts/part_XXXXX.mp4 (a killed run continues where it stopped)
    P["clean"].parent.mkdir(parents=True, exist_ok=True)
    parts = P["clean"].parent / "clean_parts"
    if a.restart:
        import shutil
        shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir(exist_ok=True)
    stamp = parts / "plan.json"
    plan = json.dumps({"removal": cfg["removal"], "ocr_mtime": P["ocr"].stat().st_mtime, "chunk": CHUNK},
                      sort_keys=True)
    if stamp.exists() and stamp.read_text(encoding="utf-8") != plan:
        print("removal settings changed since the last run - discarding finished chunks")
        for f in parts.glob("part_*.mp4"):
            f.unlink()
    stamp.write_text(plan, encoding="utf-8")
    enc = hw.encoder_args(cfg, max(10, int(cfg["render"].get("crf", 17)) - 2))
    print("Encoder: " + enc[1])
    total = cfg["video"]["frames"]
    cap = cv2.VideoCapture(cfg["source"])
    n = touched = done_before = 0
    t0 = time.time()
    ff = None
    while True:
        ci = n // CHUNK
        part = parts / f"part_{ci:05d}.mp4"
        if n % CHUNK == 0:
            if ff:
                close_part(ff)
            ff = None
            if n >= total:
                break
            if part.exists():  # finished in an earlier run: skip decoding work
                for _ in range(CHUNK):
                    if not cap.grab():
                        break
                    n += 1
                    done_before += 1
                if n % CHUNK:
                    break
                continue
            ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{W}x{H}", "-r", cfg["video"]["fps_str"], "-i", "-", *enc,
                                   str(part.with_suffix(".tmp.mp4"))], stdin=subprocess.PIPE)
        ok, img = cap.read()
        if not ok:
            break
        img, hit = process(active, img, n)
        touched += hit
        ff.stdin.write(img.tobytes())
        n += 1
        if n % CHUNK == 0:
            el = time.time() - t0
            new = n - done_before
            print(f"  {n}/{total} frames, {touched} processed, {el:.0f}s elapsed, "
                  f"~{el / max(1, new) * (total - n):.0f}s left", flush=True)
    cap.release()
    if ff:
        close_part(ff)
    if done_before:
        print(f"resumed: {done_before} frames were already done")
    if n != total:
        print(f"WARNING: decoded {n} frames, expected {total}")
    lst = parts / "list.txt"
    files = sorted(parts.glob("part_*[0-9].mp4"))
    lst.write_text("".join(f"file '{f.as_posix()}'\n" for f in files), encoding="utf-8")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(P["clean"])])
    if r.returncode:
        die("ffmpeg concat failed")
    from common import count_frames
    got = count_frames(P["clean"])
    if got != total:
        die(f"clean.mp4 has {got} frames, expected {total}; rerun with --restart")
    import shutil
    shutil.rmtree(parts, ignore_errors=True)
    print(f"Done: {P['clean']}  ({touched} frames modified this run, {time.time() - t0:.0f}s)")


def close_part(ff):
    ff.stdin.close()
    ff.wait()
    if ff.returncode:
        die("ffmpeg encode failed")
    tmp = ff.args[-1]
    import os
    os.replace(tmp, tmp.replace(".tmp.mp4", ".mp4"))

if __name__ == "__main__":
    main()
