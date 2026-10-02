"""Build captions.json (Remotion props) from the scheduled dub.

Timing comes from the dub itself: faster-whisper word timestamps on each placed TTS line,
aligned to the script words (falls back to character-proportional timing).
  - a card appears lead_frames (default 2 = ~66 ms at 30 fps) before its first word
  - gaps shorter than bridge_gap_frames are bridged (no flicker between cards)
  - words per card: captions.words_per_card ("auto" = like the original block, from OCR)
Look (size / position / angle) comes from job.json captions + the OCR anchor block.

  py -3.12 build_captions.py --job <job>
"""

import argparse
import difflib
import re
from pathlib import Path

from common import die, job_paths, load_config, load_json, save_json

CJK = ("zh", "ja", "ko")
FONT_FILES = {"impact": ("impact.ttf", "Impact.ttf"), "arial black": ("ariblk.ttf", "Arial Black.ttf"),
              "arial": ("arialbd.ttf", "Arial Bold.ttf"), "microsoft yahei": ("msyhbd.ttc",),
              "simhei": ("simhei.ttf",), "pingfang sc": ("PingFang.ttc",),
              "noto sans cjk sc": ("NotoSansCJK-Bold.ttc",)}


def font_for_measure(c):
    if c.get("font_file"):
        return c["font_file"]
    import hw
    fam = c["font_family"].split(",")[0].strip(" '\"").lower()
    f = hw.find_font(*FONT_FILES.get(fam, ()))
    if not f:
        die("no font file found for measuring captions; set captions.font_file in job.json")
    return f


def anchor_blocks(cfg, ocr):
    if not ocr:
        return []
    c = cfg["captions"]
    ids = [c["anchor_block"]] if c.get("anchor_block") else c.get("replace_blocks", [])
    bl = [b for b in ocr["blocks"] if b["id"] in ids]
    return bl


def resolve_style(cfg, ocr):
    """Absolute pixel style used by both check_script.py and the renderer."""
    c = cfg["captions"]
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    l1, l2 = dict(c["line1"]), dict(c["line2"])
    anchors = anchor_blocks(cfg, ocr)
    rotation, writing = 0.0, "horizontal-tb"
    if c["placement"] == "match_block" and anchors:
        x0 = min(b["union_bbox"][0] for b in anchors)
        y0 = min(b["union_bbox"][1] for b in anchors)
        x1 = max(b["union_bbox"][2] for b in anchors)
        y1 = max(b["union_bbox"][3] for b in anchors)
        center = [(x0 + x1) / 2, (y0 + y1) / 2]
        main = max(anchors, key=lambda b: b["coverage_s"])
        rotation = main["angle_deg"] if main["orient"] == "rotated" else 0.0
        if main["orient"] == "vertical":
            writing = "vertical-rl"
    elif c["placement"] == "custom":
        center = [c["custom_center"][0] * W, c["custom_center"][1] * H]
    else:  # bottom
        center = [0.5 * W, 0.78 * H]
    if c.get("rotation_deg") is not None:
        rotation = float(c["rotation_deg"])
    if c.get("size_mode") == "match" and anchors:
        # OCR box height ~ 1 em for caps/CJK text; keep the user's line1:line2 ratio
        ch = max(anchors, key=lambda b: b["coverage_s"])["char_height_px"]
        ratio = l2["size"] / l1["size"]
        l1["size"] = round(ch * 0.95, 1)
        l2["size"] = round(l1["size"] * ratio, 1)
    max_w = c["max_width_ratio"] * (H if writing.startswith("vertical") else W)
    return {"center": center, "rotation_deg": rotation, "writing_mode": writing, "max_width_px": max_w,
            "line1": l1, "line2": l2, "font_family": c["font_family"], "font_file_measure": font_for_measure(c)}


def norm(w):
    return re.sub(r"[^\w']", "", w.lower())


def word_times(seg_file, tokens, t0, lang, model):
    """Absolute (start, end) per caption token, from whisper words on the placed TTS line."""
    import wave
    import numpy as np
    with wave.open(seg_file, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    x = x.reshape(-1, ch).mean(1)
    dur = len(x) / sr
    rec = []
    if model is not None:
        idx = (np.arange(int(len(x) * 16000 / sr)) * sr / 16000).astype(int)
        segs, _ = model.transcribe(x[idx], language=lang, word_timestamps=True, beam_size=5, vad_filter=False)
        rec = [(w.word, w.start, w.end) for s in segs for w in s.words]
    times = [None] * len(tokens)
    if rec and lang not in CJK:
        sm = difflib.SequenceMatcher(a=[norm(t) for t in tokens], b=[norm(r[0]) for r in rec], autojunk=False)
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                r = rec[blk.b + k]
                times[blk.a + k] = (r[1], r[2])
    if not any(times):  # proportional fallback over the whole line
        lens = [max(1, len(t)) for t in tokens]
        tot, acc = sum(lens), 0
        for i, L in enumerate(lens):
            times[i] = (dur * acc / tot, dur * (acc + L) / tot)
            acc += L
    # interpolate unmatched tokens between matched neighbours
    for i in range(len(times)):
        if times[i] is None:
            prev = next((times[j][1] for j in range(i - 1, -1, -1) if times[j]), 0.0)
            nxt_j = next((j for j in range(i + 1, len(times)) if times[j]), None)
            nxt = times[nxt_j][0] if nxt_j is not None else dur
            gapn = (nxt_j if nxt_j is not None else len(times)) - i
            step = (nxt - prev) / (gapn + 0)
            times[i] = (prev, prev + step)
    return [(t0 + a, t0 + b) for a, b in times]


def auto_words_per_card(cfg, ocr, lang):
    c = cfg["captions"]
    mx = c["max_words_per_line"] * (2 if c["two_lines"] else 1)
    wpc = c.get("words_per_card", "auto")
    if wpc != "auto":
        return int(wpc)
    anchors = [b for b in anchor_blocks(cfg, ocr) if b["kind"] == "subtitle"]
    counts = [len(e["text"].split()) for b in anchors if b["language"] != "zh" for e in b["events"]]
    if counts and lang not in CJK:
        counts.sort()
        return max(2, min(mx, counts[len(counts) // 2]))
    return mx


def split_lines(words, cfg, lang):
    c = cfg["captions"]
    n = c["max_words_per_line"]
    if not c["two_lines"] or len(words) <= n:
        return [words]
    k = (len(words) + 1) // 2  # balanced, first line the longer one
    return [words[:k], words[k:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--no-whisper", action="store_true", help="proportional timing only")
    a = ap.parse_args()
    P = job_paths(a.job)
    cfg = load_config(a.job)
    c = cfg["captions"]
    fps = cfg["video"]["fps"]
    total = cfg["video"]["frames"]
    ocr = load_json(P["ocr"]) if P["ocr"].exists() else None
    script = load_json(P["script"])
    lang = script.get("language", cfg["target_language"])
    tl = {t["id"]: t for t in load_json(P["timeline"])["segments"]}
    style = resolve_style(cfg, ocr)
    wpc = auto_words_per_card(cfg, ocr, lang)

    model = None
    if not a.no_whisper:
        try:
            from faster_whisper import WhisperModel
            from analyze_audio import load_whisper
            model = load_whisper(WhisperModel, "base")
        except SystemExit:
            print("WARN: no whisper model; using proportional timing")

    # 1) timed tokens
    toks = []
    for seg in script["segments"]:
        text = seg.get("caption") or seg["text"]
        if c.get("uppercase") and lang not in CJK:
            text = text.upper()
        tokens = list(text.replace(" ", "")) if lang in CJK else text.split()
        t = tl[seg["id"]]
        times = word_times(t["file"], tokens, t["start"], lang, model)
        for k, (tok, (s, e)) in enumerate(zip(tokens, times)):
            toks.append({"w": tok, "s": s, "e": e, "seg": seg["id"], "last": k == len(tokens) - 1})

    # 2) cards: up to wpc tokens, never across a dub line, break after sentence punctuation
    cards, cur = [], []
    per = wpc if lang not in CJK else c.get("max_chars_per_line", 12) * (2 if c["two_lines"] else 1)
    for tk in toks:
        cur.append(tk)
        if len(cur) >= per or tk["last"] or (re.search(r"[.!?。！？]$", tk["w"]) and len(cur) >= 2):
            cards.append(cur)
            cur = []
    if cur:
        cards.append(cur)

    lead, bridge = int(c["lead_frames"]), int(c["bridge_gap_frames"])
    hold = int(round(0.25 * fps))
    out = []
    for i, cd in enumerate(cards):
        f0 = max(0, round(cd[0]["s"] * fps) - lead)
        f1 = round(cd[-1]["e"] * fps) + hold
        out.append({"from": f0, "to": f1, "seg": cd[0]["seg"],
                    "words": [{"w": t["w"], "f": max(0, round(t["s"] * fps) - lead)} for t in cd]})
    for i in range(len(out)):
        nxt = out[i + 1]["from"] if i + 1 < len(out) else total
        if nxt - out[i]["to"] < bridge:  # bridge small gaps / cut overlaps
            out[i]["to"] = nxt
        out[i]["to"] = min(out[i]["to"], nxt)
        minf = round(c["min_duration_s"] * fps)
        if out[i]["to"] - out[i]["from"] < minf:
            out[i]["to"] = min(nxt, out[i]["from"] + minf)
    for cd in out:
        words = [w["w"] for w in cd["words"]]
        if lang in CJK:
            n = c.get("max_chars_per_line", 12)
            s = "".join(words)
            cd["lines"] = [s[:n], s[n:]] if len(s) > n and c["two_lines"] else [s]
        else:
            cd["lines"] = [" ".join(l) for l in split_lines(words, cfg, lang)]

    props = {
        "fps": fps, "width": cfg["video"]["width"], "height": cfg["video"]["height"],
        "durationInFrames": total, "video": "clean.mp4", "lang": lang,
        "style": {
            "center": style["center"], "rotationDeg": style["rotation_deg"], "writingMode": style["writing_mode"],
            "maxWidth": style["max_width_px"], "fontFamily": c["font_family"],
            "fontFile": Path(c["font_file"]).name if c.get("font_file") else None,
            "line1": style["line1"], "line2": style["line2"], "lineGap": c.get("line_gap", 0.05),
            "strokePx": c["stroke_px"], "strokeColor": c["stroke_color"], "shadow": c["shadow"],
            "plate": c.get("plate"), "animation": c["animation"], "highlight": c.get("highlight"),
        },
        "cards": out,
    }
    save_json(P["captions"], props)
    print(f"{len(out)} caption cards (<= {per} {'chars' if lang in CJK else 'words'} each), "
          f"center {[round(v) for v in style['center']]}, size {style['line1']['size']}/{style['line2']['size']}px, "
          f"rotation {style['rotation_deg']}deg -> {P['captions']}")
    for cd in out[:6]:
        print(f"  {cd['from']:>5}-{cd['to']:<5} {' / '.join(cd['lines'])}")


if __name__ == "__main__":
    main()
