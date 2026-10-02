"""Validate script.json (the translated dub script written by the agent) BEFORE any TTS call.

script.json:
{
  "language": "en",
  "segments": [
    {"id": 0, "src_ids": [0], "speaker": "S1",
     "text": "Spoken line for TTS.",
     "caption": "Optional caption text if it should differ from the spoken text"}
  ]
}
start/end are taken from the source segments in src_ids (analysis/transcript.json).

Checks (exit 1 on any ERROR):
  - every source speech segment is covered exactly once, in order
  - speaker exists in speakers.json and has a voice in job.json dub.speaker_voices
  - word budget: words <= floor(window * rate)        (rate: dub.target_rate or measured/default)
  - every caption word fits: longest line in pixels <= max_width_ratio * frame width

  py -3.12 check_script.py --job <job>
"""

import argparse
import math

from common import count_words, die, job_paths, load_config, load_json, save_json

# Comfortable short-video speaking rates when the source language differs from the target.
DEFAULT_RATE = {"en": 2.7, "zh": 4.5, "ja": 6.0, "ko": 4.5, "es": 3.0, "fr": 3.0, "de": 2.5, "pt": 3.0}
CJK = ("zh", "ja", "ko")


def windows(script, transcript, duration):
    """start/end per script segment + the free slot until the next one starts."""
    src = {s["id"]: s for s in transcript["segments"]}
    out = []
    for seg in script["segments"]:
        ss = [src[i] for i in seg["src_ids"]]
        out.append({"start": min(s["start"] for s in ss), "end": max(s["end"] for s in ss)})
    for k, w in enumerate(out):
        w["slot_end"] = out[k + 1]["start"] if k + 1 < len(out) else duration
    return out


def budget_rate(cfg, transcript, lang):
    if cfg["dub"].get("target_rate"):
        return float(cfg["dub"]["target_rate"]), "job.json dub.target_rate"
    if transcript["language"] == lang:
        return float(transcript["speech_rate"]), "measured from the original speech"
    return DEFAULT_RATE.get(lang, 2.7), f"default for '{lang}' (source language differs)"


def caption_lines(text, cfg, lang):
    """Same splitting rule as build_captions.py: cards of <= 2 lines x max_words_per_line."""
    c = cfg["captions"]
    if c.get("uppercase") and lang not in CJK:
        text = text.upper()
    if lang in CJK:  # no spaces: split by characters
        n = c.get("max_chars_per_line", 12)
        return [text[i:i + n] for i in range(0, len(text), n)]
    words = text.split()
    n = c.get("max_words_per_line", 5)
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def line_width_px(line, size, spacing, stroke, font_file):
    from PIL import ImageFont
    font = ImageFont.truetype(font_file, max(1, round(size)))
    return font.getlength(line) + spacing * max(0, len(line) - 1) + 2 * stroke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    P = job_paths(a.job)
    cfg = load_config(a.job)
    tr = load_json(P["transcript"])
    spk = {s["id"] for s in load_json(P["speakers"])["speakers"]}
    if not P["script"].exists():
        die(f"{P['script']} missing. Write the translated script first (format in this file's docstring).")
    script = load_json(P["script"])
    lang = script.get("language", cfg["target_language"])
    errors, warns = [], []

    used = [i for s in script["segments"] for i in s["src_ids"]]
    all_ids = [s["id"] for s in tr["segments"]]
    missing = sorted(set(all_ids) - set(used))
    dup = sorted({i for i in used if used.count(i) > 1})
    if missing:
        errors.append(f"source segments not covered: {missing}")
    if dup:
        errors.append(f"source segments used twice: {dup}")
    if used != sorted(used):
        errors.append("src_ids are out of order")
    if errors:
        for e in errors:
            print("ERROR: " + e)
        raise SystemExit(1)

    voices = cfg["dub"]["speaker_voices"]
    rate, why = budget_rate(cfg, tr, lang)
    max_tempo = cfg["dub"]["max_tempo"]
    W = cfg["video"]["width"]
    cap = cfg["captions"]
    from build_captions import resolve_style  # same size logic as the renderer
    style = resolve_style(cfg, load_json(P["ocr"]) if P["ocr"].exists() else None)
    max_w = style["max_width_px"]
    print(f"Budget rate {rate:.2f} {'chars' if lang in CJK else 'words'}/s ({why}); "
          f"caption max width {max_w:.0f}px of {W}px\n")

    win = windows(script, tr, cfg["video"]["duration"])
    report = []
    for seg, w in zip(script["segments"], win):
        sid = seg.get("speaker", "S1")
        if sid not in spk:
            errors.append(f"seg {seg['id']}: unknown speaker {sid}")
        if sid not in voices:
            errors.append(f"seg {seg['id']}: no voice for {sid} in job.json dub.speaker_voices")
        n = count_words(seg["text"], "zh" if lang in CJK else "en")
        span = w["end"] - w["start"]
        slot = w["slot_end"] - w["start"]
        budget = math.floor(span * rate)
        hard = math.floor(slot * rate * max_tempo)  # what can still fit with speed-up into the gap
        status = "ok"
        if n > hard:
            status = "TOO LONG"
            errors.append(f"seg {seg['id']}: {n} words, hard limit {hard} (window {span:.2f}s + gap). Shorten.")
        elif n > budget:
            status = "tight"
            warns.append(f"seg {seg['id']}: {n} words > budget {budget}; will need speed-up or the gap.")
        elif n < 0.5 * budget and span > 2:
            status = "short"
            warns.append(f"seg {seg['id']}: only {n} words for {span:.1f}s; long silence. Consider a fuller line.")
        for line in caption_lines(seg.get("caption") or seg["text"], cfg, lang):
            for li, st in enumerate([style["line1"], style["line2"]]):
                px = line_width_px(line, st["size"], st["letter_spacing"], cap["stroke_px"], style["font_file_measure"])
                if px > max_w:
                    errors.append(f"seg {seg['id']}: caption line '{line}' is {px:.0f}px > {max_w:.0f}px "
                                  f"(line{li + 1} style). Fewer words per line or smaller size.")
                    break
        report.append({"id": seg["id"], "start": round(w["start"], 2), "end": round(w["end"], 2),
                       "slot_end": round(w["slot_end"], 2), "words": n, "budget": budget,
                       "hard_limit": hard, "status": status})
        print(f"  seg {seg['id']:>3} {w['start']:6.2f}-{w['end']:6.2f}s  {n:>3}/{budget:<3} words  {status:9} {seg['text'][:60]}")

    save_json(P["analysis"] / "script_check.json", {"rate": rate, "segments": report,
                                                     "errors": errors, "warnings": warns})
    print()
    for w_ in warns:
        print("WARN: " + w_)
    for e in errors:
        print("ERROR: " + e)
    if errors:
        raise SystemExit(1)
    print("Script OK.")


if __name__ == "__main__":
    main()
