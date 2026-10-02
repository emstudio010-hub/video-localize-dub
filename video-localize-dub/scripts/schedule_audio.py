"""Place each TTS segment on the original speech timeline, never overlapping, tempo inside
[dub.min_tempo, dub.max_tempo]; mix with the background; pad to the exact video length.

Rules per segment k (times from the source speech it replaces):
  1. start at the source start (or right after the previous dub line + min_gap)
  2. it must end before the next source line starts (- min_gap)
  3. too long  -> speed up (<= max_tempo); still too long -> start up to max_early_shift_s earlier
  4. clearly shorter than the source line -> slow down slightly (>= min_tempo) for an even pace
  5. still does not fit -> listed as FAIL, nothing is written, exit 1: shorten those lines
     in script.json, run tts.py again (only changed lines are re-synthesized), re-run this.

  py -3.12 schedule_audio.py --job <job>
Writes audio/dub.wav, audio/voice.wav, audio/segs/*.wav, audio/timeline.json
"""

import argparse
import subprocess

import numpy as np

from check_script import windows
from common import die, job_paths, load_config, load_json, save_json

SR = 48000


def decode(path, tempo=1.0):
    af = []
    if abs(tempo - 1) > 1e-3:
        af = ["-af", f"atempo={tempo:.4f}"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), *af, "-ac", "2", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                        "-c:a", "pcm_s16le", str(path)], input=np.ascontiguousarray(x, np.float32).tobytes())
    if p.returncode:
        die(f"could not write {path}")


def plan(script, tr, tts, dub, duration):
    win = windows(script, tr, duration)
    gap, lo_t, hi_t, early = dub["min_gap_s"], dub["min_tempo"], dub["max_tempo"], dub["max_early_shift_s"]
    out, fails, prev_end = [], [], 0.0
    for k, (seg, w) in enumerate(zip(script["segments"], win)):
        d = tts[seg["id"]]["duration"]
        latest_end = (w["slot_end"] - gap) if k + 1 < len(win) else duration - 0.02
        start = max(w["start"], prev_end + gap if k else 0.0)
        span = w["end"] - w["start"]
        tempo = 1.0
        if d < 0.9 * span:  # rule 4
            tempo = max(lo_t, d / span)
        if start + d / tempo > latest_end:  # rule 3
            tempo = max(tempo, d / max(1e-3, latest_end - start))
            if tempo > hi_t:
                earliest = max(prev_end + gap if k else 0.0, w["start"] - early)
                start = max(earliest, latest_end - d / hi_t)
                tempo = max(1.0, d / max(1e-3, latest_end - start))
        end = start + d / tempo
        ok = tempo <= hi_t + 1e-6 and end <= latest_end + 1e-3
        if not ok:
            need = d / hi_t - (latest_end - max(prev_end + gap if k else 0.0, w["start"] - early))
            fails.append({"id": seg["id"], "tts_s": d, "room_s": round(latest_end - w["start"] + early, 2),
                          "over_by_s": round(need, 2), "text": seg["text"]})
        out.append({"id": seg["id"], "speaker": seg.get("speaker", "S1"), "src_start": round(w["start"], 3),
                    "src_end": round(w["end"], 3), "start": round(start, 3), "end": round(end, 3),
                    "tempo": round(tempo, 4), "tts_s": d, "shift_s": round(start - w["start"], 3)})
        prev_end = end
    return out, fails


def background(cfg, P, n):
    dub = cfg["dub"]
    mode = dub.get("background", "none")
    src = {"separate": P["bgm"], "original": P["audio_full"], "file": dub.get("background_file")}.get(mode)
    if mode == "none" or src is None:
        return np.zeros((n, 2), np.float32), mode
    if mode == "separate" and not P["bgm"].exists():
        die("dub.background = separate but analysis/bgm.wav missing. Run separate_bgm.py (or change the mode).")
    x = decode(src)
    if mode == "file" and len(x) < n:  # loop a music bed
        x = np.tile(x, (n // max(1, len(x)) + 1, 1))
    x = x[:n]
    if len(x) < n:
        x = np.vstack([x, np.zeros((n - len(x), 2), np.float32)])
    return x, mode


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) if len(x) else -120.0


def level(voice, bg, dub):
    """Voice to dub.voice_level_db (RMS while speaking); music dub.background_db_below_voice under it,
    ducked by dub.duck_db while the voice is active (smoothed, no pumping clicks)."""
    env = np.abs(voice).max(1)
    active = env > 1e-3
    v_db = rms_db(voice[active]) if active.any() else -120
    voice = voice * 10 ** ((dub.get("voice_level_db", -18) - v_db) / 20) if active.any() else voice
    if not np.abs(bg).any():
        return voice, bg
    target = dub.get("voice_level_db", -18) - dub.get("background_db_below_voice", 14)
    bg = bg * 10 ** ((target - rms_db(bg)) / 20) * float(dub.get("background_volume", 1.0))
    win = int(0.15 * SR)
    duck = np.convolve(active.astype(np.float32), np.ones(win) / win, "same")
    duck = np.clip(duck * 3, 0, 1)
    bg = bg * (1 - duck * (1 - 10 ** (-dub.get("duck_db", 4) / 20)))[:, None]
    return voice, bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    P = job_paths(a.job)
    cfg = load_config(a.job)
    dub = cfg["dub"]
    duration = cfg["video"]["duration"]
    script = load_json(P["script"])
    tts = {s["id"]: s for s in load_json(P["tts_index"])["segments"]}
    missing = [s["id"] for s in script["segments"] if s["id"] not in tts or tts[s["id"]]["text"] != s["text"]]
    if missing:
        die(f"TTS missing or stale for segments {missing}. Run tts.py first.")
    tl, fails = plan(script, load_json(P["transcript"]), tts, dub, duration)

    for t in tl:
        print(f"  seg {t['id']:>3} src {t['src_start']:6.2f}-{t['src_end']:6.2f}  dub {t['start']:6.2f}-{t['end']:6.2f}"
              f"  tempo {t['tempo']:.3f}  shift {t['shift_s']:+.2f}s")
    if fails:
        print("\nFAIL - these lines do not fit even at max tempo + early start. Shorten them:")
        for f in fails:
            print(f"  seg {f['id']}: needs {f['over_by_s']:.2f}s less  \"{f['text']}\"")
        from common import count_words
        lang = script.get("language", cfg["target_language"])
        unit = "zh" if lang in ("zh", "ja", "ko") else "en"
        tts_rate = sum(count_words(s["text"], unit) for s in script["segments"]) / \
            max(1e-6, sum(tts[s["id"]]["duration"] for s in script["segments"]))
        src = load_json(P["transcript"])
        print(f"\nTTS speaks {tts_rate:.2f} {'chars' if unit == 'zh' else 'words'}/s"
              + (f"; original speech {src['speech_rate']:.2f}" if src["language"] == lang else "")
              + ". Options: shorten the lines above, and/or ask for a brisker pace in "
                "dub.style_instruction, then re-run tts.py.")
        save_json(P["timeline"].parent / "schedule_fail.json", fails)
        raise SystemExit(1)

    temps = [t["tempo"] for t in tl]
    if max(temps) - min(temps) > 0.10:
        print(f"WARN: tempo varies {min(temps):.2f}-{max(temps):.2f}; pace may sound uneven. "
              "Consider rewriting the fastest lines shorter.")

    n = int(round(duration * SR))
    voice = np.zeros((n, 2), np.float32)
    segdir = P["timeline"].parent / "segs"
    for t in tl:
        x = decode(tts[t["id"]]["file"], t["tempo"]) * float(dub.get("voice_volume", 1.0))
        f = segdir / f"seg_{t['id']:03d}.wav"
        write_wav(f, x)
        t["file"] = str(f)
        i = int(round(t["start"] * SR))
        x = x[: max(0, n - i)]
        voice[i:i + len(x)] += x
    bg, mode = background(cfg, P, n)
    voice, bg = level(voice, bg, dub)
    mix = voice + bg
    peak = float(np.abs(mix).max()) if len(mix) else 0
    if peak > 0.98:
        mix *= 0.98 / peak
    write_wav(P["timeline"].parent / "voice.wav", voice)
    write_wav(P["dub"], mix)
    save_json(P["timeline"], {"sr": SR, "duration": duration, "background": mode, "segments": tl})
    print(f"\nDub: {P['dub']}  ({n / SR:.3f}s = video length, background={mode}, peak scale "
          f"{'%.2f' % (0.98 / peak) if peak > 0.98 else '1.00'})")


if __name__ == "__main__":
    main()
