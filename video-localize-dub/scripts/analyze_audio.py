"""Phase 1 audio analysis: probe, extract audio, transcribe with word timestamps,
optionally proofread text with MiMo ASR, cluster speakers, measure speech rate.

  py -3.12 analyze_audio.py --job <job_dir> [--model small] [--mimo-asr] [--speakers N]

Writes analysis/transcript.json, analysis/speakers.json, analysis/speaker_samples/*.wav
"""

import argparse
import io
import wave
from pathlib import Path

import numpy as np

from common import die, job_paths, load_config, probe, run, save_json

SR = 16000


def read_wav16(path):
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            die("expected 16 kHz mono wav")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def wav_bytes(x):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


# ---------- speaker features (numpy only, approximate) ----------

def mel_fb(n_fft=512, n_mels=40):
    def hz2mel(h): return 2595 * np.log10(1 + h / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(60), hz2mel(7600), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[i - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


FB = mel_fb()
DCT = np.cos(np.pi / 40 * (np.arange(40)[None, :] + 0.5) * np.arange(1, 21)[:, None])


def frames(x, size=400, hop=160):
    if len(x) < size:
        return np.zeros((0, size))
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def seg_features(x):
    """-> (mfcc frames without c1 [n,19], median F0 or None). None if too short."""
    fr = frames(x)
    if len(fr) < 20:
        return None, None
    energy = (fr ** 2).mean(1)
    voiced = fr[energy > np.percentile(energy, 40)]
    spec = np.abs(np.fft.rfft(voiced * np.hamming(400), 512)) ** 2
    mfcc = (np.log(spec @ FB.T + 1e-8) @ DCT.T)[:, 1:]
    f0s = []
    for f in voiced[:: max(1, len(voiced) // 80)]:
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[len(f) - 1:]
        lo, hi = SR // 400, SR // 70
        if ac[0] <= 0:
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] / ac[0] > 0.35:
            f0s.append(SR / k)
    return mfcc, (float(np.median(f0s)) if len(f0s) >= 5 else None)


def cluster(feats, f0s, n_speakers=None, threshold=0.75):
    """Average-linkage clustering of segments.

    Distance = RMS gap of mean MFCCs (scaled by within-segment frame spread) + pitch term.
    Calibrated on MiMo voices: same voice 0.4-0.8, male vs female > 1.3, two voices of the
    same gender can be as close as 0.6 -> same-gender speakers are NOT reliably separated.
    """
    spread = np.concatenate([m - m.mean(0) for m in feats]).std(0) + 1e-6
    X = np.array([m.mean(0) for m in feats]) / spread
    n = len(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).mean(2))
    for i in range(n):
        for j in range(n):
            if f0s[i] and f0s[j]:
                D[i, j] += 0.5 * min(1.0, abs(np.log(f0s[i] / f0s[j])) / np.log(1.6))
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = D[np.ix_(clusters[i], clusters[j])].mean()
                if best is None or d < best[0]:
                    best = (d, i, j)
        if n_speakers is None and best[0] > threshold:
            break
        if n_speakers is not None and len(clusters) <= n_speakers:
            break
        d, i, j = best
        clusters[i] += clusters.pop(j)
    return clusters


def load_whisper(WhisperModel, size):
    """Online/cached first; if the Hub is unreachable fall back to any cached size. CUDA when available."""
    import hw
    dev, ct = hw.whisper_device()
    kw = {"device": dev, "compute_type": ct, "cpu_threads": hw.threads()}

    def make(name, **extra):
        try:
            return WhisperModel(name, **kw, **extra)
        except Exception as e:
            if dev == "cpu" or "local_files_only" in extra and "not found" in str(e).lower():
                raise
            print(f"WARNING: whisper on {dev} failed ({type(e).__name__}: {str(e)[:80]}); using CPU")
            kw.update(device="cpu", compute_type="int8")
            return WhisperModel(name, **kw, **extra)
    try:
        m = make(size)
        print(f"whisper '{size}' on {kw['device']}/{kw['compute_type']}")
        return m
    except Exception as e:
        print(f"WARNING: could not load whisper '{size}' ({type(e).__name__}).")
    for alt in ("large-v3", "medium", "small", "base", "tiny"):
        try:
            m = make(alt, local_files_only=True)
            print(f"Using cached whisper model '{alt}' instead. (China: set HF_ENDPOINT=https://hf-mirror.com to download.)")
            return m
        except Exception:
            pass
    die("no whisper model available. Download one (see help.md: HF_ENDPOINT mirror) and retry.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--model", default="small", help="faster-whisper size: base/small/medium/large-v3")
    ap.add_argument("--language", default=None, help="force source language, e.g. zh/en; default auto")
    ap.add_argument("--mimo-asr", action="store_true", help="proofread each segment's text with MiMo ASR")
    ap.add_argument("--speakers", type=int, default=None, help="force number of speakers")
    ap.add_argument("--threshold", type=float, default=0.75)
    a = ap.parse_args()

    P = job_paths(a.job)
    cfg = load_config(a.job)
    src = cfg["source"]
    info = probe(src)
    if not info["has_audio"]:
        die("source has no audio track")
    P["analysis"].mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", SR, P["audio16k"]])
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "2", "-ar", 48000, P["audio_full"]])
    x = read_wav16(P["audio16k"])

    from faster_whisper import WhisperModel  # local import: slow
    model = load_whisper(WhisperModel, a.model)
    segs, tinfo = model.transcribe(x, language=a.language, beam_size=5, word_timestamps=True,
                                   vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
    segs = list(segs)
    lang = tinfo.language
    print(f"Detected spoken language: {lang} (p={tinfo.language_probability:.2f})")

    out = []
    for i, s in enumerate(segs):
        words = [{"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
                  "p": round(w.probability, 3)} for w in (s.words or [])]
        start = words[0]["start"] if words else s.start
        end = words[-1]["end"] if words else s.end
        out.append({"id": i, "start": round(start, 3), "end": round(end, 3),
                    "text": s.text.strip(), "words": words})

    if a.mimo_asr:
        import mimo_client
        print(f"MiMo ASR proofreading {len(out)} segments...")
        for s in out:
            clip = x[int(max(0, s["start"] - 0.15) * SR): int((s["end"] + 0.15) * SR)]
            try:
                s["text_mimo"] = mimo_client.asr(wav_bytes(clip), lang if lang in ("zh", "en") else "auto")
            except Exception as e:  # keep going; whisper text remains
                s["text_mimo_error"] = str(e)[:200]

    # speech rate
    speak = sum(s["end"] - s["start"] for s in out) or 1
    if lang == "zh":
        units = sum(sum(1 for c in s.get("text_mimo") or s["text"] if "一" <= c <= "鿿") for s in out)
        unit = "chars/s"
    else:
        units = sum(len((s.get("text_mimo") or s["text"]).split()) for s in out)
        unit = "words/s"
    rate = units / speak

    # speakers
    feats, f0s, idx = [], [], []
    for s in out:
        f, f0 = seg_features(x[int(s["start"] * SR): int(s["end"] * SR)])
        if f is not None:
            feats.append(f)
            f0s.append(f0)
            idx.append(s["id"])
    speakers = []
    sample_dir = P["analysis"] / "speaker_samples"
    sample_dir.mkdir(exist_ok=True)
    if feats:
        groups = cluster(feats, f0s, a.speakers, a.threshold)
        groups.sort(key=lambda g: -sum(out[idx[i]]["end"] - out[idx[i]]["start"] for i in g))
        seg_spk = {}
        for k, g in enumerate(groups):
            sid = f"S{k + 1}"
            ids = sorted(idx[i] for i in g)
            for i in ids:
                seg_spk[i] = sid
            f0 = [f0s[i] for i in g if f0s[i]]
            mf0 = float(np.median(f0)) if f0 else None
            gender = None if mf0 is None else ("male" if mf0 < 145 else "female" if mf0 > 175 else "uncertain")
            clip = np.concatenate([x[int(out[i]["start"] * SR): int(out[i]["end"] * SR)] for i in ids[:4]])[: SR * 8]
            sp = sample_dir / f"{sid}.wav"
            sp.write_bytes(wav_bytes(clip))
            speakers.append({"id": sid, "gender_guess": gender, "median_f0_hz": mf0 and round(mf0),
                             "seconds": round(sum(out[i]["end"] - out[i]["start"] for i in ids), 1),
                             "segment_ids": ids, "sample": str(sp),
                             "examples": [f'{out[i]["start"]:.1f}s {out[i]["text"][:40]}' for i in ids[:3]]})
        for s in out:
            s["speaker"] = seg_spk.get(s["id"], "S1")
    save_json(P["transcript"], {
        "source": src, "video": info, "language": lang,
        "language_probability": round(tinfo.language_probability, 3),
        "speech_seconds": round(speak, 2), "speech_rate": round(rate, 2), "speech_rate_unit": unit,
        "speech_first": out[0]["start"] if out else None, "speech_last": out[-1]["end"] if out else None,
        "segments": out})
    save_json(P["speakers"], {"method": "numpy MFCC stats + F0, agglomerative (approximate - confirm with user)",
                              "speakers": speakers})

    print(f"\nSegments: {len(out)}  speech {speak:.1f}s  rate {rate:.2f} {unit}")
    print(f"Speech spans {out[0]['start']:.2f}s - {out[-1]['end']:.2f}s of {info['duration']:.2f}s" if out else "No speech found")
    for sp in speakers:
        print(f"  {sp['id']}: {sp['gender_guess']} f0={sp['median_f0_hz']} {sp['seconds']}s  sample={sp['sample']}")
    print(f"\nWrote {P['transcript']}\n      {P['speakers']}")


if __name__ == "__main__":
    main()
