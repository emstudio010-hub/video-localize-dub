"""Synthesize every script segment with MiMo TTS (voice per speaker), trim edge silence, cache.

  py -3.12 tts.py --job <job>            # only missing / changed segments are requested
  py -3.12 tts.py --job <job> --only 3,7 # force re-synthesis of some segments

Voice per speaker: job.json dub.speaker_voices {"S1": "Mia", ...}
Style: dub.style_instruction (+ optional dub.speaker_styles {"S2": "excited, faster"})
Writes tts/seg_<id>.wav (24 kHz mono) and tts/tts.json.
"""

import argparse
import hashlib
import io
import wave

import numpy as np

import mimo_client
from common import die, job_paths, load_config, load_json, save_json


def read_wav(b):
    with wave.open(io.BytesIO(b), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        die(f"unexpected TTS sample width {sw}")
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def write_wav(path, x, sr):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def trim(x, sr, db=-45, keep_ms=40):
    """Cut leading/trailing silence so scheduling uses the real speech length."""
    hop = int(sr * 0.01)
    if len(x) < hop * 3:
        return x
    rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same"))
    on = np.nonzero(20 * np.log10(rms + 1e-9) > db)[0]
    if not len(on):
        return x
    k = int(sr * keep_ms / 1000)
    return x[max(0, on[0] - k): min(len(x), on[-1] + k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    P = job_paths(a.job)
    cfg = load_config(a.job)
    script = load_json(P["script"])
    lang = script.get("language", cfg["target_language"])
    dub = cfg["dub"]
    voices = dub["speaker_voices"]
    only = {int(i) for i in a.only.split(",")} if a.only else set()
    cache = P["tts_dir"] / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    for v in set(voices.values()):
        if v not in mimo_client.VOICES:
            die(f"voice '{v}' unknown. Available: {list(mimo_client.VOICES)}")
        vlang = mimo_client.VOICES[v][0]
        if vlang != lang:
            print(f"WARNING: voice {v} is a '{vlang}' preset but the script is '{lang}'. "
                  "MiMo only ships zh/en presets; pronunciation of other languages is not guaranteed.")

    index = []
    for seg in script["segments"]:
        voice = voices[seg.get("speaker", "S1")]
        style = " ".join(x for x in [dub.get("style_instruction"),
                                     dub.get("speaker_styles", {}).get(seg.get("speaker", "S1"))] if x)
        key = hashlib.sha1(f"{voice}|{style}|{seg['text']}".encode("utf-8")).hexdigest()[:16]
        raw = cache / f"{key}.wav"
        if seg["id"] in only and raw.exists():
            raw.unlink()
        if not raw.exists():
            print(f"  TTS seg {seg['id']} [{voice}] {seg['text'][:60]}", flush=True)
            raw.write_bytes(mimo_client.tts(seg["text"], voice, style or None))
        x, sr = read_wav(raw.read_bytes())
        x = trim(x, sr)
        out = P["tts_dir"] / f"seg_{seg['id']:03d}.wav"
        write_wav(out, x, sr)
        index.append({"id": seg["id"], "file": str(out), "voice": voice, "sr": sr,
                      "duration": round(len(x) / sr, 3), "text": seg["text"]})
    save_json(P["tts_index"], {"segments": index})
    print(f"TTS ready: {len(index)} segments -> {P['tts_index']}")


if __name__ == "__main__":
    main()
