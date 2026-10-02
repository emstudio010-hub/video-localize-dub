"""Xiaomi MiMo TTS / ASR client (OpenAI-compatible /v1/chat/completions).

TTS: model mimo-v2.5-tts. Text to speak goes in role=assistant; an optional
style instruction (never spoken) goes in role=user. Returns 24 kHz mono WAV bytes.
ASR: model mimo-v2.5-asr. Audio as data URL in an input_audio content part.
Returns plain text only (no timestamps).
"""

import base64
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mimo_key import load_api_key

BASE_URL = "https://api.xiaomimimo.com/v1"
TTS_MODEL = "mimo-v2.5-tts"
ASR_MODEL = "mimo-v2.5-asr"
VOICES = {
    "Mia": ("en", "female", "lively"),
    "Chloe": ("en", "female", "sweet, soft"),
    "Milo": ("en", "male", "sunny, young"),
    "Dean": ("en", "male", "steady, gentle"),
    "冰糖": ("zh", "female", "lively"),
    "茉莉": ("zh", "female", "elegant"),
    "苏打": ("zh", "male", "sunny, young"),
    "白桦": ("zh", "male", "mature"),
}


class MimoError(RuntimeError):
    pass


def _key():
    k = load_api_key()
    if not k:
        raise MimoError("MiMo API key not configured. Run: py -3.12 mimo_key.py --store")
    return k


def _post(payload, key, timeout=180, retries=2):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries + 1):
        req = Request(BASE_URL + "/chat/completions", data=body, method="POST", headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except HTTPError as e:
            detail = e.read().decode("utf-8", "replace").replace(key, "[REDACTED]")[:800]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise MimoError(f"MiMo HTTP {e.code}: {detail}")
        except (URLError, OSError) as e:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise MimoError(f"MiMo network error: {e}")


def tts(text, voice, style=None, timeout=180):
    if voice not in VOICES:
        raise MimoError(f"Unknown voice {voice}. Choose one of {list(VOICES)}")
    msgs = []
    if style:
        msgs.append({"role": "user", "content": style})
    msgs.append({"role": "assistant", "content": text.strip()})
    data = _post({"model": TTS_MODEL, "messages": msgs,
                  "audio": {"format": "wav", "voice": voice}, "stream": False}, _key(), timeout)
    try:
        return base64.b64decode(data["choices"][0]["message"]["audio"]["data"])
    except (KeyError, IndexError, TypeError):
        raise MimoError("Unexpected TTS response: " + json.dumps(data)[:500])


def asr(wav_bytes, language="auto", timeout=120):
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    if len(b64) > 10 * 1024 * 1024:
        raise MimoError("Audio chunk exceeds 10 MB base64 limit; split it.")
    payload = {
        "model": ASR_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64," + b64}}]}],
        "asr_options": {"language": language},
        "stream": False,
    }
    data = _post(payload, _key(), timeout)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise MimoError("Unexpected ASR response: " + json.dumps(data)[:500])
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return clean_asr_text(content or "")


def clean_asr_text(t):
    """Model sometimes prefixes reasoning / language tags: '<think>..</think>', 'think>', '<chinese>'."""
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"^\s*(think>|</?think>)", "", t)
    t = re.sub(r"<\s*/?\s*[a-zA-Z_]{2,20}\s*>", "", t)
    return t.strip()


if __name__ == "__main__":
    # voice audition: py -3.12 mimo_client.py --sample "Line to hear" --voices Mia,Milo --out <dir> [--style "..."]
    import argparse
    import common  # noqa: F401  (utf-8 console output)
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="text to speak")
    ap.add_argument("--voices", default="Mia,Chloe,Milo,Dean")
    ap.add_argument("--style", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in a.voices.split(","):
        p = out / f"voice_{v.strip()}.wav"
        p.write_bytes(tts(a.sample, v.strip(), a.style))
        print(f"{v.strip()}: {p}")
