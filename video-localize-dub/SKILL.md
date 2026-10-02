---
name: video-localize-dub
description: Localize a short video into another language (default English) - remove the burned-in original captions (inpaint / blur / colored plate, per block), re-dub with Xiaomi MiMo TTS (voice per detected speaker, timed to the original speech, no overlaps), and render new captions with Remotion that match the original captions' position, angle, size and animation. Use when the user wants a Chinese (or other) video turned into an English-dubbed, English-captioned version. Asks the user for every look/voice decision before producing anything.
---

# Video localize + dub

Scripts live in `scripts/` next to this file (run them with `py -3.12`; on non-Windows use `python3`).
Everything for one video lives in a **job folder**; every user decision lives in `<job>/job.json`.
Human-oriented FAQ: `help.md`. **You (the AI) do all deployment and configuration yourself**, including storing the MiMo key in the credential
manager. The user only hands you the key once and answers the Phase 2 questions.

```
setup -> init_job -> ocr_scan + analyze_audio -> ASK THE USER (mandatory) -> script.json + check_script
-> tts -> schedule_audio -> remove_captions (preview, then full) -> build_captions -> render -> verify + LOOK -> report
```

## Hard rules

1. **Ask before producing.** Phase 2 questions are mandatory, even if the user said "just do it" - at minimum confirm the
   proposed defaults in one AskUserQuestion round. Ask in the user's language.
2. **Never print, log, pass on the command line, or write the API key anywhere.** Scripts read it themselves
   (env `XIAOMI_MIMO_API_KEY` / `MIMO_API_KEY`, else Windows Credential Manager `Codex/XiaomiMiMoAPI`).
3. **Look at every image the scripts tell you to look at** (`ocr_blocks.png`, removal previews, `qc/*.png`) with the
   Read tool before moving on. Report what you actually saw, including defects. Never claim "perfect".
4. Install whatever is missing yourself (`setup.py`) - no permission round needed. Tell the user in one line what is
   being installed and roughly how big (CUDA torch ~2.5 GB, Remotion ~300 MB, LaMa ~200 MB, demucs ~80 MB).
5. Only process material the user has the right to use. This skill is for localization; it does not do (and must not
   be described as doing) duplicate-detection evasion / "make it count as original" tricks.
6. Re-running `ocr_scan.py` renumbers blocks (B1, B2 ...). After a rescan, re-check every block id in job.json.

## Phase 0 - deploy (you run it, every new machine / first use)

```
py -3.12 scripts/setup.py          # installs everything missing, ends with env_check
```
- Long-running (first time on a fresh PC: 5-30 min, mostly downloads) -> run it in the background and tell the user.
  It installs: ffmpeg + Node 18+ (winget / brew / apt), required pip packages, LaMa (+ Pillow re-fix), demucs,
  CUDA torch + cuBLAS/cuDNN when an NVIDIA GPU is unused, the Remotion runtime (unless a usable one exists),
  whisper `small` / LaMa / htdemucs models (switches to `HF_ENDPOINT=https://hf-mirror.com` when huggingface.co is
  blocked). `--minimal` skips optional parts, `--no-gpu` skips the 2.5 GB CUDA torch, `--dry-run` only lists.
- Success = last line `RESULT: ready`. Any `[!!]` line in the summary: fix it (the line says how) and rerun setup.
  A system tool that cannot be installed automatically (no winget/brew, no sudo) -> give the user the one link.
- **MiMo key -> credential manager (you configure it)**:
  1. `py -3.12 scripts/mimo_key.py --check`. Configured -> done (setup.py already copied an env-var key into the
     store). Optional: `--test` makes one tiny TTS call to prove the key works.
  2. Not configured -> ask the user for it in one message: "create a key at https://platform.xiaomimimo.com/ ->
     API Keys -> Create, and send it to me". Pricing: TTS/ASR were free at the time of writing - "以官网为准".
  3. Store it by piping it on stdin (never as an argument, env var you set, or file), with the Bash tool:
     ```
     py -3.12 <skill>/scripts/mimo_key.py --store <<'KEY'
     <the key>
     KEY
     ```
     It goes to Windows Credential Manager (generic credential `Codex/XiaomiMiMoAPI`), or the macOS Keychain /
     Linux Secret Service via `keyring`. Then run `--test`. FAILED -> ask for a new key, store again (overwrites).
  4. Never repeat the key in your replies, never write it into job.json, scripts, notes or memory, never pass it
     in a command-line argument. After storing, only refer to it as "the stored key".
  - Scripts read it themselves: env `XIAOMI_MIMO_API_KEY` / `MIMO_API_KEY` first, then the credential store.
  - Replace: store again. Remove: `mimo_key.py --delete`.
- Hardware: env_check ends with a "Hardware" section (same as `py -3.12 scripts/hw.py`) and `hint:` lines.
  Everything adapts automatically per machine (`scripts/hw.py`); nothing is hard-coded to one PC:
  | step | GPU used when available | fallback |
  |---|---|---|
  | LaMa, demucs (torch) | CUDA (NVIDIA) > MPS (Apple) > DirectML (AMD/Intel, Windows, experimental) | CPU, all cores |
  | whisper | CUDA float16 (needs CUDA 12 + cuDNN 9 libs) | CPU int8 |
  | OCR | onnxruntime-gpu CUDA | CPU |
  | Remotion | parallel tabs from cores + free RAM | 1 tab |
  | encode | `hardware.encoder` libx264 default; `auto` = nvenc / amf / qsv / videotoolbox, tested first | libx264 |
  Every GPU path falls back to CPU on error (printed as WARNING), so a job never fails because of a driver.
  Override in job.json `hardware` (device, encoder, threads, remotion_concurrency, remotion_gl).
  A `hint:` line means a GPU is present but unused: setup.py already handles NVIDIA; DirectML (AMD/Intel) is
  experimental - install it only if the user wants to try (falls back to CPU anyway).

## Phase 1 - analyze

```
py -3.12 scripts/init_job.py --source <video> --job <job_dir>
py -3.12 scripts/ocr_scan.py --job <job>
py -3.12 scripts/analyze_audio.py --job <job> [--model small] [--mimo-asr] [--speakers N]
py -3.12 scripts/separate_bgm.py --job <job>          # only if background = separate (installs demucs if missing)
```
- `ocr_scan` finds every text block anywhere in frame (any orientation) -> `analysis/ocr.json`,
  `analysis/ocr_blocks.png`. OPEN the png. Note for each block: what it is (subtitle / title / sticker / logo /
  karaoke line), where, angle, colors, animation (`entrance_guess`), how many words show at once.
- `analyze_audio`: transcript with word times, speech rate, speaker clusters + `analysis/speaker_samples/S*.wav`.
  Speaker clustering is approximate: different genders separate well, two voices of the same gender may merge,
  one voice may split if its tone changes. Tell the user that and let them correct it (`--speakers N` rerun).
- If whisper cannot download models (China): `set HF_ENDPOINT=https://hf-mirror.com`; the script also falls back to
  any cached size automatically.
- The source must contain speech for dubbing; with no audio track set `dub.enabled=false` (captions/removal only).

## Phase 2 - ask the user (mandatory)

Show `analysis/ocr_blocks.png` (give its path) and summarize the blocks and speakers. Then ask with AskUserQuestion
(several questions per call, max 4 per call; recommended option first). Cover all of these:

**Dub**
- Target language (default English). MiMo preset voices exist only for English (Mia, Chloe, Milo, Dean) and Chinese
  (冰糖, 茉莉, 苏打, 白桦). Other target languages: an English voice may speak them with an accent - say so; let the
  user decide.
- Voice per speaker: list each S* with gender guess, seconds, sample path; propose a matching voice
  (male -> Milo/Dean, female -> Mia/Chloe). Offer an audition of the candidates with a translated line:
  `py -3.12 scripts/mimo_client.py --sample "<line>" --voices Mia,Milo --out <job>/auditions` and give the paths.
- Delivery style (`dub.style_instruction`): e.g. energetic short-video / calm explainer / match the original.
- Background: `separate` (keep original music, remove the original voice - needs demucs) | `none` | `file`.

**Captions**
- Which blocks to replace with translated captions (`captions.replace_blocks`), usually the subtitle line(s).
  Placement: `match_block` (same center + angle as `anchor_block`) | `bottom` | `custom`.
- Font: offer the original-like look (Impact / Arial Black / Montserrat ExtraBold-like via `font_file`) - ask.
- Size: `match` (same height as the original) or fixed px. Colors of line1/line2, stroke, karaoke highlight
  (`captions.highlight` - copy the original's plate color from OCR / the overview image if it had one), animation
  (`pop`/`fade`/`slide`/`none` - default to the original's `entrance_guess`), words per card (`auto` = like original).

**Removal - per block** (every block, including ones not replaced, e.g. titles, stickers):
- `inpaint` (cleanest; engine `lama` if installed - strongly better than `opencv`, which smears on texture),
  `blur` (+strength), `color` (plate: ask the color, opacity, corner radius), or `keep`.
- Honest expectation: removal reconstructs pixels; on big text over detailed/moving backgrounds traces can remain.

**Output**: output folder + base name, watermark yes/no (text, angle).

Write every answer into `<job>/job.json` (see `_doc` in it for every key). Keep the answers in sync if the user
changes their mind later.

## Phase 3 - script

Write `<job>/analysis/script.json` yourself (format in `scripts/check_script.py` docstring): one entry per source
segment group, `src_ids`, `speaker`, natural spoken `text` in the target language, optional shorter `caption`.
- Translate meaning for a native viewer of short videos: short, punchy, keep brand/model names and numbers.
- Respect the word budget: check_script reports per line `words / budget`; budget = window x rate
  (target-language default rate when languages differ, e.g. English ~2.7 words/s). Stay at or under it.
- Merge very short source segments of the same speaker into one line when that reads better (`src_ids: [3,4]`).
```
py -3.12 scripts/check_script.py --job <job>      # fix every ERROR, rerun until clean
```

## Phase 4 - voice

```
py -3.12 scripts/tts.py --job <job>
py -3.12 scripts/schedule_audio.py --job <job>
```
- schedule places each line at its original time, never overlapping (min gap 0.08 s), tempo kept inside
  0.95-1.10 so the pace sounds even, may start a line up to 0.3 s early.
- On FAIL it lists the lines that do not fit and by how much, plus TTS speaking rate vs original: shorten those lines
  in script.json (or ask for a brisker style), rerun tts.py (only changed lines are re-requested) and schedule.
- WARN about uneven tempo -> rewrite the fastest lines shorter.
- Voice is normalized (`voice_level_db`, default -18 dBFS) and the music sits 14 dB under it, ducked while speaking.

## Phase 5 - remove original captions

```
py -3.12 scripts/remove_captions.py --job <job> --preview [--at 0.1,3.5,9]
py -3.12 scripts/remove_captions.py --job <job>          # full -> video/clean.mp4
```
- Preview writes before|after PNGs in `<job>/preview/`. OPEN them. Include a frame in the first second (titles),
  frames during each block, and a karaoke/plate frame. Fix before the full run:
  - colored streaks / flat patches INSIDE the filled area = a hairline of glow/shadow was left on the hole border
    and the inpainter continued it -> raise `grow_px` for that block (default 0.3 x char height; try 12-16)
  - traces of a faint plate/glow next to the text -> lower `diff_threshold` (e.g. 20) or raise `padding_px`
  - text missed during an animation -> `"mask": "box"` for that block
  - smears with opencv -> engine `lama` (setup.py installs it)
- Time estimate (tell the user BEFORE the full run; the script prints it on start as `Estimate: ...`):
  ```
  minutes ~= (frames_with_text x s_per_frame + total_frames x 0.01) / 60
  s_per_frame: lama ~2.9 (CPU, 544x960; scales roughly with block area; CUDA GPU ~0.15-0.3) | opencv ~0.1 |
               blur/color ~0.02
  frames_with_text = union of all removed blocks' event frames (overlapping blocks count once)
  ```
  Example: 20 s x 30 fps, text on all 600 frames, lama -> (600 x 2.9 + 6) / 60 ~= 29 min (measured).
  60 s video -> ~1.5 h with lama. If that is too long, offer opencv for simple backgrounds or blur/color.
  The script measures the real speed on this machine (`hw.py` benchmark cache) and prints the device it uses.
- Run the full LaMa pass in the background; it is memory-heavy (PyTorch on CPU). The full run is resumable: it
  writes 150-frame chunks to `video/clean_parts/`; rerunning the same command continues after the last finished
  chunk (`--restart` discards them; changed removal settings discard them automatically). If the host kills it for
  low memory, it is not a script error: tell the user, ask them to free memory, and restart only when they say so.

## Phase 6 - captions + render

```
py -3.12 scripts/build_captions.py --job <job>
py -3.12 scripts/render.py --job <job> --frames 0-89      # quick review render -> preview/
py -3.12 scripts/render.py --job <job>                    # full: captioned + dub mux (+ watermark) -> output.dir
```
- Card timing comes from the dub (word timestamps), lead 2 frames, gaps < 10 frames bridged, no overlaps.
- Video stream of clean.mp4 is rendered once by Remotion (CRF 17) and then muxed with `-c:v copy`.
- Frame count, resolution and fps equal the source; audio is padded to the exact video length.

## Phase 7 - verify and report

```
py -3.12 scripts/verify.py --job <job>
```
- Hard checks: size/fps/frames, audio length, dub overlaps/tempo, caption overlaps, residual OCR text in removed blocks.
- OPEN every `qc/removal_*.png` and `qc/captions_*.png`. Check: no leftover strokes/plates, captions inside frame,
  readable, same place/angle as the original, highlight on the right word.
- Report to the user: output paths, what was done, every remaining defect you saw (with frame/time), and the
  choices they can change (job.json key + which phase to rerun). Rerun only the affected phases:
  script/voice -> Phase 4 + build_captions + render; removal -> Phase 5 + render; caption look -> build_captions + render.

## Limitations (tell the user when relevant)
- Speaker detection is acoustic clustering, not diarization AI: reliable for male/female, weak for same-gender voices.
- MiMo presets: English + Chinese voices only; no voice cloning. Other target languages are best-effort.
- No lip sync. Line timing follows the original speech, not mouth shapes.
- Inpainting is reconstruction, never the true hidden pixels; large text over complex moving background may leave
  soft areas. Blur / color plate are the fallback.
- OCR is RapidOCR (offline): very stylized/handwritten fonts or tiny text may be missed - check `ocr_blocks.png`
  against the video and tell the user about anything missed (try `ocr_scan.py --step 0.15 --min-score 0.5`).
