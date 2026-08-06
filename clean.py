"""CleanCut Phase-1 pipeline: remove foul language from an audio/video file.

Usage:
  python clean.py input.mp4 --mode beep|silence|cut [--tier 1|2|3] [-o out.mp4]

- Transcribes with faster-whisper (word-level timestamps, open source, local)
- Matches words + phrases against wordlist.json (severity tiers:
  --tier 1 censors everything incl. mild; 2 = strong+severe only; 3 = severe only)
- beep: 1 kHz tone over each flagged word; silence: mutes it; cut: removes the
  segment entirely (audio and video)
- Writes a JSON report next to the output (flagged words + timestamps) —
  this becomes the app's review screen data later.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PAD = 0.06  # seconds of padding around each flagged word

LEET = str.maketrans("013457$@", "oleastsa")

def normalize(token: str) -> str:
    t = token.strip().lower()
    t = re.sub(r"[^\w@$]+", "", t)
    return t

def variants(t: str):
    yield t
    yield t.translate(LEET)
    if t.endswith("s"):
        yield t[:-1]
    if t.endswith("ing"):
        yield t[:-3]
        yield t[:-3] + "in"
    if t.endswith("in"):
        yield t + "g"

def load_wordlist(tier: int):
    data = json.loads((HERE / "wordlist.json").read_text(encoding="utf-8"))
    singles = {w: s for w, s in data.items() if " " not in w and s >= tier}
    phrases = {w: s for w, s in data.items() if " " in w and s >= tier}
    return singles, phrases

def transcribe(audio_path: Path):
    """Word-level transcription. Uses the Groq speech API when GROQ_API_KEY is
    set (lightweight - fits a small free host); otherwise runs faster-whisper
    locally."""
    if os.environ.get("GROQ_API_KEY"):
        return transcribe_groq(audio_path, os.environ["GROQ_API_KEY"])
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"word": w.word, "start": w.start, "end": w.end})
    return words


def transcribe_groq(audio_path: Path, key: str):
    """Call Groq's Whisper endpoint for word-level timestamps. Uses only the
    stdlib so the deployed container stays tiny."""
    import urllib.request
    import urllib.error

    # Whisper tends to "clean up" profanity out of its transcript; this prompt
    # biases it to transcribe swear words verbatim so they can be censored.
    prime = ("Transcribe explicit content verbatim, including profanity and "
             "swear words like fuck, fucking, shit, ass, damn, bitch. Do not "
             "censor or omit any words.")
    boundary = "----cleancutboundary"
    parts = []
    for name, value in (("model", "whisper-large-v3"),
                        ("response_format", "verbose_json"),
                        ("prompt", prime),
                        ("timestamp_granularities[]", "word")):
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    parts.append(audio_path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "User-Agent": "curl/8.0",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Groq transcription failed: HTTP {e.code} "
                           f"{e.read().decode()[:200]}")
    return [{"word": w["word"], "start": w["start"], "end": w["end"]}
            for w in (resp.get("words") or [])]

def flag_words(words, singles, phrases):
    flagged = []
    norm_words = [normalize(w["word"]) for w in words]
    for i, nw in enumerate(norm_words):
        if not nw:
            continue
        if any(v in singles for v in variants(nw)):
            flagged.append({**words[i], "match": nw})
            continue
        # phrase check: up to 3-word windows
        for span in (2, 3):
            if i + span <= len(norm_words):
                joined = " ".join(norm_words[i:i + span])
                if joined in phrases:
                    for j in range(i, i + span):
                        flagged.append({**words[j], "match": joined})
                    break
    # dedupe + merge overlapping ranges with padding
    spans = []
    for f in sorted(flagged, key=lambda x: x["start"]):
        s, e = max(0.0, f["start"] - PAD), f["end"] + PAD
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return flagged, spans


def merge_spans(spans):
    """Union overlapping [start, end] ranges (already padded)."""
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def detect(audio_path: Path, tier: int = 1):
    """Transcribe and flag foul language. On the Groq path this runs a
    double-check pass (two transcriptions) and takes the union of what each
    catches - AI transcription randomly drops a swear on any single pass, so a
    second pass closes most of those gaps. Returns (flagged, spans)."""
    singles, phrases = load_wordlist(tier)
    passes = 2 if os.environ.get("GROQ_API_KEY") else 1
    all_flagged, all_spans = [], []
    for _ in range(passes):
        words = transcribe(audio_path)
        fl, sp = flag_words(words, singles, phrases)
        all_flagged.extend(fl)
        all_spans.extend(sp)
    # dedupe the flagged list (for the review screen) by time + word
    seen, flagged = set(), []
    for f in sorted(all_flagged, key=lambda x: x["start"]):
        k = (round(f["start"], 1), f["word"].strip().lower())
        if k not in seen:
            seen.add(k)
            flagged.append(f)
    return flagged, merge_spans(all_spans)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{r.stderr[-2000:]}")

def has_video(path: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    return "video" in r.stdout

def enable_expr(spans):
    return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in spans)

def render(inp: Path, out: Path, spans, mode: str, video: bool):
    if not spans:
        print("No foul language found - copying input to output.")
        run(["ffmpeg", "-y", "-i", str(inp), "-c", "copy", str(out)])
        return
    if mode in ("beep", "silence"):
        expr = enable_expr(spans)
        mute = f"[0:a]volume=0:enable='{expr}'[muted]"
        if mode == "silence":
            af = f"{mute};[muted]anull[aout]"
        else:
            beeps = f"sine=frequency=1000:sample_rate=44100:duration={spans[-1][1]+1:.3f}"
            af = (f"{mute};{beeps},volume=0.4,volume=0:enable='not({expr})'[beep];"
                  f"[muted][beep]amix=inputs=2:duration=first:normalize=0[aout]")
        cmd = ["ffmpeg", "-y", "-i", str(inp), "-filter_complex", af,
               "-map", "[aout]"]
        if video:
            cmd += ["-map", "0:v", "-c:v", "copy"]
        cmd += ["-c:a", "aac", str(out)]
        run(cmd)
    elif mode == "cut":
        # keep everything outside the flagged spans
        keep = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in
                        _keep_ranges(spans, _duration(inp)))
        fc = f"[0:a]aselect='{keep}',asetpts=N/SR/TB[aout]"
        cmd = ["ffmpeg", "-y", "-i", str(inp)]
        if video:
            fc += f";[0:v]select='{keep}',setpts=N/FRAME_RATE/TB[vout]"
            cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "libx264", "-preset", "fast"]
        else:
            cmd += ["-filter_complex", fc, "-map", "[aout]"]
        cmd += ["-c:a", "aac", str(out)]
        run(cmd)

def _duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())

def _keep_ranges(spans, total):
    keep, cur = [], 0.0
    for s, e in spans:
        if s > cur:
            keep.append((cur, s))
        cur = max(cur, e)
    if cur < total:
        keep.append((cur, total))
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--mode", choices=["beep", "silence", "cut"], default="beep")
    ap.add_argument("--tier", type=int, default=1,
                    help="1=censor all incl. mild, 2=strong+severe, 3=severe only")
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()

    out = args.output or args.input.with_name(
        f"{args.input.stem}.clean-{args.mode}{args.input.suffix}")
    video = has_video(args.input)

    wav = args.input.with_suffix(".cleancut-tmp.wav")
    run(["ffmpeg", "-y", "-i", str(args.input), "-ac", "1", "-ar", "16000",
         str(wav)])
    print("Transcribing...")
    flagged, spans = detect(wav, args.tier)
    wav.unlink(missing_ok=True)

    print(f"Flagged: {len(flagged)} | censor spans: {len(spans)}")
    for f in flagged:
        print(f"  {f['start']:6.2f}s  {f['word'].strip()!r}  (match: {f.get('match','')})")

    render(args.input, out, spans, args.mode, video)
    report = {"input": str(args.input), "mode": args.mode, "tier": args.tier,
              "flagged": flagged, "spans": spans, "output": str(out)}
    rp = out.with_suffix(".report.json")
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Done -> {out}\nReport -> {rp}")

if __name__ == "__main__":
    main()
