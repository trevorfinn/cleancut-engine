"""CleanCut web server - wraps the Phase-1 pipeline in an upload API.

Run:  python server.py   (serves on http://localhost:8477)

POST /jobs        multipart: file, mode(silence|beep|cut), tier(1|2|3)
GET  /jobs/{id}   -> {status, flagged, error}
GET  /jobs/{id}/download -> cleaned file
"""
import json
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

import clean as pipeline

HERE = Path(__file__).parent
JOBS_DIR = Path(os.environ.get("CLEANCUT_JOBS", HERE / "jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CleanCut")

# Allow the Vercel-hosted page (a different origin) to call this engine.
# CLEANCUT_ALLOW_ORIGINS is a comma-separated list; "*" during testing.
_origins = os.environ.get("CLEANCUT_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"], allow_headers=["*"])
jobs: dict[str, dict] = {}

_model_lock = threading.Lock()


def fetch_url(url: str, jdir: Path) -> Path:
    """Download a video/audio file from a link (Drive, YouTube, TikTok,
    direct file URLs...) via yt-dlp."""
    m = re.search(r"drive\.google\.com/file/d/([\w-]+)", url)
    if m:
        url = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    r = subprocess.run(
        ["yt-dlp", "--no-playlist", "-f", "mp4/best", "--max-filesize", "500M",
         "-o", str(jdir / "input.%(ext)s"), url],
        capture_output=True, text=True, timeout=600)
    files = [p for p in jdir.iterdir() if p.stem == "input"]
    if r.returncode != 0 or not files:
        raise RuntimeError("couldn't download that link"
                           + (f" ({r.stderr.strip().splitlines()[-1]})"
                              if r.stderr.strip() else ""))
    return files[0]


def process(job_id: str, inp: Optional[Path], mode: str, tier: int,
            url: Optional[str] = None):
    job = jobs[job_id]
    try:
        if url:
            job["status"] = "downloading link"
            inp = fetch_url(url, JOBS_DIR / job_id)
            job["name"] = inp.name
        job["status"] = "extracting audio"
        wav = inp.with_suffix(".wav")
        pipeline.run(["ffmpeg", "-y", "-i", str(inp), "-ac", "1", "-ar",
                      "16000", str(wav)])
        job["status"] = "transcribing"
        with _model_lock:  # serialize heavy work on a small instance
            flagged, spans = pipeline.detect(wav, tier)  # double-check pass
        wav.unlink(missing_ok=True)
        job["status"] = "detecting"
        job["flagged"] = [{"word": f["word"].strip(), "start": round(f["start"], 2)}
                          for f in flagged]
        job["status"] = "rendering"
        out = inp.with_name(f"{mode} clean-{inp.name}")
        video = pipeline.has_video(inp)
        pipeline.render(inp, out, spans, mode, video)
        job["output"] = str(out)
        job["status"] = "done"
    except BaseException as e:  # noqa: BLE001 - pipeline.run may SystemExit
        job["status"] = "error"
        job["error"] = str(e)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.post("/jobs")
async def create_job(file: Optional[UploadFile] = None,
                     url: str = Form(""), mode: str = Form("silence"),
                     tier: int = Form(1)):
    if mode not in ("silence", "beep"):
        return {"error": "bad mode"}
    url = url.strip()
    if not file and not url:
        return {"error": "no file or link"}
    if url and not url.lower().startswith(("http://", "https://")):
        return {"error": "link must start with http(s)://"}
    job_id = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / job_id
    jdir.mkdir()
    inp = None
    name = "link"
    if file:
        name = Path(file.filename or "input.mp4").name
        inp = jdir / name
        inp.write_bytes(await file.read())
        url = ""
    jobs[job_id] = {"status": "queued", "flagged": None, "output": None,
                    "error": None, "name": name}
    threading.Thread(target=process, args=(job_id, inp, mode, tier, url),
                     daemon=True).start()
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "not found"}
    return {"status": job["status"], "flagged": job["flagged"],
            "error": job["error"]}


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.get("output"):
        return {"error": "not ready"}
    out = Path(job["output"])
    return FileResponse(out, filename=out.name,
                        media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8477)))
