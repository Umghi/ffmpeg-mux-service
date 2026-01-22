import os
import uuid
import shutil
import tempfile
import subprocess
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

app = FastAPI()

# --- Config ---
DOWNLOAD_TIMEOUT = 120  # seconds per file
FFMPEG_TIMEOUT = 900    # seconds (15 min) - adjust if needed
MAX_BYTES = 300 * 1024 * 1024  # 300MB safety cap

class MuxRequest(BaseModel):
    video_url: HttpUrl
    audio_url: HttpUrl

def _download(url: str, dest_path: str) -> None:
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True) as r:
            r.raise_for_status()
            total = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(status_code=413, detail="File too large")
                    f.write(chunk)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

def _ffprobe_duration_seconds(path: str) -> float:
    # Returns duration in seconds (float). Uses ffprobe.
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
        dur = float(out)
        if dur <= 0:
            raise ValueError("Non-positive duration")
        return dur
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")

@app.post("/mux")
def mux(req: MuxRequest):
    """
    Returns the final MP4 as binary.
    Make.com should call this with HTTP 'Get a file' behavior (or equivalent),
    then upload the returned file to Dropbox.
    """
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    video_in = os.path.join(tmpdir, "input_video.mp4")
    audio_in = os.path.join(tmpdir, "input_audio.mp3")
    out_mp4  = os.path.join(tmpdir, "output.mp4")

    try:
        _download(str(req.video_url), video_in)
        _download(str(req.audio_url), audio_in)

        audio_dur = _ffprobe_duration_seconds(audio_in)

        # IMPORTANT:
        # - We do NOT use -shortest (per your lock).
        # - Because -stream_loop -1 makes video infinite, we MUST cap output.
        # - We cap via "-t <audio_duration>" so output duration == audio duration.
        #
        # Video is re-encoded (libx264), audio -> AAC.
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", video_in,
            "-i", audio_in,
            "-t", f"{audio_dur:.3f}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_mp4
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="ignore")[-4000:]
            raise HTTPException(status_code=500, detail=f"ffmpeg failed: {err}")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="ffmpeg timed out")

        # Return file as binary
        return FileResponse(
            out_mp4,
            media_type="video/mp4",
            filename=f"final_{job_id}.mp4"
        )

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
