import os
import uuid
import shutil
import tempfile
import subprocess
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

app = FastAPI()

# ----------------------------
# Config
# ----------------------------
DOWNLOAD_TIMEOUT = 120          # seconds per file download
FFMPEG_TIMEOUT = 900            # seconds (15 minutes)
MAX_BYTES = 300 * 1024 * 1024   # 300MB safety limit


# ----------------------------
# Request model
# ----------------------------
class MuxRequest(BaseModel):
    video_url: HttpUrl
    audio_url: HttpUrl
    subtitles_url: Optional[HttpUrl] = None


# ----------------------------
# Helpers
# ----------------------------
def download_file(url: str, dest_path: str) -> None:
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


def get_audio_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=30
        ).decode().strip()
        duration = float(output)
        if duration <= 0:
            raise ValueError("Invalid duration")
        return duration
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


# ----------------------------
# Endpoint
# ----------------------------
@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    video_path = os.path.join(tmpdir, "video.mp4")
    audio_path = os.path.join(tmpdir, "audio.mp3")
    output_path = os.path.join(tmpdir, "output.mp4")
    subs_path = os.path.join(tmpdir, "subs.srt")

    # ----------------------------
    # Download inputs
    # ----------------------------
    download_file(str(request.video_url), video_path)
    download_file(str(request.audio_url), audio_path)

    has_subtitles = request.subtitles_url is not None
    if has_subtitles:
        download_file(str(request.subtitles_url), subs_path)

    # ----------------------------
    # Probe audio duration
    # ----------------------------
    audio_duration = get_audio_duration_seconds(audio_path)

    # ----------------------------
    # Build FFmpeg command
    # ----------------------------
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", video_path,
        "-i", audio_path,
    ]

    # Optional subtitle burn-in
    if has_subtitles:
        ffmpeg_cmd += [
            "-vf", f"subtitles={subs_path}"
        ]

    ffmpeg_cmd += [
        "-filter_complex",
        (
            "[0:a]volume=0.25[a0];"
            "[a0][1:a]amix=inputs=2:weights=1 3:dropout_transition=0[a]"
        ),
        "-map", "0:v:0",
        "-map", "[a]",
        "-t", f"{audio_duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ]

    # ----------------------------
    # Run FFmpeg
    # ----------------------------
    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT
        )
    except subprocess.CalledProcessError as e:
        error_tail = e.stderr.decode(errors="ignore")[-4000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {error_tail}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out")

    # ----------------------------
    # Cleanup AFTER response
    # ----------------------------
    background_tasks.add_task(shutil.rmtree, tmpdir, True)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"final_{job_id}.mp4"
    )
