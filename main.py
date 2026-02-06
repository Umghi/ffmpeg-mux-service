import os
import uuid
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel, HttpUrl

app = FastAPI()

# ----------------------------
# Config
# ----------------------------
DOWNLOAD_TIMEOUT = 120          # seconds per file download
FFMPEG_TIMEOUT = 900            # seconds
MAX_BYTES = 300 * 1024 * 1024   # 300MB safety limit


# ----------------------------
# Request models
# ----------------------------
class MuxRequest(BaseModel):
    video_url: HttpUrl
    audio_url: HttpUrl
    subtitles_url: Optional[HttpUrl] = None


# ----------------------------
# Helpers
# ----------------------------
def _looks_like_html(first_bytes: bytes) -> bool:
    """
    Crude but effective detection for HTML pages (e.g. Dropbox share / error pages)
    when we expect binary media.
    """
    s = first_bytes.lstrip().lower()
    return (
        s.startswith(b"<!doctype html")
        or s.startswith(b"<html")
        or b"<head" in s[:2000]
    )


def download_file(url: str, dest_path: str) -> None:
    """
    Stream-download a file to dest_path with size and HTML checks.
    """
    try:
        with requests.get(
            url,
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
            allow_redirects=True,
        ) as r:
            r.raise_for_status()

            total = 0
            first_chunk = b""
            wrote_any = False

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue

                    if not wrote_any:
                        first_chunk = chunk[:4096]
                        wrote_any = True
                        if _looks_like_html(first_chunk):
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    "Downloaded HTML instead of a media file. "
                                    "(Check Dropbox link is direct: dl=1 or use temporary link.)"
                                ),
                            )

                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="File too large",
                        )

                    f.write(chunk)

            if not wrote_any:
                raise HTTPException(
                    status_code=400,
                    detail="Downloaded file is empty",
                )

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")


def get_audio_duration_seconds(path: str) -> float:
    """
    Use ffprobe to get the duration of an audio file in seconds.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).decode().strip()

        if output in ("", "N/A"):
            raise ValueError(f"ffprobe duration unavailable: '{output}'")

        duration = float(output)
        if duration <= 0:
            raise ValueError("Invalid duration")

        return duration

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


def srt_timestamp(seconds: float) -> str:
    """
    Convert seconds (float) to SRT timestamp: HH:MM:SS,mmm
    """
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    hh = ms // 3600000
    ms -= hh * 3600000
    mm = ms // 60000
    ms -= mm * 60000
    ss = ms // 1000
    ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def words_to_captions(
    words: List[Dict[str, Any]],
    max_chars: int = 38,
    max_words: int = 7,
    max_duration: float = 2.2,
    min_duration: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Turns word-level timestamps into short caption chunks.

    - max_chars: approximate max characters per caption
    - max_words: max words per caption
    - max_duration: max caption duration in seconds
    - min_duration: minimum duration to avoid instant-flash captions
    """
    # Filter only actual words (drop spacing tokens)
    w = [
        x
        for x in words
        if x.get("type") == "word" and str(x.get("text", "")).strip()
    ]

    caps: List[Dict[str, Any]] = []
    buf: List[str] = []
    start: Optional[float] = None
    last_end: Optional[float] = None

    def flush() -> None:
        nonlocal buf, start, last_end
        if not buf or start is None or last_end is None:
            buf = []
            start = None
            last_end = None
            return

        text = " ".join(buf).strip()
        caps.append(
            {
                "start": float(start),
                "end": float(last_end),
                "text": text,
            }
        )
        buf = []
        start = None
        last_end = None

    for item in w:
        t = str(item.get("text", "")).strip()
        s = float(item.get("start", 0.0))
        e = float(item.get("end", s))

        if start is None:
            start = s

        proposed = (" ".join(buf + [t])).strip()
        dur = e - start

        # chunking rules
        if (
            len(buf) >= max_words
            or len(proposed) > max_chars
            or dur > max_duration
        ):
            flush()
            start = s
            buf.append(t)
            last_end = e
            continue

        buf.append(t)
        last_end = e

        # long pause between words => new caption
        if last_end is not None and (s - last_end) > 0.6:
            flush()

    flush()

    # ensure minimum duration
    for c in caps:
        if (c["end"] - c["start"]) < min_duration:
            c["end"] = c["start"] + min_duration

    return caps


def captions_to_srt(captions: List[Dict[str, Any]]) -> str:
    """
    Turn caption chunks into SRT text.
    """
    lines: List[str] = []
    for idx, c in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(
            f"{srt_timestamp(c['start'])} --> {srt_timestamp(c['end'])}"
        )
        lines.append(c["text"])
        lines.append("")  # blank line
    return "\n".join(lines).strip() + "\n"


def _extract_words_from_payload(payload: Any) -> List[Dict[str, Any]]:
    """
    Support multiple shapes of STT JSON:

    1) {"data": {"words": [...] , ...}}
    2) {"words": [...], ...}
    3) [ {...}, ... ] with either of the above in the first element
    """
    # Direct dict
    if isinstance(payload, dict):
        # Preferred: wrapped under "data"
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("words"), list):
            return data["words"]

        # Flattened: words at root
        if isinstance(payload.get("words"), list):
            return payload["words"]

    # List of bundles (defensive, e.g. some webhook styles)
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            data = first.get("data")
            if isinstance(data, dict) and isinstance(data.get("words"), list):
                return data["words"]
            if isinstance(first.get("words"), list):
                return first["words"]

    return []


# ----------------------------
# Endpoints
# ----------------------------
@app.post("/srt", response_class=PlainTextResponse)
def srt_from_stt(payload: Any = Body(...)):
    """
    Accepts STT JSON payload and returns SRT text.

    Supported payload shapes:
    - {"data": {"words": [...], ...}}
    - {"words": [...], ...}
    - Or a list containing such an object as the first element.
    """
    try:
        words = _extract_words_from_payload(payload)

        if not isinstance(words, list) or not words:
            raise HTTPException(
                status_code=400,
                detail=(
                    "STT payload missing words[]. "
                    "Expected either payload['data']['words'] or payload['words']."
                ),
            )

        captions = words_to_captions(words)
        srt = captions_to_srt(captions)
        return PlainTextResponse(content=srt, media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SRT generation failed: {e}")


@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
    """
    Mux video + audio (and optional burned-in subtitles) into a final MP4.
    """
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    video_path = os.path.join(tmpdir, "video.mp4")
    audio_path = os.path.join(tmpdir, "audio.mp3")
    subs_path = os.path.join(tmpdir, "subs.srt")
    output_path = os.path.join(tmpdir, "output.mp4")

    # Download inputs
    download_file(str(request.video_url), video_path)
    download_file(str(request.audio_url), audio_path)

    if request.subtitles_url:
        download_file(str(request.subtitles_url), subs_path)

    # Probe audio duration to set -t
    audio_duration = get_audio_duration_seconds(audio_path)

    # Build FFmpeg command:
    # - loop video indefinitely
    # - cut at audio length
    # - burn subtitles if they exist
    vf_parts: List[str] = []
    if request.subtitles_url:
        # subtitle filter reads local file; tempdir paths are safe
        vf_parts.append(f"subtitles={subs_path}")

    vf_arg = ",".join(vf_parts) if vf_parts else None

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{audio_duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]

    if vf_arg:
        ffmpeg_cmd += ["-vf", vf_arg]

    ffmpeg_cmd.append(output_path)

    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT,
        )
    except subprocess.CalledProcessError as e:
        error_tail = e.stderr.decode(errors="ignore")[-4000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {error_tail}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out")
    finally:
        # Cleanup AFTER response fully sent
        background_tasks.add_task(shutil.rmtree, tmpdir, True)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"final_{job_id}.mp4",
    )
