import os
import uuid
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse  # NEW

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

# How long to keep the video running after audio stops (in seconds)
VIDEO_TAIL_SECONDS = 1.0


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


def _normalize_dropbox_url(url: str) -> str:
    """
    Make Dropbox share links behave like direct-download links.

    - Ensures ?dl=1 is present (or replaces dl=0 with dl=1)
    - Works for .../s/..., .../scl/fi/..., etc.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "dropbox.com" not in host:
        return url

    # Parse existing query params, force dl=1
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["dl"] = "1"

    new_query = urlencode(qs)
    normalized = urlunparse(parsed._replace(query=new_query))
    return normalized


def download_file(url: str, dest_path: str) -> None:
    """
    Stream-download a file to dest_path with size and HTML checks.
    Automatically normalizes Dropbox URLs to direct-download links.
    """
    # normalize Dropbox share URLs
    url = _normalize_dropbox_url(url)

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

                        # If it's still HTML after normalization, bail.
                        if _looks_like_html(first_chunk):
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    "Downloaded HTML instead of a media file. "
                                    "If this is a Dropbox link, ensure it is a direct or temporary link."
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

    # List of bundles (defensive)
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
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
    """
    Mux video + audio (and optional burned-in subtitles) into a final MP4.

    - Video: whatever resolution/AR the source has (your Grok output is already 9:16).
    - Audio: your ElevenLabs track.
    - Optional: burn SRT subtitles on top if provided.
    """
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    video_path = os.path.join(tmpdir, "video.mp4")
    audio_path = os.path.join(tmpdir, "audio.mp3")
    subs_path = os.path.join(tmpdir, "subs.srt")
    output_path = os.path.join(tmpdir, "output.mp4")

    # Download inputs (robust to Dropbox share URLs)
    download_file(str(request.video_url), video_path)
    download_file(str(request.audio_url), audio_path)

    has_subs = False
    if request.subtitles_url:
        download_file(str(request.subtitles_url), subs_path)
        try:
            if os.path.getsize(subs_path) > 0:
                has_subs = True
        except OSError:
            has_subs = False

    # Probe audio duration to set -t (stop slightly AFTER audio length)
    audio_duration = get_audio_duration_seconds(audio_path)
    total_duration = max(0.0, audio_duration + VIDEO_TAIL_SECONDS)

    # Build subtitles filter with styling if we have subs
    vf_arg = None
    if has_subs:
        # Tweak these to taste
        font_name = "Arial"           # must exist in the container
        font_size = 20                # 30–40 tends to be nice for 1080p vertical
        margin_v = 60                 # pixels from bottom; higher = further up
        outline = 2                   # outline thickness
        shadow = 0                    # drop shadow amount
        primary_colour = "&H00FFFFFF&"   # white
        outline_colour = "&H00000000&"   # black

        style = (
            f"FontName={font_name},"
            f"FontSize={font_size},"
            f"PrimaryColour={primary_colour},"
            f"OutlineColour={outline_colour},"
            f"BorderStyle=1,"
            f"Outline={outline},"
            f"Shadow={shadow},"
            f"MarginV={margin_v},"
            f"Alignment=2"  # bottom-centre; keep this for “just off the bottom”
        )

        # Note: we rely on FFmpeg parsing this as one argument; tmp paths have no spaces by default
        vf_arg = f"subtitles={subs_path}:force_style='{style}'"

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
        f"{total_duration:.3f}",   # <-- audio duration + 1 second
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

    print("FFMPEG CMD:", " ".join(ffmpeg_cmd))

    try:
        proc = subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT,
        )
        stderr_tail = proc.stderr.decode(errors="ignore")[-1000:]
        if stderr_tail:
            print("FFMPEG STDERR (tail):", stderr_tail)

    except subprocess.CalledProcessError as e:
        error_tail = e.stderr.decode(errors="ignore")[-4000:]
        print("FFMPEG FAILED, STDERR TAIL:\n", error_tail)
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
