import os
import re
import uuid
import json
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel, HttpUrl

app = FastAPI()

# ----------------------------
# Config
# ----------------------------
DOWNLOAD_TIMEOUT = 180          # seconds per file download
FFMPEG_TIMEOUT = 1800           # seconds (library concat can take time)
MAX_BYTES = 300 * 1024 * 1024   # 300MB safety limit
VIDEO_TAIL_SECONDS = 1.0

# Library normalisation target (match your Grok output)
TARGET_W = 720
TARGET_H = 1280
TARGET_FPS = 24
TARGET_PIXFMT = "yuv420p"
TARGET_TIMESCALE = "60000"  # helps avoid "timescale not set" quirks


# ----------------------------
# Request models
# ----------------------------
class MuxRequest(BaseModel):
    # If library_folder is provided, video_url can be null
    video_url: Optional[HttpUrl] = None
    audio_url: HttpUrl
    subtitles_url: Optional[HttpUrl] = None

    subtitle_profile: str = "menopause"  # "menopause" | "bible"
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None

    library_folder: Optional[str] = None   # Dropbox path like "/BIBLE/BibleLibrary/jerusalem_vertical/"
    library_count: int = 10                # how many clips to fetch
    transition: str = "fadeblack"          # currently only fadeblack supported
    transition_duration: float = 0.35


# ----------------------------
# Helpers: download
# ----------------------------
def _looks_like_html(first_bytes: bytes) -> bool:
    s = first_bytes.lstrip().lower()
    return (
        s.startswith(b"<!doctype html")
        or s.startswith(b"<html")
        or b"<head" in s[:2000]
    )


def _normalize_dropbox_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "dropbox.com" not in host:
        return url
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["dl"] = "1"
    new_query = urlencode(qs)
    return urlunparse(parsed._replace(query=new_query))


def download_file(url: str, dest_path: str) -> None:
    url = _normalize_dropbox_url(url)
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True) as r:
            r.raise_for_status()

            total = 0
            wrote_any = False

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue

                    if not wrote_any:
                        wrote_any = True
                        if _looks_like_html(chunk[:4096]):
                            raise HTTPException(
                                status_code=400,
                                detail="Downloaded HTML instead of media. Check Dropbox link permissions / dl=1."
                            )

                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(status_code=413, detail="File too large")

                    f.write(chunk)

            if not wrote_any:
                raise HTTPException(status_code=400, detail="Downloaded file is empty")

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")


# ----------------------------
# Helpers: ffmpeg/ffprobe
# ----------------------------
def _run(cmd: List[str], timeout: int = FFMPEG_TIMEOUT) -> str:
    """Run a subprocess and return stderr tail for logging/debug. Raises on failure."""
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return proc.stderr.decode(errors="ignore")[-4000:]
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out")
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="ignore")[-6000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {tail}")


def get_audio_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
        if out in ("", "N/A"):
            raise ValueError(f"duration unavailable: {out}")
        d = float(out)
        if d <= 0:
            raise ValueError("invalid duration")
        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


def normalize_clip(src_mp4: str, dst_mp4: str) -> None:
    """
    Make every library clip identical for concat:
    - keep ONLY video stream
    - remove attachments/subs/data
    - force scale, fps (CFR), pix_fmt
    - force track timescale
    - strip metadata
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", src_mp4,
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-map_metadata", "-1",
        "-vf", f"scale={TARGET_W}:{TARGET_H},fps={TARGET_FPS},format={TARGET_PIXFMT}",
        "-fps_mode", "cfr",
        "-video_track_timescale", TARGET_TIMESCALE,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", TARGET_PIXFMT,
        "-movflags", "+faststart",
        dst_mp4,
    ]
    _run(cmd, timeout=FFMPEG_TIMEOUT)


def concat_library(clean_paths: List[str], joined_out: str) -> None:
    """
    Concat using concat demuxer (very reliable) and re-encode once.
    """
    # Build list file
    list_path = os.path.join(os.path.dirname(joined_out), "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clean_paths:
            # concat demuxer needs: file '/path'
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-vf", f"format={TARGET_PIXFMT}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", TARGET_PIXFMT,
        "-movflags", "+faststart",
        joined_out,
    ]
    _run(cmd, timeout=FFMPEG_TIMEOUT)


# ----------------------------
# Subtitles (SRT) generation
# ----------------------------
def srt_timestamp(seconds: float) -> str:
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


def _extract_words_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("words"), list):
            return data["words"]
        if isinstance(payload.get("words"), list):
            return payload["words"]

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            data = first.get("data")
            if isinstance(data, dict) and isinstance(data.get("words"), list):
                return data["words"]
            if isinstance(first.get("words"), list):
                return first["words"]
    return []


def words_to_captions(words: List[Dict[str, Any]], profile: str) -> List[Dict[str, Any]]:
    # Defaults
    if profile == "bible":
        max_chars = 54
        max_words = 11
        max_duration = 3.2
        min_duration = 0.6
    else:
        max_chars = 38
        max_words = 7
        max_duration = 2.2
        min_duration = 0.35

    w = [x for x in words if x.get("type") == "word" and str(x.get("text", "")).strip()]
    caps: List[Dict[str, Any]] = []
    buf: List[str] = []
    start: Optional[float] = None
    last_end: Optional[float] = None

    def flush():
        nonlocal buf, start, last_end
        if not buf or start is None or last_end is None:
            buf, start, last_end = [], None, None
            return
        caps.append({"start": float(start), "end": float(last_end), "text": " ".join(buf).strip()})
        buf, start, last_end = [], None, None

    for item in w:
        t = str(item.get("text", "")).strip()
        s = float(item.get("start", 0.0))
        e = float(item.get("end", s))

        if start is None:
            start = s

        proposed = (" ".join(buf + [t])).strip()
        dur = e - start

        if len(buf) >= max_words or len(proposed) > max_chars or dur > max_duration:
            flush()
            start = s
            buf.append(t)
            last_end = e
            continue

        buf.append(t)
        last_end = e

    flush()

    for c in caps:
        if (c["end"] - c["start"]) < min_duration:
            c["end"] = c["start"] + min_duration

    return caps


def captions_to_srt(captions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, c in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(f"{srt_timestamp(c['start'])} --> {srt_timestamp(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


@app.post("/srt", response_class=PlainTextResponse)
def srt_from_stt(payload: Any = Body(...), profile: str = "menopause"):
    words = _extract_words_from_payload(payload)
    if not words:
        raise HTTPException(status_code=400, detail="STT payload missing words[]")
    captions = words_to_captions(words, profile=profile)
    return PlainTextResponse(content=captions_to_srt(captions), media_type="text/plain")


# ----------------------------
# Fonts helper endpoint
# ----------------------------
@app.get("/fonts", response_class=PlainTextResponse)
def list_fonts():
    # fc-list is available on most linux images; if not, return a helpful message
    try:
        out = subprocess.check_output(["fc-list", ":family"], timeout=10).decode(errors="ignore")
        # de-dup and keep it readable
        families = sorted(set([x.split(":")[0].strip() for x in out.splitlines() if x.strip()]))
        return PlainTextResponse("\n".join(families))
    except Exception:
        return PlainTextResponse("fc-list not available in this container.")


# ----------------------------
# /mux endpoint
# ----------------------------
@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    audio_path = os.path.join(tmpdir, "audio.mp3")
    subs_path = os.path.join(tmpdir, "subs.srt")
    output_path = os.path.join(tmpdir, "output.mp4")

    # download audio
    download_file(str(request.audio_url), audio_path)

    # subtitles (optional)
    has_subs = False
    if request.subtitles_url:
        download_file(str(request.subtitles_url), subs_path)
        has_subs = os.path.getsize(subs_path) > 0

    # duration
    audio_duration = get_audio_duration_seconds(audio_path)
    total_duration = max(0.0, audio_duration + VIDEO_TAIL_SECONDS)

    # choose video source:
    # 1) library mode -> build a joined video
    # 2) direct video_url -> use it
    video_path = None
    if request.library_folder:
        # NOTE: Your service already has Dropbox list_folder logic somewhere.
        # This stub assumes you have a function that returns direct-download URLs for N clips.
        # If not, keep your existing Dropbox API logic and just plug in the returned URLs below.
        raise HTTPException(
            status_code=400,
            detail="library_folder provided but Dropbox listing is not wired in this script stub."
        )

    if request.video_url:
        video_path = os.path.join(tmpdir, "video.mp4")
        download_file(str(request.video_url), video_path)

    if not video_path:
        raise HTTPException(
            status_code=400,
            detail="video_url is required unless library selection is implemented in the mux service."
        )

    # Subtitle styling per profile
    vf_arg = None
    if has_subs:
        if request.subtitle_profile == "bible":
            font_name = request.subtitle_font or "DejaVu Sans"
            font_size = request.subtitle_font_size or 28
            margin_v = 90
            outline = 3
        else:
            font_name = request.subtitle_font or "Arial"
            font_size = request.subtitle_font_size or 16
            margin_v = 40
            outline = 2

        style = (
            f"FontName={font_name},"
            f"FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF&,"
            f"OutlineColour=&H00000000&,"
            f"BorderStyle=1,"
            f"Outline={outline},"
            f"Shadow=0,"
            f"MarginV={margin_v},"
            f"Alignment=2"
        )
        vf_arg = f"subtitles={subs_path}:force_style='{style}'"

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", f"{total_duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
    ]
    if vf_arg:
        ffmpeg_cmd += ["-vf", vf_arg]

    ffmpeg_cmd.append(output_path)

    try:
        _run(ffmpeg_cmd, timeout=FFMPEG_TIMEOUT)
    finally:
        background_tasks.add_task(shutil.rmtree, tmpdir, True)

    return FileResponse(output_path, media_type="video/mp4", filename=f"final_{job_id}.mp4")
