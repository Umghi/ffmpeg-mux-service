import os
import uuid
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Query
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

    # NEW: subtitle profiles
    # "menopause" (default) or "bible"
    subtitle_profile: str = "menopause"

    # OPTIONAL: override font and size per request (Make can pass these)
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None


# ----------------------------
# Helpers
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
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
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
    pause_split: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Turns word-level timestamps into caption chunks.
    """
    w = [
        x for x in words
        if x.get("type") == "word" and str(x.get("text", "")).strip()
    ]

    caps: List[Dict[str, Any]] = []
    buf: List[str] = []
    start: Optional[float] = None
    last_end: Optional[float] = None
    last_word_end: Optional[float] = None

    def flush() -> None:
        nonlocal buf, start, last_end
        if not buf or start is None or last_end is None:
            buf = []
            start = None
            last_end = None
            return

        text = " ".join(buf).strip()
        caps.append({"start": float(start), "end": float(last_end), "text": text})
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

        if (
            len(buf) >= max_words
            or len(proposed) > max_chars
            or dur > max_duration
        ):
            flush()
            start = s
            buf.append(t)
            last_end = e
            last_word_end = e
            continue

        buf.append(t)
        last_end = e

        # long pause => new caption
        if last_word_end is not None and (s - last_word_end) > pause_split:
            flush()

        last_word_end = e

    flush()

    # ensure minimum duration
    for c in caps:
        if (c["end"] - c["start"]) < min_duration:
            c["end"] = c["start"] + min_duration

    return caps


def words_to_captions_profile(words: List[Dict[str, Any]], profile: str) -> List[Dict[str, Any]]:
    """
    Profile-based caption chunking:
      - menopause: short, punchy
      - bible: longer chunks, slower changes, 2-line friendly
    """
    p = (profile or "menopause").strip().lower()
    if p == "bible":
        return words_to_captions(
            words,
            max_chars=90,       # enough for 2 lines
            max_words=16,       # more words per caption
            max_duration=4.5,   # keep on screen longer
            min_duration=0.75,  # avoid flicker
            pause_split=0.85,   # allow longer pauses before splitting
        )
    # default: menopause
    return words_to_captions(
        words,
        max_chars=38,
        max_words=7,
        max_duration=2.2,
        min_duration=0.35,
        pause_split=0.6,
    )


def wrap_to_lines(text: str, max_line_chars: int = 42, max_lines: int = 2) -> str:
    """
    Simple word-wrap into up to max_lines lines for SRT.
    """
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for w in words:
        add_len = (1 if cur else 0) + len(w)
        if cur_len + add_len <= max_line_chars:
            cur.append(w)
            cur_len += add_len
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
            if len(lines) >= max_lines - 1:
                # squash remaining words into last line (soft cap)
                remaining = [*cur, *words[words.index(w)+1:]]
                last = " ".join(remaining)
                lines.append(last[: max_line_chars * 2])
                cur = []
                break

    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))

    return "\n".join(lines[:max_lines]).strip()


def captions_to_srt(captions: List[Dict[str, Any]], wrap: bool = False) -> str:
    lines: List[str] = []
    for idx, c in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(f"{srt_timestamp(c['start'])} --> {srt_timestamp(c['end'])}")
        txt = c["text"]
        if wrap:
            txt = wrap_to_lines(txt, max_line_chars=42, max_lines=2)
        lines.append(txt)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


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


# ----------------------------
# Endpoints
# ----------------------------
@app.post("/srt", response_class=PlainTextResponse)
def srt_from_stt(payload: Any = Body(...), profile: str = Query("menopause")):
    """
    Accepts STT JSON payload and returns SRT text.

    Use:
      - /srt?profile=menopause  (default)
      - /srt?profile=bible      (longer captions + 2-line wrapping)
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

        p = (profile or "menopause").strip().lower()
        captions = words_to_captions_profile(words, p)
        srt = captions_to_srt(captions, wrap=(p == "bible"))
        return PlainTextResponse(content=srt, media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SRT generation failed: {e}")


@app.get("/fonts", response_class=PlainTextResponse)
def list_fonts():
    """
    Lists available font families inside the container (requires fontconfig installed).
    """
    try:
        cmd = ["bash", "-lc", "fc-list : family | sed 's/,/\\n/g' | sort -u | head -n 200"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode("utf-8", errors="ignore")
        if not out.strip():
            return PlainTextResponse(
                content="No fonts listed. Ensure 'fontconfig' is installed in the container.",
                media_type="text/plain",
            )
        return PlainTextResponse(content=out, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Font listing failed: {e}")


@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
    """
    Mux video + audio (and optional burned-in subtitles) into a final MP4.

    - Video: looped to match audio duration + tail
    - Audio: your ElevenLabs track
    - Optional: burn SRT subtitles if provided

    New:
      - subtitle_profile: "menopause" or "bible"
      - subtitle_font / subtitle_font_size overrides
    """
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    video_path = os.path.join(tmpdir, "video.mp4")
    audio_path = os.path.join(tmpdir, "audio.mp3")
    subs_path = os.path.join(tmpdir, "subs.srt")
    output_path = os.path.join(tmpdir, "output.mp4")

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

    audio_duration = get_audio_duration_seconds(audio_path)
    total_duration = max(0.0, audio_duration + VIDEO_TAIL_SECONDS)

    vf_arg = None
    if has_subs:
        profile = (request.subtitle_profile or "menopause").strip().lower()

        # Optional overrides
        font_override = request.subtitle_font
        size_override = request.subtitle_font_size

        if profile == "bible":
            # Bible: larger, calmer, more readable; often benefits from a subtle background box
            font_name = font_override or "DejaVu Serif"
            font_size = int(size_override or 26)
            margin_v = 90
            outline = 2
            shadow = 0

            # ASS colors are &HAABBGGRR (alpha first). Larger alpha = more transparent.
            primary_colour = "&H00FFFFFF&"   # white
            outline_colour = "&H00000000&"   # black
            back_colour = "&H80000000&"      # semi-transparent black background box

            style = (
                f"FontName={font_name},"
                f"FontSize={font_size},"
                f"PrimaryColour={primary_colour},"
                f"OutlineColour={outline_colour},"
                f"BackColour={back_colour},"
                f"BorderStyle=3,"    # background box
                f"Outline={outline},"
                f"Shadow={shadow},"
                f"MarginV={margin_v},"
                f"Alignment=2"
            )
        else:
            # Menopause: smaller, punchier; no background box by default
            font_name = font_override or "Arial"
            font_size = int(size_override or 16)
            margin_v = 40
            outline = 2
            shadow = 0
            primary_colour = "&H00FFFFFF&"
            outline_colour = "&H00000000&"

            style = (
                f"FontName={font_name},"
                f"FontSize={font_size},"
                f"PrimaryColour={primary_colour},"
                f"OutlineColour={outline_colour},"
                f"BorderStyle=1,"
                f"Outline={outline},"
                f"Shadow={shadow},"
                f"MarginV={margin_v},"
                f"Alignment=2"
            )

        # If you bundle custom fonts later, add: fontsdir=/app/fonts:
        vf_arg = f"subtitles={subs_path}:force_style='{style}'"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop", "-1",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", f"{total_duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
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
        stderr_tail = proc.stderr.decode(errors="ignore")[-1200:]
        if stderr_tail:
            print("FFMPEG STDERR (tail):", stderr_tail)

    except subprocess.CalledProcessError as e:
        error_tail = e.stderr.decode(errors="ignore")[-4000:]
        print("FFMPEG FAILED, STDERR TAIL:\n", error_tail)
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {error_tail}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out")
    finally:
        background_tasks.add_task(shutil.rmtree, tmpdir, True)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"final_{job_id}.mp4",
    )
