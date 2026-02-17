import os
import re
import uuid
import shutil
import tempfile
import subprocess
import random
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
VIDEO_TAIL_SECONDS = 1.0        # seconds after audio ends

# Library normalization target (keeps concat stable)
TARGET_W = 720
TARGET_H = 1280
TARGET_FPS = 24

# Dropbox token for library mode
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "").strip()

# ----------------------------
# Request models
# ----------------------------
class MuxRequest(BaseModel):
    # Direct video mode (optional if library mode)
    video_url: Optional[HttpUrl] = None

    # Always required
    audio_url: HttpUrl

    # Optional subtitles
    subtitles_url: Optional[HttpUrl] = None

    # Subtitle options
    subtitle_profile: str = "menopause"          # "menopause" | "bible"
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None

    # Library mode options (optional)
    library_folder: Optional[str] = None         # e.g. "/BIBLE/BibleLibrary/jerusalem_vertical/"
    library_count: int = 10
    transition: str = "fadeblack"                # "fadeblack" (supported)
    transition_duration: float = 0.35


# ----------------------------
# Helpers (download / normalize)
# ----------------------------
def _looks_like_html(first_bytes: bytes) -> bool:
    s = first_bytes.lstrip().lower()
    return (
        s.startswith(b"<!doctype html")
        or s.startswith(b"<html")
        or b"<head" in s[:2000]
    )


def _normalize_dropbox_url(url: str) -> str:
    """
    Make Dropbox share links behave like direct-download links.
    Forces dl=1 (replaces dl=0).
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "dropbox.com" not in host:
        return url
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(qs)))


def download_file(url: str, dest_path: str) -> None:
    """
    Stream-download a file to dest_path with size and HTML checks.
    Automatically normalizes Dropbox share links to dl=1.
    """
    url = _normalize_dropbox_url(url)

    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True) as r:
            r.raise_for_status()

            total = 0
            wrote_any = False
            first_chunk = b""

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
                                    "If this is a Dropbox link, ensure it is a direct/share link (we force dl=1)."
                                ),
                            )

                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(status_code=413, detail="File too large")

                    f.write(chunk)

            if not wrote_any:
                raise HTTPException(status_code=400, detail="Downloaded file is empty")

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")


def _run(cmd: List[str], timeout: int = FFMPEG_TIMEOUT) -> None:
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        # Keep stderr tail around for debug (optional)
        tail = proc.stderr.decode(errors="ignore")[-1200:]
        if tail:
            print("CMD STDERR (tail):", tail)
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="ignore")[-4000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg/ffprobe failed: {tail}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out")


def get_media_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
        if out in ("", "N/A"):
            raise ValueError("duration unavailable")
        dur = float(out)
        if dur <= 0:
            raise ValueError("invalid duration")
        return dur
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


def normalize_clip_for_concat(src_path: str, dst_path: str) -> None:
    """
    CRITICAL:
    - Keep only the primary video stream (0:v:0)
    - Drop audio + attached pics + metadata
    - Re-encode to stable format (yuv420p, TARGET_WxTARGET_H, TARGET_FPS)
    This prevents concat failures like: mjpeg attached pic / timescale not set.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-map", "0:v:0",
        "-an",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-vf", f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
               f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2",
        "-r", str(TARGET_FPS),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
        "-movflags", "+faststart",
        dst_path,
    ]
    _run(cmd)


# ----------------------------
# Subtitles: STT -> SRT
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
    """
    Two profiles:
    - menopause: short, punchy captions
    - bible: longer phrases on screen (more words / longer duration)
    """
    profile = (profile or "menopause").strip().lower()

    if profile == "bible":
        max_chars = 58
        max_words = 12
        max_duration = 3.6
        min_duration = 0.6
        pause_split = 0.85
    else:
        max_chars = 38
        max_words = 7
        max_duration = 2.2
        min_duration = 0.35
        pause_split = 0.6

    w = [
        x for x in words
        if x.get("type") == "word" and str(x.get("text", "")).strip()
    ]

    caps: List[Dict[str, Any]] = []
    buf: List[str] = []
    start: Optional[float] = None
    last_end: Optional[float] = None

    def flush() -> None:
        nonlocal buf, start, last_end
        if not buf or start is None or last_end is None:
            buf, start, last_end = [], None, None
            return
        text = " ".join(buf).strip()
        caps.append({"start": float(start), "end": float(last_end), "text": text})
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

        # big pause => flush
        if last_end is not None and (s - last_end) > pause_split:
            flush()

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
def srt_from_stt(
    payload: Any = Body(...),
    profile: str = Query("menopause"),
):
    words = _extract_words_from_payload(payload)
    if not isinstance(words, list) or not words:
        raise HTTPException(
            status_code=400,
            detail="STT payload missing words[]. Expected payload['data']['words'] or payload['words']."
        )

    captions = words_to_captions(words, profile=profile)
    srt = captions_to_srt(captions)
    return PlainTextResponse(content=srt, media_type="text/plain")


# ----------------------------
# Fonts endpoint
# ----------------------------
@app.get("/fonts", response_class=PlainTextResponse)
def list_fonts():
    # Prefer fc-list if available
    try:
        out = subprocess.check_output(["fc-list", ":", "family"], stderr=subprocess.STDOUT, timeout=10).decode()
        # Normalize / de-dupe
        families = []
        seen = set()
        for line in out.splitlines():
            fam = line.split(",")[0].strip()
            if fam and fam not in seen:
                seen.add(fam)
                families.append(fam)
        families.sort(key=lambda x: x.lower())
        return PlainTextResponse("\n".join(families) + "\n")
    except Exception:
        # Fallback: list files in common font dirs
        roots = ["/usr/share/fonts", "/usr/local/share/fonts"]
        found = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith((".ttf", ".otf")):
                        found.append(os.path.join(dirpath, fn))
        found.sort()
        return PlainTextResponse("\n".join(found) + "\n")


# ----------------------------
# Dropbox helpers (library mode)
# ----------------------------
def _dropbox_headers() -> Dict[str, str]:
    if not DROPBOX_ACCESS_TOKEN:
        raise HTTPException(
            status_code=400,
            detail="DROPBOX_ACCESS_TOKEN not set (required for library_folder mode)."
        )
    return {"Authorization": f"Bearer {DROPBOX_ACCESS_TOKEN}", "Content-Type": "application/json"}


def dropbox_list_mp4s(folder_path: str) -> List[Dict[str, Any]]:
    """
    Returns Dropbox entries for files ending with .mp4 in folder_path.
    Requires scopes: files.metadata.read
    """
    url = "https://api.dropboxapi.com/2/files/list_folder"
    entries: List[Dict[str, Any]] = []
    body = {"path": folder_path, "recursive": False, "include_media_info": False}

    r = requests.post(url, headers=_dropbox_headers(), json=body, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox list_folder failed: {r.text}")

    data = r.json()
    entries.extend(data.get("entries", []))

    # pagination
    while data.get("has_more"):
        cursor = data.get("cursor")
        r = requests.post(
            "https://api.dropboxapi.com/2/files/list_folder/continue",
            headers=_dropbox_headers(),
            json={"cursor": cursor},
            timeout=30,
        )
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Dropbox list_folder/continue failed: {r.text}")
        data = r.json()
        entries.extend(data.get("entries", []))

    # Filter mp4 files
    mp4s = [
        e for e in entries
        if e.get(".tag") == "file" and str(e.get("name", "")).lower().endswith(".mp4")
    ]
    return mp4s


def dropbox_get_temp_link(path_lower: str) -> str:
    """
    Requires scopes: files.content.read
    """
    url = "https://api.dropboxapi.com/2/files/get_temporary_link"
    r = requests.post(url, headers=_dropbox_headers(), json={"path": path_lower}, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox get_temporary_link failed: {r.text}")
    return r.json()["link"]


def build_library_video(
    tmpdir: str,
    library_folder: str,
    library_count: int,
    transition: str,
    transition_duration: float,
) -> str:
    """
    1) list mp4s in dropbox folder
    2) choose N random
    3) download each
    4) normalize each to video-only stable encoding
    5) concat with simple fade-out/fade-in between clips
    Returns path to joined mp4.
    """
    mp4_entries = dropbox_list_mp4s(library_folder)
    if not mp4_entries:
        raise HTTPException(status_code=400, detail=f"No .mp4 files found in Dropbox folder: {library_folder}")

    count = max(1, min(int(library_count or 1), len(mp4_entries)))
    chosen = random.sample(mp4_entries, k=count)

    clean_paths: List[str] = []
    for i, entry in enumerate(chosen):
        path_lower = entry.get("path_lower")
        if not path_lower:
            continue
        link = dropbox_get_temp_link(path_lower)

        raw_path = os.path.join(tmpdir, f"lib_{i}.mp4")
        clean_path = os.path.join(tmpdir, f"lib_{i}_clean.mp4")

        download_file(link, raw_path)
        normalize_clip_for_concat(raw_path, clean_path)
        clean_paths.append(clean_path)

    if not clean_paths:
        raise HTTPException(status_code=400, detail="Failed to prepare any library clips.")

    # Build filter_complex to fade out/in then concat.
    # Fade style: fade-out at end of clip, fade-in at start of next (cut occurs during black -> black).
    # This is robust and avoids xfade complexity.
    parts = []
    labels = []

    for idx, p in enumerate(clean_paths):
        dur = get_media_duration_seconds(p)
        # Guard: if clip is very short, reduce fade duration
        fd = min(max(0.0, float(transition_duration)), max(0.0, dur * 0.25))
        fade_out_start = max(0.0, dur - fd)

        in_label = f"[{idx}:v]"
        out_label = f"[v{idx}]"

        # First clip: no fade-in; Last clip: no fade-out
        if idx == 0 and idx == (len(clean_paths) - 1):
            # single clip
            filt = f"{in_label}format=yuv420p,setpts=PTS-STARTPTS{out_label}"
        elif idx == 0:
            # first: fade out only
            filt = (
                f"{in_label}format=yuv420p,setpts=PTS-STARTPTS,"
                f"fade=t=out:st={fade_out_start:.3f}:d={fd:.3f}{out_label}"
            )
        elif idx == (len(clean_paths) - 1):
            # last: fade in only
            filt = (
                f"{in_label}format=yuv420p,setpts=PTS-STARTPTS,"
                f"fade=t=in:st=0:d={fd:.3f}{out_label}"
            )
        else:
            # middle: fade in + out
            filt = (
                f"{in_label}format=yuv420p,setpts=PTS-STARTPTS,"
                f"fade=t=in:st=0:d={fd:.3f},"
                f"fade=t=out:st={fade_out_start:.3f}:d={fd:.3f}{out_label}"
            )

        parts.append(filt)
        labels.append(out_label)

    # concat
    concat_out = "[vcat]"
    parts.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0{concat_out}")

    filter_complex = ";".join(parts)

    joined_path = os.path.join(tmpdir, "library_joined.mp4")
    cmd = ["ffmpeg", "-y"]
    for p in clean_paths:
        cmd += ["-i", p]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", concat_out,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(TARGET_FPS),
        "-movflags", "+faststart",
        joined_path,
    ]
    _run(cmd)

    return joined_path


# ----------------------------
# Subtitle styling (profiles)
# ----------------------------
def build_subtitle_vf(subs_path: str, profile: str, font: Optional[str], font_size: Optional[int]) -> str:
    profile = (profile or "menopause").strip().lower()

    # Defaults by profile
    if profile == "bible":
        default_font = "Arial"
        default_size = 22
        margin_v = 64
        outline = 2
        shadow = 0
    else:
        default_font = "Arial"
        default_size = 16
        margin_v = 40
        outline = 2
        shadow = 0

    font_name = font or default_font
    size = int(font_size) if font_size else default_size

    primary_colour = "&H00FFFFFF&"   # white
    outline_colour = "&H00000000&"   # black

    style = (
        f"FontName={font_name},"
        f"FontSize={size},"
        f"PrimaryColour={primary_colour},"
        f"OutlineColour={outline_colour},"
        f"BorderStyle=1,"
        f"Outline={outline},"
        f"Shadow={shadow},"
        f"MarginV={margin_v},"
        f"Alignment=2"
    )

    # tmp paths are space-free; if you change that, escape appropriately
    return f"subtitles={subs_path}:force_style='{style}'"


# ----------------------------
# /mux endpoint
# ----------------------------
@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    # Paths
    video_path = os.path.join(tmpdir, "video.mp4")
    audio_path = os.path.join(tmpdir, "audio.mp3")
    subs_path = os.path.join(tmpdir, "subs.srt")
    output_path = os.path.join(tmpdir, "output.mp4")

    try:
        # Download audio (always)
        download_file(str(request.audio_url), audio_path)

        # Determine video source:
        # 1) If library_folder set => build joined library video (no need for video_url)
        # 2) Else require video_url
        library_mode = bool(request.library_folder and str(request.library_folder).strip())

        if library_mode:
            joined = build_library_video(
                tmpdir=tmpdir,
                library_folder=str(request.library_folder),
                library_count=int(request.library_count or 10),
                transition=str(request.transition or "fadeblack"),
                transition_duration=float(request.transition_duration or 0.35),
            )
            # We'll use joined as the input video
            video_input = joined
            loop_video = True   # loop the joined sequence to cover long audio
        else:
            if not request.video_url:
                raise HTTPException(
                    status_code=400,
                    detail="video_url is required unless library selection is implemented in the mux service."
                )
            download_file(str(request.video_url), video_path)
            video_input = video_path
            loop_video = True   # your previous behaviour

        # Optional subtitles
        has_subs = False
        if request.subtitles_url:
            download_file(str(request.subtitles_url), subs_path)
            try:
                has_subs = os.path.getsize(subs_path) > 0
            except OSError:
                has_subs = False

        # Compute duration target
        audio_duration = get_media_duration_seconds(audio_path)
        total_duration = max(0.0, audio_duration + VIDEO_TAIL_SECONDS)

        # Build ffmpeg command
        ffmpeg_cmd = ["ffmpeg", "-y"]

        if loop_video:
            ffmpeg_cmd += ["-stream_loop", "-1"]

        ffmpeg_cmd += ["-i", video_input, "-i", audio_path]

        # video/audio mapping
        ffmpeg_cmd += [
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

        # Burn subtitles if present
        if has_subs:
            vf = build_subtitle_vf(
                subs_path=subs_path,
                profile=request.subtitle_profile,
                font=request.subtitle_font,
                font_size=request.subtitle_font_size,
            )
            ffmpeg_cmd += ["-vf", vf]

        ffmpeg_cmd.append(output_path)

        print("FFMPEG CMD:", " ".join(ffmpeg_cmd))
        _run(ffmpeg_cmd)

    finally:
        # Cleanup after response is sent
        background_tasks.add_task(shutil.rmtree, tmpdir, True)

    return FileResponse(output_path, media_type="video/mp4", filename=f"final_{job_id}.mp4")
