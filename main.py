import os
import uuid
import shutil
import tempfile
import subprocess
import time
import base64
import threading
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
DOWNLOAD_TIMEOUT = 180
FFMPEG_TIMEOUT = 2400
MAX_BYTES = int(os.getenv("MAX_BYTES", str(300 * 1024 * 1024)))
VIDEO_TAIL_SECONDS = 1.0

TARGET_W = 720
TARGET_H = 1280
TARGET_FPS = 24
TARGET_PIXFMT = "yuv420p"
TARGET_TIMESCALE = "60000"

TARGET_AR = 44100
TARGET_AC = 2
NARRATION_AB = "192k"

DEBUG_PROBES = os.getenv("DEBUG_PROBES", "0").strip() not in ("0", "false", "False")

# ----------------------------
# Dropbox OAuth (Refresh Token)
# ----------------------------
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY", "").strip()
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET", "").strip()
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN", "").strip()

_dbx_lock = threading.Lock()
_dbx_access_token: Optional[str] = None
_dbx_access_token_exp: float = 0.0  # unix seconds


def _require_dropbox_oauth() -> None:
    if not (DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN):
        raise HTTPException(
            status_code=500,
            detail="Dropbox OAuth not configured. Set DROPBOX_APP_KEY, DROPBOX_APP_SECRET, DROPBOX_REFRESH_TOKEN.",
        )


def _refresh_dbx_access_token() -> None:
    """
    Exchange refresh_token -> short-lived access_token.
    Cached in-memory; refresh 60s early to avoid edge expiry.
    """
    _require_dropbox_oauth()

    token_url = "https://api.dropboxapi.com/oauth2/token"
    basic = base64.b64encode(
        f"{DROPBOX_APP_KEY}:{DROPBOX_APP_SECRET}".encode("utf-8")
    ).decode("ascii")

    r = requests.post(
        token_url,
        headers={"Authorization": f"Basic {basic}"},
        data={
            "grant_type": "refresh_token",
            "refresh_token": DROPBOX_REFRESH_TOKEN,
        },
        timeout=30,
    )

    if r.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Dropbox token refresh failed: {r.status_code} {r.text}",
        )

    payload = r.json()
    access_token = payload.get("access_token")
    expires_in = payload.get("expires_in")  # seconds

    if not access_token:
        raise HTTPException(
            status_code=500,
            detail=f"Dropbox token refresh missing access_token: {payload}",
        )

    # Some responses include expires_in; if absent, be conservative.
    ttl = int(expires_in) if expires_in else 1800

    global _dbx_access_token, _dbx_access_token_exp
    _dbx_access_token = access_token
    _dbx_access_token_exp = time.time() + max(60, ttl - 60)


def get_dbx_access_token() -> str:
    with _dbx_lock:
        if _dbx_access_token and time.time() < _dbx_access_token_exp:
            return _dbx_access_token
        _refresh_dbx_access_token()
        return _dbx_access_token  # type: ignore


# ----------------------------
# Request models
# ----------------------------
class MuxRequest(BaseModel):
    video_url: Optional[HttpUrl] = None
    audio_url: HttpUrl
    subtitles_url: Optional[HttpUrl] = None

    subtitle_profile: str = "menopause"  # "menopause" | "bible"
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None

    library_folder: Optional[str] = None
    library_count: int = 10


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
    return urlunparse(parsed._replace(query=urlencode(qs)))


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
                                detail="Downloaded HTML instead of media. Check link permissions / dl=1.",
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
# Helpers: Dropbox library
# ----------------------------
def _dbx_headers() -> Dict[str, str]:
    token = get_dbx_access_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def dropbox_list_mp4_paths(folder: str) -> List[str]:
    url = "https://api.dropboxapi.com/2/files/list_folder"
    body = {"path": folder, "recursive": False, "include_media_info": False}
    r = requests.post(url, headers=_dbx_headers(), json=body, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox list_folder failed: {r.text}")

    data = r.json()
    out: List[str] = []
    for e in data.get("entries", []):
        if e.get(".tag") == "file":
            p = e.get("path_lower") or e.get("path_display")
            if p and p.lower().endswith(".mp4"):
                out.append(p)

    if not out:
        raise HTTPException(status_code=400, detail=f"No .mp4 files found in Dropbox folder: {folder}")
    return out


def dropbox_temp_link(path: str) -> str:
    url = "https://api.dropboxapi.com/2/files/get_temporary_link"
    r = requests.post(url, headers=_dbx_headers(), json={"path": path}, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox get_temporary_link failed: {r.text}")
    link = r.json().get("link")
    if not link:
        raise HTTPException(status_code=400, detail="Dropbox get_temporary_link missing link")
    return link


def stable_pick(paths: List[str], count: int, seed: str) -> List[str]:
    if count <= 0:
        return []
    paths = list(paths)
    h = 2166136261
    for ch in seed:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    offset = h % len(paths)
    rotated = paths[offset:] + paths[:offset]
    return rotated[: min(count, len(rotated))]


# ----------------------------
# Helpers: ffmpeg/ffprobe
# ----------------------------
def _run(cmd: List[str], timeout: int = FFMPEG_TIMEOUT) -> str:
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
        tail = e.stderr.decode(errors="ignore")[-8000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {tail}")


def ffprobe_streams(path: str) -> str:
    return subprocess.check_output(
        ["ffprobe", "-hide_banner", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        stderr=subprocess.STDOUT,
        timeout=30,
    ).decode(errors="ignore")


def get_duration_seconds(path: str) -> float:
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
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
        if out in ("", "N/A"):
            raise ValueError(f"duration unavailable: {out}")
        d = float(out)
        if d <= 0:
            raise ValueError("invalid duration")
        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


def normalize_narration(src_audio: str, dst_m4a: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_audio,
        "-vn",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-af",
        (
            "asetpts=PTS-STARTPTS,"
            f"aresample={TARGET_AR}:async=1,"
            "aformat=channel_layouts=stereo,"
            "loudnorm=I=-16:TP=-1.5:LRA=11"
        ),
        "-ac",
        str(TARGET_AC),
        "-ar",
        str(TARGET_AR),
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-b:a",
        NARRATION_AB,
        "-map_metadata",
        "-1",
        "-metadata:s:a:0",
        "handler_name=SoundHandler",
        "-metadata:s:a:0",
        "title=Narration",
        "-movflags",
        "+faststart",
        dst_m4a,
    ]
    _run(cmd, timeout=FFMPEG_TIMEOUT)


def normalize_clip_video_only(src_mp4: str, dst_mp4: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_mp4,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-vf",
        f"scale={TARGET_W}:{TARGET_H},fps={TARGET_FPS},format={TARGET_PIXFMT}",
        "-fps_mode",
        "cfr",
        "-video_track_timescale",
        TARGET_TIMESCALE,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        TARGET_PIXFMT,
        "-metadata:s:v:0",
        "handler_name=VideoHandler",
        "-movflags",
        "+faststart",
        dst_mp4,
    ]
    _run(cmd, timeout=FFMPEG_TIMEOUT)


def concat_library_video_only(clean_paths: List[str], joined_out: str) -> None:
    list_path = os.path.join(os.path.dirname(joined_out), "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clean_paths:
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-map_metadata",
        "-1",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        TARGET_PIXFMT,
        "-an",
        "-metadata:s:v:0",
        "handler_name=VideoHandler",
        "-movflags",
        "+faststart",
        joined_out,
    ]
    _run(cmd, timeout=FFMPEG_TIMEOUT)


# ----------------------------
# Subtitles (SRT) generation (unchanged)
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
    if profile == "bible":
        max_chars, max_words, max_duration, min_duration = 60, 14, 3.2, 0.6
    else:
        max_chars, max_words, max_duration, min_duration = 38, 7, 2.2, 0.35

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


@app.get("/fonts", response_class=PlainTextResponse)
def list_fonts():
    try:
        out = subprocess.check_output(["fc-list", ":family"], timeout=10).decode(errors="ignore")
        families = sorted(set([x.split(":")[0].strip() for x in out.splitlines() if x.strip()]))
        return PlainTextResponse("\n".join(families))
    except Exception:
        return PlainTextResponse("fc-list not available in this container.")


# ----------------------------
# /mux endpoint (video audio removed, narration added)
# ----------------------------
@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    raw_audio_path = os.path.join(tmpdir, "narration_raw.mp3")
    narration_path = os.path.join(tmpdir, "narration_norm.m4a")

    subs_path = os.path.join(tmpdir, "subs.srt")

    source_video_path = None
    rendered_video_path = os.path.join(tmpdir, "video_only_render.mp4")
    output_path = os.path.join(tmpdir, "output.mp4")

    try:
        # 1) Download + normalize narration (ONLY audio in final)
        download_file(str(request.audio_url), raw_audio_path)
        normalize_narration(raw_audio_path, narration_path)

        # 2) Optional subtitles
        has_subs = False
        if request.subtitles_url:
            download_file(str(request.subtitles_url), subs_path)
            has_subs = os.path.getsize(subs_path) > 0

        # 3) Duration based on narration
        narration_dur = get_duration_seconds(narration_path)
        total_duration = max(0.0, narration_dur + VIDEO_TAIL_SECONDS)

        # 4) Resolve video source (library or direct)
        if request.library_folder:
            # OAuth refresh tokens are required; this ensures env vars exist early.
            _require_dropbox_oauth()

            all_paths = dropbox_list_mp4_paths(request.library_folder)
            picked = stable_pick(all_paths, request.library_count, seed=str(request.audio_url))

            clean_paths: List[str] = []
            for i, p in enumerate(picked):
                link = dropbox_temp_link(p)
                raw = os.path.join(tmpdir, f"lib_{i}.mp4")
                clean = os.path.join(tmpdir, f"lib_{i}_clean.mp4")
                download_file(link, raw)
                normalize_clip_video_only(raw, clean)
                clean_paths.append(clean)

            joined = os.path.join(tmpdir, "library_joined_video_only.mp4")
            concat_library_video_only(clean_paths, joined_out=joined)
            source_video_path = joined

        if request.video_url:
            source_video_path = os.path.join(tmpdir, "video.mp4")
            download_file(str(request.video_url), source_video_path)

        if not source_video_path:
            raise HTTPException(status_code=400, detail="Provide video_url or library_folder")

        # 5) Subtitle styling per profile
        vf_subs = None
        if has_subs:
            if request.subtitle_profile == "bible":
                font_name = request.subtitle_font or "DejaVu Sans"
                font_size = request.subtitle_font_size or 18
                margin_v = 90
                outline = 2
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
            vf_subs = f"subtitles={subs_path}:force_style='{style}'"

        # STEP A: render VIDEO-ONLY (no audio)
        vf_chain = []
        if vf_subs:
            vf_chain.append(vf_subs)
        vf_chain.append(f"scale={TARGET_W}:{TARGET_H}")
        vf_chain.append(f"fps={TARGET_FPS}")
        vf_chain.append(f"format={TARGET_PIXFMT}")
        vf_final = ",".join(vf_chain)

        step_a = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", source_video_path,
            "-t", f"{total_duration:.3f}",
            "-map_metadata", "-1",
            "-fflags", "+genpts",
            "-avoid_negative_ts", "make_zero",
            "-map", "0:v:0",
            "-an",
            "-vf", vf_final,
            "-fps_mode", "cfr",
            "-video_track_timescale", TARGET_TIMESCALE,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", TARGET_PIXFMT,
            "-movflags", "+faststart",
            "-metadata:s:v:0", "handler_name=VideoHandler",
            rendered_video_path,
        ]
        _run(step_a, timeout=FFMPEG_TIMEOUT)

        # STEP B: mux narration as ONLY audio
        step_b = [
            "ffmpeg", "-y",
            "-i", rendered_video_path,
            "-i", narration_path,
            "-t", f"{total_duration:.3f}",
            "-map_metadata", "-1",
            "-fflags", "+genpts",
            "-avoid_negative_ts", "make_zero",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-profile:a", "aac_low",
            "-b:a", NARRATION_AB,
            "-ar", str(TARGET_AR),
            "-ac", str(TARGET_AC),
            "-metadata:s:v:0", "handler_name=VideoHandler",
            "-metadata:s:a:0", "handler_name=SoundHandler",
            "-metadata:s:a:0", "title=Narration",
            "-disposition:a:0", "default",
            "-movflags", "+faststart",
            output_path,
        ]
        _run(step_b, timeout=FFMPEG_TIMEOUT)

        if DEBUG_PROBES:
            print("RENDERED VIDEO PROBE:", ffprobe_streams(rendered_video_path))
            print("NARRATION PROBE:", ffprobe_streams(narration_path))
            print("OUTPUT PROBE:", ffprobe_streams(output_path))

        return FileResponse(output_path, media_type="video/mp4", filename=f"final_{job_id}.mp4")

    finally:
        background_tasks.add_task(shutil.rmtree, tmpdir, True)
