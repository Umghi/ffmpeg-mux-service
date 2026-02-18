import os
import uuid
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
FFMPEG_TIMEOUT = 2400           # seconds (library concat can take time)
MAX_BYTES = int(os.getenv("MAX_BYTES", str(300 * 1024 * 1024)))  # 300MB default
VIDEO_TAIL_SECONDS = 1.0

# Normalization target (match your Grok outputs)
TARGET_W = 720
TARGET_H = 1280
TARGET_FPS = 24
TARGET_PIXFMT = "yuv420p"
TARGET_TIMESCALE = "60000"  # helps avoid "timescale not set" quirks

TARGET_AR = 44100
TARGET_AC = 2
TARGET_AB = "128k"

DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN", "").strip()


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

    # Ambience mixing controls (Option B)
    include_ambience: bool = True          # if input video has audio, mix it under narration
    ambience_volume: float = 0.12          # ambience gain before ducking (0.08–0.18 typical)

    transition: str = "fadeblack"          # currently not used (kept for compatibility)
    transition_duration: float = 0.35      # currently not used (kept for compatibility)


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
    """
    Converts www.dropbox.com shared links to dl=1 direct download.
    Works for both ?dl=0 and ?dl=1 cases.
    """
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
                                detail="Downloaded HTML instead of media. Check link permissions / dl=1."
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
    if not DROPBOX_TOKEN:
        raise HTTPException(status_code=500, detail="DROPBOX_TOKEN not set")
    return {
        "Authorization": f"Bearer {DROPBOX_TOKEN}",
        "Content-Type": "application/json",
    }


def dropbox_list_mp4_paths(folder: str) -> List[str]:
    """
    Returns a list of Dropbox file paths for .mp4 items in folder (non-recursive).
    """
    url = "https://api.dropboxapi.com/2/files/list_folder"
    body = {"path": folder, "recursive": False, "include_media_info": False}
    r = requests.post(url, headers=_dbx_headers(), json=body, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox list_folder failed: {r.text}")

    data = r.json()
    entries = data.get("entries", [])
    out = []
    for e in entries:
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
    return r.json().get("link")


def stable_pick(paths: List[str], count: int, seed: str) -> List[str]:
    """
    Deterministic selection: hash seed -> rotates list.
    """
    if count <= 0:
        return []
    paths = list(paths)
    # simple deterministic offset
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
    """Run a subprocess and return stderr tail. Raises on failure."""
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


def get_duration_seconds(path: str) -> float:
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


def video_has_audio_stream(path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15).decode().strip()
        return out != ""
    except Exception:
        return False


def normalize_clip(src_mp4: str, dst_mp4: str, keep_audio: bool) -> None:
    """
    Make every library clip concat-safe:
    - strips attachments/subs/data
    - forces scale, fps (CFR), pix_fmt
    - forces track timescale
    - optionally keeps + normalizes audio (for ambience mixing later)
    """
    if keep_audio:
        has_a = video_has_audio_stream(src_mp4)
        dur = get_duration_seconds(src_mp4)

        if has_a:
            cmd = [
                "ffmpeg", "-y",
                "-i", src_mp4,
                "-map", "0:v:0",
                "-map", "0:a:0",
                "-sn", "-dn",
                "-map_metadata", "-1",
                "-vf", f"scale={TARGET_W}:{TARGET_H},fps={TARGET_FPS},format={TARGET_PIXFMT}",
                "-fps_mode", "cfr",
                "-video_track_timescale", TARGET_TIMESCALE,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", TARGET_PIXFMT,
                "-af", f"aresample={TARGET_AR}:async=1",
                "-ac", str(TARGET_AC),
                "-ar", str(TARGET_AR),
                "-c:a", "aac",
                "-b:a", TARGET_AB,
                "-t", f"{dur:.3f}",
                "-movflags", "+faststart",
                dst_mp4,
            ]
        else:
            # Generate silent audio if clip has no audio, so concat doesn't break
            cmd = [
                "ffmpeg", "-y",
                "-i", src_mp4,
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate={TARGET_AR}",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-sn", "-dn",
                "-map_metadata", "-1",
                "-vf", f"scale={TARGET_W}:{TARGET_H},fps={TARGET_FPS},format={TARGET_PIXFMT}",
                "-fps_mode", "cfr",
                "-video_track_timescale", TARGET_TIMESCALE,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", TARGET_PIXFMT,
                "-c:a", "aac",
                "-b:a", TARGET_AB,
                "-t", f"{dur:.3f}",
                "-movflags", "+faststart",
                dst_mp4,
            ]
    else:
        # Video-only normalization (most robust)
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


def concat_library(clean_paths: List[str], joined_out: str, keep_audio: bool) -> None:
    """
    Concat using concat demuxer and re-encode once.
    If keep_audio=True, re-encodes audio too.
    """
    list_path = os.path.join(os.path.dirname(joined_out), "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clean_paths:
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", TARGET_PIXFMT,
        "-movflags", "+faststart",
    ]

    if keep_audio:
        cmd += [
            "-c:a", "aac",
            "-b:a", TARGET_AB,
            "-ar", str(TARGET_AR),
            "-ac", str(TARGET_AC),
        ]
    else:
        cmd += ["-an"]

    cmd.append(joined_out)
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
    if profile == "bible":
        max_chars = 60
        max_words = 14
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


@app.get("/fonts", response_class=PlainTextResponse)
def list_fonts():
    try:
        out = subprocess.check_output(["fc-list", ":family"], timeout=10).decode(errors="ignore")
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

    audio_path = os.path.join(tmpdir, "narration.mp3")
    subs_path = os.path.join(tmpdir, "subs.srt")
    output_path = os.path.join(tmpdir, "output.mp4")

    try:
        # 1) Download narration
        download_file(str(request.audio_url), audio_path)

        # 2) Optional subtitles
        has_subs = False
        if request.subtitles_url:
            download_file(str(request.subtitles_url), subs_path)
            has_subs = os.path.getsize(subs_path) > 0

        # 3) Compute final duration
        narration_dur = get_duration_seconds(audio_path)
        total_duration = max(0.0, narration_dur + VIDEO_TAIL_SECONDS)

        # 4) Resolve video source
        video_path = None

        # Library mode: download N clips from Dropbox, normalize, concat
        if request.library_folder:
            if not DROPBOX_TOKEN:
                raise HTTPException(status_code=500, detail="library_folder used but DROPBOX_TOKEN is not set")

            all_paths = dropbox_list_mp4_paths(request.library_folder)
            picked = stable_pick(all_paths, request.library_count, seed=str(request.audio_url))

            raw_paths: List[str] = []
            clean_paths: List[str] = []

            # If we plan to include ambience, we must keep audio through normalization+concat
            keep_library_audio = bool(request.include_ambience)

            for i, p in enumerate(picked):
                link = dropbox_temp_link(p)
                raw = os.path.join(tmpdir, f"lib_{i}.mp4")
                clean = os.path.join(tmpdir, f"lib_{i}_clean.mp4")
                download_file(link, raw)
                normalize_clip(raw, clean, keep_audio=keep_library_audio)
                raw_paths.append(raw)
                clean_paths.append(clean)

            joined = os.path.join(tmpdir, "library_joined.mp4")
            concat_library(clean_paths, joined_out=joined, keep_audio=keep_library_audio)
            video_path = joined

        # Direct video mode
        if request.video_url:
            video_path = os.path.join(tmpdir, "video.mp4")
            download_file(str(request.video_url), video_path)

        if not video_path:
            raise HTTPException(status_code=400, detail="Provide video_url or library_folder")

        # 5) Subtitle styling per profile
        vf_arg = None
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
            vf_arg = f"subtitles={subs_path}:force_style='{style}'"

        # 6) Build ffmpeg command:
        #    - ALWAYS use narration as primary
        #    - Option B: mix ambience from video (if exists) under narration
        has_video_audio = request.include_ambience and video_has_audio_stream(video_path)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", video_path,   # input 0
            "-i", audio_path,   # input 1
            "-t", f"{total_duration:.3f}",
        ]

        # Video encode args
        video_out_args = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", TARGET_PIXFMT,
            "-movflags", "+faststart",
        ]
        if vf_arg:
            video_out_args = ["-vf", vf_arg] + video_out_args

        if has_video_audio:
            # Mix ambience with narration:
            # - ambience volume low
            # - duck ambience when narration present
            # - limiter to avoid clipping
            amb_vol = max(0.0, min(float(request.ambience_volume), 1.0))
            audio_filter = (
                f"[0:a]aformat=sample_rates={TARGET_AR}:channel_layouts=stereo,volume={amb_vol}[amb];"
                f"[1:a]aformat=sample_rates={TARGET_AR}:channel_layouts=stereo,volume=1.0[nar];"
                f"[amb][nar]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=250[ducked];"
                f"[ducked][nar]amix=inputs=2:duration=first:dropout_transition=2,alimiter=limit=0.95[aout]"
            )

            ffmpeg_cmd += [
                "-filter_complex", audio_filter,
                "-map", "0:v:0",
                "-map", "[aout]",
                "-c:a", "aac",
                "-b:a", "192k",
            ] + video_out_args + [output_path]
        else:
            # Narration only (video audio stripped)
            ffmpeg_cmd += [
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:a", "aac",
                "-b:a", "192k",
            ] + video_out_args + [output_path]

        _run(ffmpeg_cmd, timeout=FFMPEG_TIMEOUT)

        return FileResponse(output_path, media_type="video/mp4", filename=f"final_{job_id}.mp4")

    finally:
        background_tasks.add_task(shutil.rmtree, tmpdir, True)
