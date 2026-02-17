import os
import uuid
import shutil
import random
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
DOWNLOAD_TIMEOUT = 120
FFMPEG_TIMEOUT = 900
MAX_BYTES = 300 * 1024 * 1024
VIDEO_TAIL_SECONDS = 1.0

TARGET_W = 1080
TARGET_H = 1920
TARGET_FPS = 30

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "").strip()
DROPBOX_API = "https://api.dropboxapi.com/2"


# ----------------------------
# Request models
# ----------------------------
class MuxRequest(BaseModel):
    video_url: Optional[HttpUrl] = None
    audio_url: HttpUrl
    subtitles_url: Optional[HttpUrl] = None

    subtitle_profile: str = "menopause"      # "menopause" | "bible"
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None

    library_folder: Optional[str] = None     # Dropbox folder path e.g. "/BIBLE/BibleLibrary/jerusalem_vertical/"
    library_count: int = 10                  # how many unique clips to sample
    transition: str = "fadeblack"            # currently only fadeblack supported
    transition_duration: float = 0.35        # seconds


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
    return urlunparse(parsed._replace(query=urlencode(qs)))


def download_file(url: str, dest_path: str) -> None:
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
                                detail="Downloaded HTML instead of media. Check Dropbox link type.",
                            )

                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(status_code=413, detail="File too large")

                    f.write(chunk)

            if not wrote_any:
                raise HTTPException(status_code=400, detail="Downloaded file is empty")

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")


def _run_ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
        if out in ("", "N/A"):
            raise ValueError("duration unavailable")
        d = float(out)
        if d <= 0:
            raise ValueError("invalid duration")
        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


def get_audio_duration_seconds(path: str) -> float:
    return _run_ffprobe_duration(path)


def get_video_duration_seconds(path: str) -> float:
    return _run_ffprobe_duration(path)


# ----------------------------
# Dropbox library helpers
# ----------------------------
def _dbx_headers() -> Dict[str, str]:
    if not DROPBOX_ACCESS_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="DROPBOX_ACCESS_TOKEN is not set on the mux service (Railway env var).",
        )
    return {
        "Authorization": f"Bearer {DROPBOX_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def dropbox_list_mp4_paths(folder_path: str) -> List[str]:
    """
    Returns file paths for .mp4 files in the given Dropbox folder.
    """
    folder_path = folder_path.rstrip("/")
    url = f"{DROPBOX_API}/files/list_folder"
    payload = {"path": folder_path}

    r = requests.post(url, headers=_dbx_headers(), json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox list_folder failed: {r.text}")

    data = r.json()
    entries = data.get("entries", []) or []
    mp4s = []

    for e in entries:
        if e.get(".tag") != "file":
            continue
        name = (e.get("name") or "").lower()
        if name.endswith(".mp4"):
            mp4s.append(e.get("path_lower") or e.get("path_display"))

    # NOTE: no pagination handling here; keep your folder reasonably sized.
    return [p for p in mp4s if p]


def dropbox_get_temporary_link(file_path: str) -> str:
    url = f"{DROPBOX_API}/files/get_temporary_link"
    payload = {"path": file_path}
    r = requests.post(url, headers=_dbx_headers(), json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox get_temporary_link failed: {r.text}")
    data = r.json()
    link = data.get("link")
    if not link:
        raise HTTPException(status_code=400, detail="Dropbox temporary link missing")
    return link


# ----------------------------
# SRT generation helpers
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


def words_to_captions(
    words: List[Dict[str, Any]],
    max_chars: int,
    max_words: int,
    max_duration: float,
    min_duration: float,
) -> List[Dict[str, Any]]:
    w = [x for x in words if x.get("type") == "word" and str(x.get("text", "")).strip()]

    caps: List[Dict[str, Any]] = []
    buf: List[str] = []
    start: Optional[float] = None
    last_end: Optional[float] = None

    def flush() -> None:
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

        if (len(buf) >= max_words) or (len(proposed) > max_chars) or (dur > max_duration):
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


# ----------------------------
# Endpoints
# ----------------------------
@app.post("/srt", response_class=PlainTextResponse)
def srt_from_stt(payload: Any = Body(...), profile: str = Query("menopause")):
    """
    /srt?profile=menopause  (snappier)
    /srt?profile=bible      (more text on screen)
    """
    words = _extract_words_from_payload(payload)
    if not isinstance(words, list) or not words:
        raise HTTPException(
            status_code=400,
            detail="STT payload missing words[]. Expected payload['data']['words'] or payload['words'].",
        )

    if str(profile).lower() == "bible":
        captions = words_to_captions(words, max_chars=56, max_words=12, max_duration=3.6, min_duration=0.60)
    else:
        captions = words_to_captions(words, max_chars=38, max_words=7, max_duration=2.2, min_duration=0.35)

    return PlainTextResponse(content=captions_to_srt(captions), media_type="text/plain")


@app.get("/fonts", response_class=PlainTextResponse)
def list_fonts():
    try:
        out = subprocess.check_output(["fc-list"], stderr=subprocess.STDOUT, timeout=20).decode(errors="ignore")
        return PlainTextResponse("\n".join(out.splitlines()[:400]), media_type="text/plain")
    except Exception as e:
        return PlainTextResponse(f"Could not list fonts (fc-list missing?). Error: {e}", media_type="text/plain")


# ----------------------------
# Mux core
# ----------------------------
def _subtitle_style(profile: str, subtitle_font: Optional[str], subtitle_font_size: Optional[int]) -> str:
    profile = (profile or "menopause").lower()
    if profile == "bible":
        default_font = "Georgia"
        default_size = 20
        margin_v = 70
        outline = 2
    else:
        default_font = "Arial"
        default_size = 16
        margin_v = 40
        outline = 2

    font_name = subtitle_font or default_font
    font_size = subtitle_font_size or default_size

    primary_colour = "&H00FFFFFF&"
    outline_colour = "&H00000000&"

    return (
        f"FontName={font_name},"
        f"FontSize={font_size},"
        f"PrimaryColour={primary_colour},"
        f"OutlineColour={outline_colour},"
        f"BorderStyle=1,"
        f"Outline={outline},"
        f"Shadow=0,"
        f"MarginV={margin_v},"
        f"Alignment=2"
    )


def _build_library_video(
    tmpdir: str,
    library_folder: str,
    library_count: int,
    needed_duration: float,
    transition_duration: float,
) -> str:
    """
    Downloads N random clips from Dropbox folder, repeats them as needed, concats with fade-to-black,
    outputs a temp video file path.
    """
    mp4_paths = dropbox_list_mp4_paths(library_folder)
    if not mp4_paths:
        raise HTTPException(status_code=400, detail=f"No .mp4 files found in Dropbox folder: {library_folder}")

    # sample unique set
    k = max(1, min(int(library_count or 10), len(mp4_paths)))
    sampled = random.sample(mp4_paths, k=k)

    # download and measure durations
    local_files: List[str] = []
    local_durs: List[float] = []

    for i, p in enumerate(sampled):
        link = dropbox_get_temporary_link(p)
        lp = os.path.join(tmpdir, f"lib_{i}.mp4")
        download_file(link, lp)
        d = get_video_duration_seconds(lp)
        local_files.append(lp)
        local_durs.append(d)

    # build a sequence long enough
    seq_files: List[str] = []
    seq_durs: List[float] = []
    total = 0.0
    idx = 0

    # repeat the sampled set until we have enough
    while total < needed_duration + 0.5:
        f = local_files[idx % len(local_files)]
        d = local_durs[idx % len(local_durs)]
        seq_files.append(f)
        seq_durs.append(d)
        total += d
        idx += 1
        if idx > 200:  # safety
            break

    out_path = os.path.join(tmpdir, "library_concat.mp4")

    # build ffmpeg filter_complex
    # normalize each input to 1080x1920, 30fps, SAR=1, and fade out/in (fadeblack)
    inputs = []
    for f in seq_files:
        inputs += ["-i", f]

    filter_lines = []
    v_labels = []

    for i, d in enumerate(seq_durs):
        # guard for very short clips
        td = float(max(0.05, min(transition_duration, max(0.05, d / 4.0))))
        fade_out_start = max(0.0, d - td)

        chain = (
            f"[{i}:v]"
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={TARGET_FPS},format=yuv420p"
        )

        # fade in (skip on first clip to avoid repeated fade-from-black if you don’t want it)
        if i != 0:
            chain += f",fade=t=in:st=0:d={td}"

        # fade out (skip on last clip; we will hard-trim final anyway)
        if i != len(seq_durs) - 1:
            chain += f",fade=t=out:st={fade_out_start}:d={td}"

        vout = f"[v{i}]"
        chain += vout
        filter_lines.append(chain)
        v_labels.append(vout)

    # concat all video streams
    concat_inputs = "".join(v_labels)
    filter_lines.append(f"{concat_inputs}concat=n={len(v_labels)}:v=1:a=0[vcat]")

    filter_complex = ";".join(filter_lines)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vcat]",
        "-t", f"{needed_duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT)
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="ignore")[-4000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg library concat failed: {tail}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg library concat timed out")

    return out_path


@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    audio_path = os.path.join(tmpdir, "audio.mp3")
    subs_path = os.path.join(tmpdir, "subs.srt")
    output_path = os.path.join(tmpdir, "output.mp4")

    try:
        # required
        download_file(str(request.audio_url), audio_path)

        audio_duration = get_audio_duration_seconds(audio_path)
        total_duration = max(0.0, audio_duration + VIDEO_TAIL_SECONDS)

        # Determine video source:
        # 1) use provided video_url
        # 2) else build from library_folder
        if request.video_url:
            video_path = os.path.join(tmpdir, "video.mp4")
            download_file(str(request.video_url), video_path)
            use_library = False
        else:
            if not request.library_folder:
                raise HTTPException(
                    status_code=400,
                    detail="video_url is required unless library selection is implemented in the mux service.",
                )
            use_library = True
            # seed random based on job_id for variety
            random.seed(job_id)
            video_path = _build_library_video(
                tmpdir=tmpdir,
                library_folder=str(request.library_folder),
                library_count=int(request.library_count or 10),
                needed_duration=total_duration,
                transition_duration=float(request.transition_duration or 0.35),
            )

        # subtitles?
        has_subs = False
        if request.subtitles_url:
            download_file(str(request.subtitles_url), subs_path)
            try:
                has_subs = os.path.getsize(subs_path) > 0
            except OSError:
                has_subs = False

        # build video filter
        vf = None
        if has_subs:
            style = _subtitle_style(request.subtitle_profile, request.subtitle_font, request.subtitle_font_size)
            # If we used library concat, the video is already normalized; if not, still ok.
            vf = f"subtitles={subs_path}:force_style='{style}'"

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
            "-profile:v", "high",
            "-level", "4.1",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
        ]

        if vf:
            ffmpeg_cmd += ["-vf", vf]

        ffmpeg_cmd.append(output_path)

        proc = subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT,
        )

        stderr_tail = proc.stderr.decode(errors="ignore")[-1500:]
        if stderr_tail:
            print("FFMPEG STDERR (tail):", stderr_tail)

    except subprocess.CalledProcessError as e:
        error_tail = e.stderr.decode(errors="ignore")[-4000:]
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {error_tail}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out")
    finally:
        background_tasks.add_task(shutil.rmtree, tmpdir, True)

    return FileResponse(output_path, media_type="video/mp4", filename=f"final_{job_id}.mp4")

