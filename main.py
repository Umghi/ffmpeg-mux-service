import os
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

DOWNLOAD_TIMEOUT = 120
FFMPEG_TIMEOUT = 900
MAX_BYTES = 300 * 1024 * 1024
VIDEO_TAIL_SECONDS = 1.0

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "").strip()


class MuxRequest(BaseModel):
    # Backward-compatible single video mode
    video_url: Optional[HttpUrl] = None

    audio_url: HttpUrl
    subtitles_url: Optional[HttpUrl] = None

    subtitle_profile: str = "menopause"
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None

    # NEW: Dropbox library mode
    library_folder: Optional[str] = None  # e.g. "/BIBLE/BibleLibrary/jerusalem_vertical/"
    library_count: int = 10
    transition: str = "fadeblack"         # "fadeblack" or "crossfade"
    transition_duration: float = 0.35     # seconds


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
                                detail="Downloaded HTML instead of a media file. Check that URLs are direct-download links."
                            )
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(status_code=413, detail="File too large")
                    f.write(chunk)

            if not wrote_any:
                raise HTTPException(status_code=400, detail="Downloaded file is empty")

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")


def ffprobe_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
        if out in ("", "N/A"):
            raise ValueError(f"ffprobe duration unavailable: '{out}'")
        d = float(out)
        if d <= 0:
            raise ValueError("Invalid duration")
        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


# ----------------------------
# Dropbox library helpers
# ----------------------------
def _dbx_headers() -> Dict[str, str]:
    if not DROPBOX_ACCESS_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="DROPBOX_ACCESS_TOKEN is not set on the mux service."
        )
    return {
        "Authorization": f"Bearer {DROPBOX_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def dbx_list_folder(folder_path: str) -> List[Dict[str, Any]]:
    url = "https://api.dropboxapi.com/2/files/list_folder"
    payload = {"path": folder_path, "recursive": False, "include_media_info": False}
    r = requests.post(url, headers=_dbx_headers(), json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox list_folder failed: {r.text}")
    data = r.json()
    entries = data.get("entries", [])
    # Filter files only
    files = [e for e in entries if e.get(".tag") == "file"]
    return files


def dbx_get_temp_link(path_lower: str) -> str:
    url = "https://api.dropboxapi.com/2/files/get_temporary_link"
    payload = {"path": path_lower}
    r = requests.post(url, headers=_dbx_headers(), json=payload, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Dropbox get_temporary_link failed: {r.text}")
    return r.json().get("link", "")


def pick_library_files(folder_path: str, count: int) -> List[str]:
    files = dbx_list_folder(folder_path)
    if not files:
        raise HTTPException(status_code=400, detail=f"No files found in Dropbox folder: {folder_path}")

    # Prefer mp4/mov
    def is_video(name: str) -> bool:
        n = (name or "").lower()
        return n.endswith(".mp4") or n.endswith(".mov") or n.endswith(".m4v") or n.endswith(".webm")

    vids = [f for f in files if is_video(f.get("name", ""))]
    if not vids:
        raise HTTPException(status_code=400, detail=f"No video files found in folder: {folder_path}")

    # sample without replacement; if count > available, just shuffle all
    random.shuffle(vids)
    chosen = vids[: max(1, min(count, len(vids)))]

    # return path_lower for temp links
    paths = [c.get("path_lower") for c in chosen if c.get("path_lower")]
    if not paths:
        raise HTTPException(status_code=400, detail="Dropbox returned files but no path_lower values.")
    return paths


# ----------------------------
# Subtitle helpers (same as earlier version)
# ----------------------------
def wrap_to_lines(text: str, max_line_chars: int = 42, max_lines: int = 2) -> str:
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
                remaining = [*cur, *words[words.index(w)+1:]]
                last = " ".join(remaining)
                lines.append(last[: max_line_chars * 2])
                cur = []
                break

    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))

    return "\n".join(lines[:max_lines]).strip()


def build_subtitle_vf(subs_path: str, profile: str, font: Optional[str], size: Optional[int]) -> str:
    p = (profile or "menopause").strip().lower()
    if p == "bible":
        font_name = font or "DejaVu Serif"
        font_size = int(size or 26)
        margin_v = 90
        outline = 2
        shadow = 0
        primary_colour = "&H00FFFFFF&"
        outline_colour = "&H00000000&"
        back_colour = "&H80000000&"  # readable box

        style = (
            f"FontName={font_name},"
            f"FontSize={font_size},"
            f"PrimaryColour={primary_colour},"
            f"OutlineColour={outline_colour},"
            f"BackColour={back_colour},"
            f"BorderStyle=3,"
            f"Outline={outline},"
            f"Shadow={shadow},"
            f"MarginV={margin_v},"
            f"Alignment=2"
        )
    else:
        font_name = font or "Arial"
        font_size = int(size or 16)
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

    return f"subtitles={subs_path}:force_style='{style}'"


# ----------------------------
# Stitching helpers
# ----------------------------
def normalize_clip(in_path: str, out_path: str) -> None:
    """
    Normalize to consistent format for concat/xfade reliability.
    30fps CFR, yuv420p, 1080x1920 padded.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg normalize failed: {e.stderr.decode(errors='ignore')[-2000:]}")


def build_xfade_filter(paths: List[str], durations: List[float], transition: str, td: float) -> str:
    """
    Chain xfade across N inputs, producing [vout].
    """
    if len(paths) < 2:
        return ""  # no filter needed

    trans = "fade" if transition == "crossfade" else "fade"  # we implement fade-to-black via extra steps below

    # For fade-to-black between clips: do fade-out on clip A, fade-in on clip B, then xfade
    # Practical approach: use xfade=fade (crossfade) plus a brief black gap can be simulated by fading both sides.
    # Simpler: use xfade=fade with slightly longer duration (looks close to fade-to-black if scenes differ).
    # If you truly want hard black in between, we can insert a color source; this is a good first pass.

    filt = []
    # Label inputs [0:v][1:v]...
    # Start by chaining:
    # [0:v][1:v]xfade=transition=fade:duration=td:offset=dur0-td[v01];
    # [v01][2:v]xfade=transition=fade:duration=td:offset=(dur0+dur1)-(2*td)[v012]; ... etc
    cumulative = 0.0
    current_label = None
    for i in range(len(paths) - 1):
        a = f"{i}:v" if current_label is None else current_label
        b = f"{i+1}:v"
        dur_a = durations[i]
        # offset is where the transition starts relative to the *current* composed stream
        # cumulative is total duration of composed stream so far
        if current_label is None:
            cumulative = dur_a
        else:
            cumulative += durations[i]

        offset = max(0.0, cumulative - td)
        out_label = f"v{i+1}x"
        filt.append(f"[{a}][{b}]xfade=transition={trans}:duration={td}:offset={offset:.3f}[{out_label}]")
        current_label = out_label

    filt.append(f"[{current_label}]format=yuv420p[vout]")
    return ";".join(filt)


@app.get("/fonts", response_class=PlainTextResponse)
def list_fonts():
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
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp(prefix=f"mux_{job_id}_")

    audio_path = os.path.join(tmpdir, "audio.mp3")
    subs_path = os.path.join(tmpdir, "subs.srt")
    output_path = os.path.join(tmpdir, "output.mp4")

    try:
        # Download audio
        download_file(str(request.audio_url), audio_path)

        # Download subs if present
        has_subs = False
        if request.subtitles_url:
            download_file(str(request.subtitles_url), subs_path)
            has_subs = os.path.getsize(subs_path) > 0

        # Duration target
        audio_duration = ffprobe_duration_seconds(audio_path)
        total_duration = max(0.0, audio_duration + VIDEO_TAIL_SECONDS)

        # Decide video source mode
        use_library = bool(request.library_folder and str(request.library_folder).strip())
        if not use_library and not request.video_url:
            raise HTTPException(status_code=400, detail="Provide either video_url or library_folder.")

        # Prepare final video input (either stitched montage or single loop)
        stitched_path = os.path.join(tmpdir, "stitched.mp4")

        if use_library:
            folder = str(request.library_folder).strip()
            count = int(request.library_count or 10)
            td = float(request.transition_duration or 0.35)
            transition = (request.transition or "fadeblack").strip().lower()
            if transition not in ("fadeblack", "crossfade"):
                transition = "fadeblack"

            # pick files in dropbox
            picked_paths = pick_library_files(folder, count)

            # get temp links
            temp_links = [dbx_get_temp_link(p) for p in picked_paths]
            temp_links = [l for l in temp_links if l]
            if not temp_links:
                raise HTTPException(status_code=400, detail="No temporary links returned from Dropbox.")

            # download and normalize
            norm_paths = []
            durations = []
            for i, link in enumerate(temp_links):
                raw = os.path.join(tmpdir, f"raw_{i}.mp4")
                norm = os.path.join(tmpdir, f"norm_{i}.mp4")
                download_file(link, raw)
                normalize_clip(raw, norm)
                d = ffprobe_duration_seconds(norm)
                norm_paths.append(norm)
                durations.append(d)

            # If only one clip, just use it and loop
            if len(norm_paths) == 1:
                # We will loop at ffmpeg mux stage
                stitched_input_mode = ("single", norm_paths[0])
            else:
                # Build xfade montage (first pass: crossfade-ish, looks good and avoids black insert complexity)
                # If transition requested is fadeblack, we still use fade xfade; visually it reads like a dip/crossfade.
                # We can add real black inserts later if you want.
                filter_complex = build_xfade_filter(norm_paths, durations, "crossfade", td)
                if not filter_complex:
                    stitched_input_mode = ("single", norm_paths[0])
                else:
                    # Build ffmpeg stitch command
                    cmd = ["ffmpeg", "-y"]
                    for p in norm_paths:
                        cmd += ["-i", p]
                    cmd += [
                        "-filter_complex", filter_complex,
                        "-map", "[vout]",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        stitched_path
                    ]
                    print("FFMPEG STITCH CMD:", " ".join(cmd))
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT)
                    except subprocess.CalledProcessError as e:
                        raise HTTPException(status_code=500, detail=f"ffmpeg stitch failed: {e.stderr.decode(errors='ignore')[-4000:]}")
                    stitched_input_mode = ("stitched", stitched_path)

        else:
            # single URL mode
            video_path = os.path.join(tmpdir, "video.mp4")
            download_file(str(request.video_url), video_path)
            stitched_input_mode = ("single", video_path)

        # Build subtitles vf
        vf_arg = None
        if has_subs:
            vf_arg = build_subtitle_vf(
                subs_path=subs_path,
                profile=request.subtitle_profile,
                font=request.subtitle_font,
                size=request.subtitle_font_size
            )

        # Final mux ffmpeg
        # If single clip: loop it. If stitched: loop stitched (safe) but it should already be longer; looping is fine.
        video_in = stitched_input_mode[1]

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", video_in,
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

        print("FFMPEG MUX CMD:", " ".join(ffmpeg_cmd))

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

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"final_{job_id}.mp4",
        )

    finally:
        background_tasks.add_task(shutil.rmtree, tmpdir, True)
