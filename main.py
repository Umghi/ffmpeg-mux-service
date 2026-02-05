import os
import uuid
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, HttpUrl

app = FastAPI()

# ----------------------------
# Config
# ----------------------------
DOWNLOAD_TIMEOUT = 120          # seconds per file download
FFMPEG_TIMEOUT = 900            # seconds (15 minutes)
MAX_BYTES = 300 * 1024 * 1024   # 300MB safety limit

# ----------------------------
# Request models
# ----------------------------
class SrtRequest(BaseModel):
    # pass the parsed STT JSON (as a dict)
    stt: Dict[str, Any]
    # optional formatting controls
    words_per_caption: int = 6
    max_line_chars: int = 42


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
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
        duration = float(output)
        if duration <= 0:
            raise ValueError("Invalid duration")
        return duration
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {e}")


def srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hh = ms // 3600000
    ms -= hh * 3600000
    mm = ms // 60000
    ms -= mm * 60000
    ss = ms // 1000
    ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def normalize_words(stt: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    ElevenLabs STT often includes entries of type 'word' and type 'spacing'.
    We only want type == 'word' (and anything else that has start/end).
    """
    words = stt.get("words") or stt.get("data", {}).get("words") or []
    out = []
    for w in words:
        if not isinstance(w, dict):
            continue
        wtype = w.get("type")
        if wtype == "spacing":
            continue
        if "start" in w and "end" in w and "text" in w:
            # keep actual words; ignore pure whitespace tokens
            txt = w.get("text", "")
            if txt.strip() == "":
                continue
            out.append({"text": txt, "start": float(w["start"]), "end": float(w["end"])})
    return out


def build_srt_from_words(
    words: List[Dict[str, Any]],
    words_per_caption: int = 6,
    max_line_chars: int = 42
) -> str:
    """
    Simple, robust captioning:
    - groups every N words into a caption
    - uses start time of first word and end time of last word
    - wraps into up to 2 lines based on max_line_chars
    """
    if not words:
        return ""

    captions = []
    idx = 1
    for i in range(0, len(words), words_per_caption):
        chunk = words[i:i + words_per_caption]
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        text = " ".join(w["text"] for w in chunk).strip()

        # wrap into up to 2 lines
        if len(text) > max_line_chars:
            # find a split near the middle
            mid = len(text) // 2
            split = text.rfind(" ", 0, mid)
            if split == -1:
                split = text.find(" ", mid)
            if split != -1:
                line1 = text[:split].strip()
                line2 = text[split + 1:].strip()
                text = f"{line1}\n{line2}"

        captions.append(
            f"{idx}\n"
            f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n"
            f"{text}\n"
        )
        idx += 1

    return "\n".join(captions).strip() + "\n"


# ----------------------------
# Endpoints
# ----------------------------
@app.post("/srt", response_class=PlainTextResponse)
def srt(req: SrtRequest):
    try:
        words = normalize_words(req.stt)
        srt_text = build_srt_from_words(
            words,
            words_per_caption=max(2, int(req.words_per_caption)),
            max_line_chars=max(20, int(req.max_line_chars)),
        )
        if not srt_text.strip():
            raise HTTPException(status_code=400, detail="No words found in STT payload")
        return srt_text
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SRT build failed: {e}")


@app.post("/mux")
def mux(request: MuxRequest, background_tasks: BackgroundTasks):
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
        have_subs = True
    else:
        have_subs = False

    # Probe audio duration
    audio_duration = get_audio_duration_seconds(audio_path)

    # Video filter
    # Note: subtitles filter uses libass; should work with .srt
    vf = []
    if have_subs:
        # Basic, readable style (you can tweak later)
        vf.append(f"subtitles={subs_path}")

    vf_arg = ",".join(vf) if vf else None

    # ffmpeg command:
    # - loop video infinitely
    # - mix audio: (optional) bed track on input 0 audio at low volume + generated audio on input 1
    #   (your current video likely has no useful audio, but we keep your pattern)
    # - stop output exactly at audio length
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", video_path,
        "-i", audio_path,
    ]

    if vf_arg:
        ffmpeg_cmd += ["-vf", vf_arg]

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
    finally:
        # Cleanup AFTER response is sent
        background_tasks.add_task(shutil.rmtree, tmpdir, True)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"final_{job_id}.mp4"
    )
