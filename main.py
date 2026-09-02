from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
import yt_dlp
import os
import uuid
import subprocess
import re
from typing import Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Memuat model AI Whisper (tiny) ke memori server...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Menyimpan metadata file sementara (internal filename -> download name)
file_registry: Dict[str, str] = {}


class TikTokRequest(BaseModel):
    url: str


def sanitize_username(name: str) -> str:
    """Bersihkan username agar aman dipakai sebagai nama file."""
    if not name:
        return "unknown"
    name = name.lstrip("@").strip()
    # Hanya izinkan huruf, angka, underscore, dan strip
    name = re.sub(r"[^\w\-]", "", name)
    return name[:40] if name else "unknown"


@app.post("/api/process")
def process_tiktok(data: TikTokRequest):
    file_id = str(uuid.uuid4())[:8]

    ydl_opts = {
        "format": "best",
        "outtmpl": f"video_{file_id}.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(data.url, download=True)
            ext = info.get("ext", "mp4")
            actual_video_file = f"video_{file_id}.{ext}"

            # Ambil username dari beberapa kemungkinan field
            uploader = (
                info.get("uploader")
                or info.get("creator")
                or info.get("uploader_id")
                or info.get("channel")
                or "unknown"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal ekstrak TikTok: {str(e)}")

    # Buat nama file yang rapi
    safe_username = sanitize_username(uploader)
    download_filename = f"TickText_@{safe_username}.mp4"

    # Simpan ke registry supaya endpoint download tahu nama yang diinginkan
    file_registry[actual_video_file] = download_filename

    # Ekstrak audio untuk Whisper
    audio_file = f"audio_{file_id}.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", actual_video_file, "-q:a", "0", "-map", "a", audio_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass

    transcript_results = []
    if os.path.exists(audio_file):
        try:
            segments, _ = model.transcribe(audio_file, beam_size=5)
            for segment in segments:
                transcript_results.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                })
        except Exception:
            pass
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)

    return {
        "status": "success",
        "download_url": f"http://127.0.0.1:8000/api/download/{actual_video_file}",
        "filename": download_filename,          # <-- frontend bisa pakai ini
        "transcript": transcript_results
    }


@app.get("/api/download/{filename}")
def download_file(filename: str, background_tasks: BackgroundTasks):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File video tidak ditemukan.")

    # Ambil nama file yang cantik, fallback kalau tidak ada
    download_name = file_registry.get(filename, "TickText_video.mp4")

    # Hapus file setelah selesai dikirim ke user
    def cleanup(path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
            file_registry.pop(path, None)
        except Exception:
            pass

    background_tasks.add_task(cleanup, filename)

    return FileResponse(
        path=filename,
        media_type="video/mp4",
        filename=download_name,          # ini yang muncul saat user Save As
    )