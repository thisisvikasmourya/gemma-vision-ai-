import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    BASE_DIR,
    DEFAULT_LLM_MODEL,
    FRAMES_DIR,
    TRANSCRIPTS_DIR,
    UPLOADS_DIR,
    VIDEOS_DIR,
)
from app.llm import GemmaVideoExplainer
from app.pipeline import VideoUnderstandingPipeline
from app.video import get_video_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("video-trans")

app = FastAPI(
    title="Video Intelligence & Transcription Platform",
    description="Local Video Understanding powered by Gemma 4 and whisper.cpp Metal",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job state store
jobs_db: Dict[str, Dict[str, Any]] = {}


class ProcessRequest(BaseModel):
    video_path: Optional[str] = None
    video_id: Optional[str] = None
    llm_model: Optional[str] = DEFAULT_LLM_MODEL
    keyframe_interval: Optional[float] = 5.0


class ChatRequest(BaseModel):
    video_id: str
    message: str
    chat_history: Optional[List[Dict[str, str]]] = []


def run_pipeline_task(job_id: str, video_path: Path, llm_model: str, keyframe_interval: float):
    """Background worker executing the processing pipeline."""
    try:
        jobs_db[job_id]["status"] = "processing"

        def progress_cb(step: str, pct: float, msg: str):
            jobs_db[job_id]["step"] = step
            jobs_db[job_id]["progress"] = pct
            jobs_db[job_id]["message"] = msg

        pipeline = VideoUnderstandingPipeline(llm_model=llm_model)
        result = pipeline.process_video(
            video_path=video_path,
            keyframe_interval=keyframe_interval,
            progress_callback=progress_cb,
        )

        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 100.0
        jobs_db[job_id]["message"] = "Processing completed successfully."
        jobs_db[job_id]["result"] = result
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
        jobs_db[job_id]["message"] = f"Failed: {e}"


@app.get("/api/health")
def health_check():
    gemma = GemmaVideoExplainer()
    return {
        "status": "online",
        "ollama_connected": gemma.check_connection(),
        "default_llm": DEFAULT_LLM_MODEL,
    }


@app.get("/api/videos")
def list_available_videos():
    """List all available videos from data/videos and uploads directory."""
    videos = []
    seen = set()

    for folder in [VIDEOS_DIR, UPLOADS_DIR]:
        for ext in ["*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm", "*.m4v"]:
            for v_file in folder.glob(ext):
                if v_file.name in seen:
                    continue
                seen.add(v_file.name)
                try:
                    info = get_video_info(v_file)
                    # Check if already processed
                    analysis_file = TRANSCRIPTS_DIR / f"{v_file.stem}_full_analysis.json"
                    info["is_processed"] = analysis_file.exists()
                    videos.append(info)
                except Exception as e:
                    logger.debug(f"Could not probe {v_file}: {e}")

    return {"videos": videos}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a new video file to uploads/."""
    file_ext = Path(file.filename).suffix
    safe_name = f"{Path(file.filename).stem}_{uuid.uuid4().hex[:6]}{file_ext}"
    dest_path = UPLOADS_DIR / safe_name

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    info = get_video_info(dest_path)
    return {
        "filename": safe_name,
        "original_name": file.filename,
        "filepath": str(dest_path),
        "video_id": dest_path.stem,
        "metadata": info,
    }


@app.post("/api/process")
def start_processing(req: ProcessRequest, bg_tasks: BackgroundTasks):
    """Start video transcription and understanding job."""
    target_path = None
    if req.video_path:
        target_path = Path(req.video_path)
    elif req.video_id:
        # Search in videos and uploads
        for folder in [VIDEOS_DIR, UPLOADS_DIR]:
            candidates = list(folder.glob(f"{req.video_id}.*"))
            if candidates:
                target_path = candidates[0]
                break

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    jobs_db[job_id] = {
        "job_id": job_id,
        "video_id": target_path.stem,
        "video_path": str(target_path),
        "status": "queued",
        "progress": 0.0,
        "step": "queued",
        "message": "Queued for processing...",
        "result": None,
    }

    bg_tasks.add_task(
        run_pipeline_task,
        job_id,
        target_path,
        req.llm_model or DEFAULT_LLM_MODEL,
        req.keyframe_interval or 5.0,
    )

    return {"job_id": job_id, "video_id": target_path.stem, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs_db[job_id]


@app.get("/api/results/{video_id}")
def get_video_results(video_id: str):
    """Fetch previously processed analysis result for a video."""
    analysis_file = TRANSCRIPTS_DIR / f"{video_id}_full_analysis.json"
    if not analysis_file.exists():
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    with open(analysis_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.post("/api/chat")
def chat_with_video(req: ChatRequest):
    """Chat with local Gemma 4 about the video."""
    analysis_file = TRANSCRIPTS_DIR / f"{req.video_id}_full_analysis.json"
    if not analysis_file.exists():
        raise HTTPException(status_code=404, detail="Video analysis must be completed before chatting.")

    with open(analysis_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    context_prompt = data.get("context_prompt", "")
    gemma = GemmaVideoExplainer()
    answer = gemma.chat_query(
        query=req.message,
        context_prompt=context_prompt,
        chat_history=req.chat_history,
    )

    return {"answer": answer}


# Serve raw media assets
@app.get("/media/video/{filename}")
def stream_video(filename: str):
    for folder in [VIDEOS_DIR, UPLOADS_DIR]:
        path = folder / filename
        if path.exists():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Video not found")


@app.get("/media/frames/{video_id}/{filename}")
def stream_frame(video_id: str, filename: str):
    path = FRAMES_DIR / video_id / filename
    if path.exists():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Frame not found")


# Static Frontend mount
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
