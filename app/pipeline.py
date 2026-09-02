import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.config import (
    AUDIO_DIR,
    DEFAULT_FPS,
    DEFAULT_KEYFRAME_INTERVAL_SEC,
    DEFAULT_LLM_MODEL,
    DEFAULT_WHISPER_MODEL,
    FRAMES_DIR,
    MAX_FRAMES_PER_VIDEO,
    TRANSCRIPTS_DIR,
    VISION_MODEL,
)
from app.fusion import build_multimodal_context_prompt, fuse_audio_and_visuals
from app.llm import GemmaVideoExplainer
from app.transcription import WhisperTranscriber
from app.video import extract_audio, extract_keyframes, get_video_info
from app.vision import VisionAnalyzer

logger = logging.getLogger(__name__)


class VideoUnderstandingPipeline:
    """End-to-end pipeline for audio transcription and visual video understanding."""

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        vision_model: str = VISION_MODEL,
        whisper_model: Optional[str | Path] = None,
    ):
        self.llm = GemmaVideoExplainer(model_name=llm_model)
        self.vision_analyzer = VisionAnalyzer(model=vision_model)
        self.transcriber = WhisperTranscriber(model_path=whisper_model)

    def process_video(
        self,
        video_path: str | Path,
        target_fps: float = DEFAULT_FPS,
        keyframe_interval: Optional[float] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete video intelligence pipeline:
        1. Video inspection
        2. Audio demuxing
        3. 30 FPS frame sampling
        4. Whisper speech-to-text transcription
        5. Visual frame inspection: sends the last frame of every 30fps (1-second) chunk to the LLM
        6. Multimodal fusion
        7. Gemma 4 deep reasoning ('What is inside the video')
        """
        start_time = time.time()
        video_path = Path(video_path)
        video_id = video_path.stem

        def report_progress(step_name: str, pct: float, message: str):
            if progress_callback:
                progress_callback(step_name, pct, message)
            logger.info(f"[{pct:.0f}%] {step_name}: {message}")

        report_progress("init", 5.0, f"Analyzing video metadata for {video_path.name}...")
        meta = get_video_info(video_path)

        # 1. Extract Audio
        audio_wav = AUDIO_DIR / f"{video_id}.wav"
        has_audio = meta.get("has_audio", False)

        transcript_data = {
            "total_segments": 0,
            "full_text": "",
            "segments": [],
            "files": {},
        }

        if has_audio:
            report_progress("audio_extract", 15.0, "Extracting audio track (16kHz WAV)...")
            if extract_audio(video_path, audio_wav):
                report_progress("transcribe", 25.0, "Transcribing speech via local whisper.cpp (Metal GPU)...")
                try:
                    transcript_data = self.transcriber.transcribe(audio_wav, output_name=video_id)
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    report_progress("transcribe_error", 40.0, f"Transcription note: {e}")
            else:
                report_progress("audio_extract_failed", 25.0, "Could not extract audio or audio stream empty.")
        else:
            report_progress("no_audio", 25.0, "Video does not contain an audio stream.")

        # 2. Extract Keyframes (30 FPS)
        effective_fps = float(target_fps) if target_fps and target_fps > 0 else DEFAULT_FPS
        report_progress("keyframes", 45.0, f"Extracting visual frames at {effective_fps} FPS...")
        video_frames_dir = FRAMES_DIR / video_id
        raw_frames = extract_keyframes(
            video_path,
            video_frames_dir,
            target_fps=effective_fps,
            max_frames=MAX_FRAMES_PER_VIDEO,
            interval_sec=keyframe_interval,
        )

        # 3. Vision Analysis: Send the LAST frame of every 30fps (1-second) window to the LLM
        report_progress("vision", 55.0, f"Analyzing visual keyframes (last frame of each 30 FPS second)...")
        
        def vision_progress_cb(curr: int, total: int, msg: str):
            pct = 55.0 + (curr / max(1, total)) * 22.0
            report_progress("vision", pct, msg)

        vision_data = self.vision_analyzer.analyze_second_keyframes(
            raw_frames,
            fps=effective_fps,
            progress_callback=vision_progress_cb,
        )
        analyzed_frames = vision_data.get("all_frames", [])
        second_keyframes = vision_data.get("second_keyframes", [])

        # 4. Multimodal Fusion
        report_progress("fusion", 78.0, "Fusing audio transcript & visual timeline...")
        timeline_blocks = fuse_audio_and_visuals(
            transcript_data.get("segments", []),
            second_keyframes if second_keyframes else analyzed_frames,
            window_duration_sec=15.0,
        )

        context_prompt = build_multimodal_context_prompt(
            meta,
            timeline_blocks,
            transcript_data.get("full_text", ""),
        )

        # 5. Local Gemma 4 Reasoning
        report_progress("gemma_reasoning", 85.0, f"Generating 'What is inside the video' report using Gemma 4...")
        gemma_result = self.llm.explain_video_contents(context_prompt)

        elapsed = round(time.time() - start_time, 2)
        report_progress("complete", 100.0, f"Completed in {elapsed}s!")

        full_result = {
            "video_id": video_id,
            "metadata": meta,
            "transcription": transcript_data,
            "keyframes": analyzed_frames,
            "second_keyframes": second_keyframes,
            "timeline_blocks": timeline_blocks,
            "context_prompt": context_prompt,
            "gemma_analysis": gemma_result,
            "elapsed_seconds": elapsed,
        }

        # Save final complete output bundle
        output_file = TRANSCRIPTS_DIR / f"{video_id}_full_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(full_result, f, indent=2)

        return full_result
