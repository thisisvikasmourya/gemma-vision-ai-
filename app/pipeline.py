import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.config import (
    AUDIO_DIR,
    DEFAULT_KEYFRAME_INTERVAL_SEC,
    DEFAULT_LLM_MODEL,
    DEFAULT_WHISPER_MODEL,
    FRAMES_DIR,
    TRANSCRIPTS_DIR,
)
from app.fusion import build_multimodal_context_prompt, fuse_audio_and_visuals
from app.llm import GemmaVideoExplainer
from app.transcription import WhisperTranscriber
from app.video import extract_audio, extract_keyframes, get_video_info

logger = logging.getLogger(__name__)


class VideoUnderstandingPipeline:
    """End-to-end pipeline for audio transcription and visual video understanding."""

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        whisper_model: Optional[str | Path] = None,
    ):
        self.llm = GemmaVideoExplainer(model_name=llm_model)
        self.transcriber = WhisperTranscriber(model_path=whisper_model)

    def process_video(
        self,
        video_path: str | Path,
        keyframe_interval: float = DEFAULT_KEYFRAME_INTERVAL_SEC,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete video intelligence pipeline:
        1. Video inspection
        2. Audio demuxing
        3. Keyframe sampling
        4. Whisper speech-to-text transcription
        5. Visual frame inspection
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
                report_progress("transcribe", 30.0, "Transcribing speech via local whisper.cpp (Metal GPU)...")
                try:
                    transcript_data = self.transcriber.transcribe(audio_wav, output_name=video_id)
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    report_progress("transcribe_error", 45.0, f"Transcription note: {e}")
            else:
                report_progress("audio_extract_failed", 30.0, "Could not extract audio or audio stream empty.")
        else:
            report_progress("no_audio", 30.0, "Video does not contain an audio stream.")

        # 2. Extract Keyframes
        report_progress("keyframes", 50.0, "Extracting visual keyframes & scenes...")
        video_frames_dir = FRAMES_DIR / video_id
        raw_frames = extract_keyframes(
            video_path,
            video_frames_dir,
            interval_sec=keyframe_interval,
            max_frames=120,
        )

        # 3. Vision Analysis
        report_progress("vision", 65.0, f"Processing {len(raw_frames)} visual frames...")
        # Prepare structured frame entries
        analyzed_frames = []
        for f in raw_frames:
            analyzed_frames.append({
                "timestamp": f["timestamp"],
                "timestamp_formatted": f["timestamp_formatted"],
                "description": f"Visual frame at {f['timestamp_formatted']}",
                "frame_path": f["filepath"],
                "filename": f["filename"],
            })

        # 4. Multimodal Fusion
        report_progress("fusion", 75.0, "Fusing audio transcript & visual timeline...")
        timeline_blocks = fuse_audio_and_visuals(
            transcript_data.get("segments", []),
            analyzed_frames,
            window_duration_sec=20.0,
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
