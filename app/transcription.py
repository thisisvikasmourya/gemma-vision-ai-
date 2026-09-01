import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import (
    DEFAULT_WHISPER_MODEL,
    TRANSCRIPTS_DIR,
    WHISPER_BIN,
    WHISPER_BIN_FALLBACKS,
)
from app.video import format_timestamp

logger = logging.getLogger(__name__)


def find_whisper_binary() -> Optional[Path]:
    """Find available whisper-cli binary in whisper.cpp build directory."""
    if WHISPER_BIN.exists():
        return WHISPER_BIN
    for fb in WHISPER_BIN_FALLBACKS:
        if fb.exists():
            return fb
    return None


class WhisperTranscriber:
    """Wrapper around whisper.cpp Metal-accelerated transcription engine."""

    def __init__(self, model_path: Optional[str | Path] = None, language: str = "en"):
        self.model_path = Path(model_path) if model_path else DEFAULT_WHISPER_MODEL
        self.language = language
        self.binary_path = find_whisper_binary()

        if not self.binary_path or not self.binary_path.exists():
            raise FileNotFoundError(
                f"whisper-cli binary not found. Please build whisper.cpp first."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found at {self.model_path}")

    def transcribe(
        self,
        audio_wav_path: str | Path,
        output_name: Optional[str] = None,
        threads: int = 6,
    ) -> Dict[str, Any]:
        """
        Transcribe a 16kHz mono WAV file into timestamped segments and text formats.
        Generates JSON, SRT, VTT, and plain text transcripts.
        """
        audio_wav_path = Path(audio_wav_path)
        if not audio_wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_wav_path}")

        base_name = output_name or audio_wav_path.stem
        output_prefix = TRANSCRIPTS_DIR / base_name

        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-f", str(audio_wav_path),
            "-l", self.language,
            "-t", str(threads),
            "-oj",   # JSON output
            "-osrt", # SRT subtitle output
            "-ovtt", # VTT subtitle output
            "-otxt", # Plain text output
            "-of", str(output_prefix),
            "-pp",   # print progress
        ]

        logger.info(f"Running Whisper transcription on {audio_wav_path.name}...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Whisper transcription failed: {e.stderr}")
            raise RuntimeError(f"Whisper transcription failed: {e.stderr}")

        # Parse generated JSON
        json_file = Path(f"{output_prefix}.json")
        segments = []
        full_text = ""

        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                # Parse whisper.cpp JSON schema
                raw_segments = raw_data.get("transcription", [])
                for idx, seg in enumerate(raw_segments):
                    t_offsets = seg.get("offsets", {})
                    # Timestamps in whisper.cpp json are in milliseconds or format
                    start_ms = t_offsets.get("from", 0)
                    end_ms = t_offsets.get("to", 0)
                    start_sec = round(start_ms / 1000.0, 3)
                    end_sec = round(end_ms / 1000.0, 3)
                    text = seg.get("text", "").strip()

                    if text:
                        segments.append({
                            "id": idx,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "start_timestamp": format_timestamp(start_sec),
                            "end_timestamp": format_timestamp(end_sec),
                            "text": text,
                        })

                full_text = " ".join([s["text"] for s in segments])
            except Exception as e:
                logger.warning(f"Error parsing whisper JSON: {e}. Falling back to text file.")

        # Fallback if json wasn't parsed properly
        txt_file = Path(f"{output_prefix}.txt")
        if not full_text and txt_file.exists():
            full_text = txt_file.read_text(encoding="utf-8").strip()

        srt_file = Path(f"{output_prefix}.srt")
        vtt_file = Path(f"{output_prefix}.vtt")

        payload = {
            "audio_filename": audio_wav_path.name,
            "total_segments": len(segments),
            "full_text": full_text,
            "segments": segments,
            "files": {
                "json": str(json_file) if json_file.exists() else None,
                "srt": str(srt_file) if srt_file.exists() else None,
                "vtt": str(vtt_file) if vtt_file.exists() else None,
                "txt": str(txt_file) if txt_file.exists() else None,
            },
        }

        # Save clean normalized transcript JSON
        normalized_json_path = TRANSCRIPTS_DIR / f"{base_name}_normalized.json"
        with open(normalized_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Transcription completed with {len(segments)} segments.")
        return payload
