import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from app.config import OLLAMA_BASE_URL, VISION_MODEL
from app.video import format_timestamp

logger = logging.getLogger(__name__)


def encode_image_base64(image_path: str | Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class VisionAnalyzer:
    """Analyzes extracted video keyframes to capture visual scenes and on-screen content."""

    def __init__(self, ollama_url: str = OLLAMA_BASE_URL, model: str = VISION_MODEL):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    def analyze_frame(self, frame_path: str | Path, timestamp: float) -> Dict[str, Any]:
        """
        Analyze an individual keyframe image.
        Uses Ollama VLM to generate a visual scene description and detect on-screen text.
        """
        frame_path = Path(frame_path)
        if not frame_path.exists():
            return {
                "timestamp": timestamp,
                "timestamp_formatted": format_timestamp(timestamp),
                "description": "Frame not found",
                "frame_path": str(frame_path),
            }

        base64_img = encode_image_base64(frame_path)
        prompt = (
            "Describe what is happening in this video frame in 1-2 concise sentences. "
            "Note key visual elements: people, actions, setting, presentation slides, diagrams, or on-screen text."
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [base64_img],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 120,
                    },
                },
                timeout=30,
            )
            if response.status_code == 200:
                res_data = response.json()
                description = res_data.get("response", "").strip()
            else:
                # If model doesn't support direct vision, note the keyframe timestamp
                description = f"Key visual frame at {format_timestamp(timestamp)}"
        except Exception as e:
            logger.debug(f"Direct VLM call skipped for frame {frame_path.name}: {e}")
            description = f"Visual scene at {format_timestamp(timestamp)}"

        return {
            "timestamp": timestamp,
            "timestamp_formatted": format_timestamp(timestamp),
            "description": description,
            "frame_path": str(frame_path),
            "filename": frame_path.name,
        }

    def analyze_keyframes_batch(
        self,
        frames: List[Dict[str, Any]],
        sample_step: int = 1,
        progress_callback: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze a sequence of extracted keyframes with optional downsampling for long videos.
        """
        results = []
        selected_frames = frames[::sample_step]
        total = len(selected_frames)

        for i, f in enumerate(selected_frames):
            ts = f.get("timestamp", 0.0)
            fpath = f.get("filepath", "")
            analysis = self.analyze_frame(fpath, ts)
            results.append(analysis)

            if progress_callback:
                progress_callback(i + 1, total)

        return results
