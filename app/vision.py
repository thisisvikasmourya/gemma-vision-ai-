import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import requests

from app.config import OLLAMA_BASE_URL, VISION_MODEL
from app.video import format_timestamp

logger = logging.getLogger(__name__)


def encode_image_base64(image_path: str | Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def clean_llm_response(text: str) -> str:
    """Clean up thinking tags or conversational prefixes from LLM response."""
    if not text:
        return ""
    # Remove <think>...</think> block if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    return text


def select_last_frame_per_second(
    frames: List[Dict[str, Any]],
    fps: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Groups 30 FPS frames into 1-second chunks (30 frames per 1 second)
    and selects the LAST frame of each 30-frame window.
    
    For example:
    - Second 1 (frames 1 to 30): selects frame 30
    - Second 2 (frames 31 to 60): selects frame 60
    - Second 3 (frames 61 to 90): selects frame 90
    - ...
    """
    if not frames:
        return []

    chunk_size = max(1, int(round(fps)))
    selected_keyframes = []

    total_frames = len(frames)
    for start_idx in range(0, total_frames, chunk_size):
        chunk = frames[start_idx : start_idx + chunk_size]
        if not chunk:
            continue
        
        # Pick the LAST frame of this 30-frame (1-second) chunk
        last_frame = chunk[-1]
        second_num = (start_idx // chunk_size) + 1
        
        # Clone and annotate
        kf = dict(last_frame)
        kf["is_second_keyframe"] = True
        kf["second_number"] = second_num
        kf["chunk_start_idx"] = start_idx
        kf["chunk_end_idx"] = start_idx + len(chunk) - 1
        kf["frames_in_second"] = len(chunk)
        selected_keyframes.append(kf)

    logger.info(
        f"Selected {len(selected_keyframes)} last-frames (1 per second) from {total_frames} total frames at {fps} FPS."
    )
    return selected_keyframes


class VisionAnalyzer:
    """
    Analyzes video keyframes using local Vision Language Models (VLM) via Ollama.
    Auto-detects available vision models (e.g. qwen3.5:9b-mlx, gemma-vision, etc.).
    """

    def __init__(self, ollama_url: str = OLLAMA_BASE_URL, model: str = VISION_MODEL):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._resolved_model: Optional[str] = None

    def get_effective_vision_model(self) -> str:
        """
        Detects the best available vision model in Ollama.
        If self.model is vision-capable, uses it. Otherwise finds an installed vision model.
        """
        if self._resolved_model:
            return self._resolved_model

        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if res.status_code == 200:
                models = res.json().get("models", [])
                
                # Check if configured model has vision capability
                for m in models:
                    m_name = m.get("name", "")
                    caps = m.get("capabilities", []) or []
                    if m_name == self.model and "vision" in caps:
                        self._resolved_model = self.model
                        return self._resolved_model

                # If configured model is in tags but without explicit vision caps, try checking if it's named as vision
                for m in models:
                    m_name = m.get("name", "")
                    if m_name == self.model and ("vision" in m_name.lower() or "vl" in m_name.lower()):
                        self._resolved_model = self.model
                        return self._resolved_model

                # Search for any installed model with vision capability
                for m in models:
                    caps = m.get("capabilities", []) or []
                    if "vision" in caps:
                        self._resolved_model = m.get("name")
                        logger.info(f"Auto-selected vision model: {self._resolved_model}")
                        return self._resolved_model

                # Fallback: check models by common vision names
                for m in models:
                    m_name = m.get("name", "").lower()
                    if any(k in m_name for k in ["qwen3.5", "qwen2.5-vl", "llava", "vision", "paligemma", "minicpm"]):
                        self._resolved_model = m.get("name")
                        logger.info(f"Auto-selected vision model by name: {self._resolved_model}")
                        return self._resolved_model

        except Exception as e:
            logger.debug(f"Could not query Ollama tags: {e}")

        self._resolved_model = self.model
        return self._resolved_model

    def analyze_frame(self, frame_path: str | Path, timestamp: float) -> Dict[str, Any]:
        """
        Analyze an individual 1-second keyframe image (the last frame of the 30fps window).
        Sends base64 image to local VLM to describe what is happening inside that frame.
        """
        frame_path = Path(frame_path)
        if not frame_path.exists():
            return {
                "timestamp": timestamp,
                "timestamp_formatted": format_timestamp(timestamp),
                "description": "Frame not found",
                "frame_path": str(frame_path),
                "is_second_keyframe": True,
            }

        active_model = self.get_effective_vision_model()
        base64_img = encode_image_base64(frame_path)

        prompt = (
            "Analyze what is inside this video frame in 1-2 clear, concise sentences. "
            "Describe the key visual elements: people, actions, objects, setting, slides, diagrams, or on-screen text."
        )

        description = ""
        try:
            # 1. Try /api/chat endpoint
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64_img],
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 180,
                    },
                },
                timeout=45,
            )

            if response.status_code == 200:
                res_json = response.json()
                msg = res_json.get("message", {})
                raw_text = msg.get("content", "")
                description = clean_llm_response(raw_text)

            # Fallback to /api/generate if /api/chat failed or returned empty
            if not description:
                gen_res = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": active_model,
                        "prompt": prompt,
                        "images": [base64_img],
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 180,
                        },
                    },
                    timeout=45,
                )
                if gen_res.status_code == 200:
                    description = clean_llm_response(gen_res.json().get("response", ""))

        except Exception as e:
            logger.warning(f"VLM call failed for frame {frame_path.name} ({active_model}): {e}")

        if not description:
            description = f"Visual scene at {format_timestamp(timestamp)}"

        return {
            "timestamp": timestamp,
            "timestamp_formatted": format_timestamp(timestamp),
            "description": description,
            "frame_path": str(frame_path),
            "filename": frame_path.name,
            "is_second_keyframe": True,
            "model_used": active_model,
        }

    def analyze_second_keyframes(
        self,
        raw_frames: List[Dict[str, Any]],
        fps: float = 30.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        1. Takes full 30 FPS frame extraction.
        2. Selects the last frame of every 30-frame (1-second) window.
        3. Sends each 1-second last frame to the local VLM to understand what is inside that frame.
        4. Enriches the entire frame stream with the visual descriptions.
        """
        if not raw_frames:
            return {"all_frames": [], "second_keyframes": []}

        # Select the last frame for each 1-second segment
        second_keyframes = select_last_frame_per_second(raw_frames, fps=fps)
        total_seconds = len(second_keyframes)
        
        logger.info(f"Analyzing {total_seconds} 1-second last-frames using VLM ({self.get_effective_vision_model()})...")

        # Analyze each 1-second keyframe
        descriptions_by_second: Dict[int, Dict[str, Any]] = {}
        analyzed_keyframes = []

        for idx, kf in enumerate(second_keyframes):
            sec_num = kf.get("second_number", idx + 1)
            fpath = kf.get("filepath", "")
            ts = kf.get("timestamp", 0.0)

            if progress_callback:
                progress_callback(
                    idx + 1,
                    total_seconds,
                    f"Analyzing 1s frame {idx + 1}/{total_seconds} ({kf.get('timestamp_formatted', '')} - frame {kf.get('filename')})...",
                )

            analysis = self.analyze_frame(fpath, ts)
            
            # Merge metadata
            combined = dict(kf)
            combined.update(analysis)
            analyzed_keyframes.append(combined)
            descriptions_by_second[sec_num] = combined

        # Enrich all 30 FPS frames with their corresponding 1-second visual context
        chunk_size = max(1, int(round(fps)))
        enriched_all_frames = []

        for idx, frame in enumerate(raw_frames):
            sec_num = (idx // chunk_size) + 1
            sec_data = descriptions_by_second.get(sec_num, {})
            is_last = (idx % chunk_size == (chunk_size - 1)) or (idx == len(raw_frames) - 1)

            enriched_frame = {
                "index": frame.get("index", idx + 1),
                "timestamp": frame.get("timestamp", round(idx / fps, 4)),
                "timestamp_formatted": frame.get("timestamp_formatted", format_timestamp(idx / fps)),
                "filepath": frame.get("filepath", ""),
                "filename": frame.get("filename", ""),
                "second_number": sec_num,
                "is_second_keyframe": is_last,
                "description": sec_data.get("description", f"Visual frame at {frame.get('timestamp_formatted', '')}"),
            }
            enriched_all_frames.append(enriched_frame)

        return {
            "all_frames": enriched_all_frames,
            "second_keyframes": analyzed_keyframes,
        }

