import json
import logging
from typing import Any, Dict, Generator, List, Optional
import requests

from app.config import DEFAULT_LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class GemmaVideoExplainer:
    """Uses local Gemma 4 12B model via Ollama to analyze and explain video contents."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model_name: str = DEFAULT_LLM_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model_name

    def check_connection(self) -> bool:
        """Verify Ollama server is running and reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def explain_video_contents(self, context_prompt: str) -> Dict[str, Any]:
        """
        Synthesizes multimodal timeline and audio transcripts into a detailed
        'What is Inside the Video' report.
        """
        system_instruction = (
            "You are an expert Video Understanding & Intelligence AI. "
            "Your job is to examine the provided video metadata, timestamped visual descriptions, and audio transcription, "
            "then produce a comprehensive, structured breakdown explaining exactly what is inside this video."
        )

        user_prompt = f"""Based on the following timestamped video data (visual scenes + spoken audio dialogue):

{context_prompt}

Please provide a detailed video understanding analysis in clear Markdown with the following structured sections:

### 1. 📌 Executive Summary
What is this video about? Provide a clear 2-3 paragraph overview describing the core subject, setting, and purpose of the video.

### 2. 🎬 Scene-by-Scene Timeline Breakdown
Break down the video into key timestamped segments. For each segment, provide:
- **Timestamp range** (e.g. `[00:00 - 02:30]`)
- **Scene Title**
- **What is happening** (combining what is seen visually and what is being said)

### 3. 🔍 Key Visual Elements & On-Screen Content
- What people, settings, objects, diagrams, or presentations appear?
- Any notable on-screen text, code, or visual cues.

### 4. 🎙️ Key Audio Topics & Spoken Highlights
- Core themes, discussions, or narration highlights from the speaker(s).

### 5. 💡 Key Takeaways & Summary Points
- 3 to 5 bullet points summarizing the most important takeaways from this video.
"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{system_instruction}\n\n{user_prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                    },
                },
                timeout=180,
            )

            if response.status_code == 200:
                result_text = response.json().get("response", "").strip()
                return {
                    "status": "success",
                    "model": self.model,
                    "report_markdown": result_text,
                }
            else:
                return {
                    "status": "error",
                    "error": f"Ollama returned HTTP {response.status_code}: {response.text}",
                    "report_markdown": "Unable to generate AI analysis from local Ollama model.",
                }
        except Exception as e:
            logger.error(f"Error invoking Gemma model: {e}")
            return {
                "status": "error",
                "error": str(e),
                "report_markdown": f"Error connecting to local Ollama model ({self.model}): {e}",
            }

    def chat_query(
        self,
        query: str,
        context_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Answer questions about the video using timestamp references.
        """
        history_text = ""
        if chat_history:
            for msg in chat_history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role.upper()}: {content}\n"

        prompt = f"""You are an assistant answering questions about a video based on its verified audio transcript and visual frame timeline.

VIDEO CONTEXT:
{context_prompt}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {query}

Please provide a direct, helpful, and accurate response. When referencing events, cite the specific timestamps (e.g., [02:15])."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                    },
                },
                timeout=90,
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"Error: Ollama status {response.status_code}"
        except Exception as e:
            return f"Error communicating with local Gemma model: {e}"
