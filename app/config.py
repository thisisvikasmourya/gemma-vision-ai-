import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Paths
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
FRAMES_DIR = DATA_DIR / "frames"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
UPLOADS_DIR = BASE_DIR / "uploads"
TMP_DIR = BASE_DIR / "tmp"

# Ensure all required directories exist
for directory in [DATA_DIR, VIDEOS_DIR, AUDIO_DIR, FRAMES_DIR, TRANSCRIPTS_DIR, UPLOADS_DIR, TMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Whisper Configuration
WHISPER_CPP_DIR = BASE_DIR / "whisper.cpp"
WHISPER_BIN = WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli"
# Fallback bin names if build structure differs
WHISPER_BIN_FALLBACKS = [
    WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli",
    WHISPER_CPP_DIR / "whisper-cli",
    WHISPER_CPP_DIR / "main",
    WHISPER_CPP_DIR / "build" / "bin" / "main",
]
DEFAULT_WHISPER_MODEL = WHISPER_CPP_DIR / "models" / "ggml-base.en.bin"

# Ollama / Local LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:12b-mlx")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.5:9b-mlx")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))

# Processing Defaults (30 FPS)
DEFAULT_FPS = float(os.getenv("DEFAULT_FPS", "30.0"))  # 30 frames per second (1 sec = 30 frames)
DEFAULT_KEYFRAME_INTERVAL_SEC = 1.0 / DEFAULT_FPS      # 0.0333s step per frame
SCENE_DETECTION_THRESHOLD = 0.35                       # Scene change sensitivity
MAX_FRAMES_PER_VIDEO = int(os.getenv("MAX_FRAMES_PER_VIDEO", "18000"))  # Cap up to 10 mins @ 30 FPS




