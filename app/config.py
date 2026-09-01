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
VISION_MODEL = os.getenv("VISION_MODEL", "gemma4:12b-mlx")

# Processing Defaults
DEFAULT_KEYFRAME_INTERVAL_SEC = 5.0  # 1 keyframe every 5 seconds
SCENE_DETECTION_THRESHOLD = 0.35     # Scene change sensitivity (0.0 to 1.0)
MAX_FRAMES_PER_VIDEO = 300           # Cap for frame analysis to keep inference fast
