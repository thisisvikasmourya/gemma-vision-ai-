import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    total_sec = max(0.0, float(seconds))
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def get_video_info(video_path: str | Path) -> Dict[str, Any]:
    """Inspect video file using ffprobe and return detailed metadata."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except Exception as e:
        logger.error(f"Error probing video {video_path}: {e}")
        return {
            "filename": video_path.name,
            "duration": 0.0,
            "duration_formatted": "00:00:00",
            "has_audio": False,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "codec_video": "unknown",
            "codec_audio": "none",
        }

    format_info = data.get("format", {})
    streams = data.get("streams", [])

    duration = float(format_info.get("duration", 0.0))
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0
    codec_video = video_stream.get("codec_name", "unknown") if video_stream else "none"
    codec_audio = audio_stream.get("codec_name", "none") if audio_stream else "none"

    fps = 0.0
    if video_stream and "r_frame_rate" in video_stream:
        try:
            num, den = video_stream["r_frame_rate"].split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except Exception:
            fps = 30.0

    return {
        "filename": video_path.name,
        "filepath": str(video_path),
        "duration": duration,
        "duration_formatted": format_timestamp(duration),
        "has_audio": audio_stream is not None,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "codec_video": codec_video,
        "codec_audio": codec_audio,
        "size_bytes": video_path.stat().st_size,
    }


def extract_audio(video_path: str | Path, output_wav_path: str | Path) -> bool:
    """Extract audio track from video and convert to 16kHz 16-bit Mono WAV."""
    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_wav_path.exists() and output_wav_path.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
        return False


def extract_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    target_fps: float = 30.0,
    max_frames: int = 18000,
    interval_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts frames at 30 FPS (30 frames for every 1 second of video).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale frames if directory previously had old extraction runs
    for old_file in output_dir.glob("frame_*.jpg"):
        try:
            old_file.unlink()
        except Exception:
            pass

    fps_val = float(target_fps) if target_fps and target_fps > 0 else 30.0
    fps_filter = f"fps={fps_val}"

    output_pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", fps_filter,
        "-q:v", "2",
        "-frames:v", str(max_frames),
        output_pattern,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg frame extraction failed: {e.stderr}")
        return []

    # Map each frame (30 per sec) to its exact sub-second timestamp
    frames = []
    jpg_files = sorted(output_dir.glob("frame_*.jpg"))
    for idx, frame_file in enumerate(jpg_files):
        # 1 frame = 1/30th of a second (~0.0333s step)
        timestamp = idx / float(fps_val)
        frames.append({
            "index": idx + 1,
            "timestamp": round(timestamp, 4),
            "timestamp_formatted": format_timestamp(timestamp),
            "filepath": str(frame_file),
            "filename": frame_file.name,
        })

    logger.info(f"Extracted {len(frames)} frames at {fps_val} FPS (30 frames/sec).")
    return frames


