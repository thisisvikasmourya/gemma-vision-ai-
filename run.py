import argparse
import sys
from pathlib import Path
import uvicorn

from app.config import DEFAULT_FPS, DEFAULT_LLM_MODEL, VIDEOS_DIR, VISION_MODEL
from app.pipeline import VideoUnderstandingPipeline


def main():
    parser = argparse.ArgumentParser(description="OmniVid AI: Video Intelligence & Metal Transcription")
    parser.add_argument("--video", "-v", type=str, help="Process a specific video file directly in CLI")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_LLM_MODEL, help="Ollama LLM model name")
    parser.add_argument("--vision-model", type=str, default=VISION_MODEL, help="Ollama Vision model name")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame rate for extraction (default 30 FPS)")
    parser.add_argument("--interval", "-i", type=float, default=None, help="Keyframe interval in seconds")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for web server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port for web server")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI Web Server")

    args = parser.parse_args()

    # If a video is specified, run CLI pipeline directly
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"❌ Error: Video file not found at {video_path}")
            sys.exit(1)

        print(f"🎬 Processing video: {video_path.name} at {args.fps} FPS...")
        pipeline = VideoUnderstandingPipeline(llm_model=args.model, vision_model=args.vision_model)

        def progress(step, pct, msg):
            print(f"  [{pct:3.0f}%] {step}: {msg}")

        result = pipeline.process_video(
            video_path=video_path,
            target_fps=args.fps,
            keyframe_interval=args.interval,
            progress_callback=progress,
        )

        print("\n" + "=" * 60)
        print("🎯 WHAT IS INSIDE THE VIDEO (Gemma 4 Analysis):")
        print("=" * 60)
        print(result.get("gemma_analysis", {}).get("report_markdown", "No summary generated."))

        print("\n" + "=" * 60)
        print("🎙️ AUDIO TRANSCRIPT SUMMARY:")
        print("=" * 60)
        total_segments = result.get("transcription", {}).get("total_segments", 0)
        print(f"Total Spoken Dialogue Segments: {total_segments}")
        print(f"Full Text Excerpt: {result.get('transcription', {}).get('full_text', '')[:500]}...")
        print("=" * 60)
        return

    # Default: Start Web Application
    print(f"🚀 Starting OmniVid AI Web Dashboard on http://{args.host}:{args.port}")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
