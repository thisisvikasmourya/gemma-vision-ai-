import logging
from typing import Any, Dict, List
from app.video import format_timestamp

logger = logging.getLogger(__name__)


def fuse_audio_and_visuals(
    transcript_segments: List[Dict[str, Any]],
    visual_frames: List[Dict[str, Any]],
    window_duration_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Combines timestamped audio dialogue segments and visual frame descriptions
    into synchronized time-window blocks.
    
    Each fused block contains:
    - start_sec / end_sec & formatted timestamps
    - spoken dialogue within this window
    - visual frame descriptions & image paths within this window
    """
    if not transcript_segments and not visual_frames:
        return []

    # Determine maximum duration from both streams
    max_t = 0.0
    for s in transcript_segments:
        max_t = max(max_t, s.get("end_sec", 0.0))
    for f in visual_frames:
        max_t = max(max_t, f.get("timestamp", 0.0))

    if max_t <= 0:
        max_t = 60.0

    timeline_blocks = []
    current_start = 0.0

    while current_start < max_t:
        current_end = current_start + window_duration_sec

        # Collect audio dialogue in this window
        window_dialogue = []
        for seg in transcript_segments:
            s_start = seg.get("start_sec", 0.0)
            s_end = seg.get("end_sec", 0.0)
            # Check overlap
            if not (s_end < current_start or s_start > current_end):
                window_dialogue.append(seg.get("text", "").strip())

        # Collect visual frames in this window
        window_visuals = []
        window_frame_images = []
        for frame in visual_frames:
            f_ts = frame.get("timestamp", 0.0)
            if current_start <= f_ts < current_end:
                desc = frame.get("description", "")
                if desc and not desc.startswith("Visual scene at"):
                    window_visuals.append(f"[{format_timestamp(f_ts)}] {desc}")
                window_frame_images.append({
                    "timestamp": f_ts,
                    "timestamp_formatted": frame.get("timestamp_formatted", format_timestamp(f_ts)),
                    "frame_path": frame.get("frame_path", ""),
                    "filename": frame.get("filename", ""),
                })

        dialogue_text = " ".join(window_dialogue).strip()
        visual_text = " | ".join(window_visuals).strip()

        # Only add block if there is either audio or visual content
        if dialogue_text or visual_text or window_frame_images:
            timeline_blocks.append({
                "window_index": len(timeline_blocks) + 1,
                "start_sec": current_start,
                "end_sec": current_end,
                "start_timestamp": format_timestamp(current_start),
                "end_timestamp": format_timestamp(current_end),
                "dialogue": dialogue_text,
                "visual_summary": visual_text if visual_text else "Scene progressing",
                "keyframes": window_frame_images,
            })

        current_start = current_end

    return timeline_blocks


def build_multimodal_context_prompt(
    video_meta: Dict[str, Any],
    timeline_blocks: List[Dict[str, Any]],
    full_transcript: str,
) -> str:
    """
    Format fused multimodal timeline into a clear, structured prompt context
    for the Gemma 4 LLM to synthesize 'What is inside the video'.
    """
    context_lines = [
        f"VIDEO METADATA:",
        f"- Title / File: {video_meta.get('filename', 'Unknown')}",
        f"- Total Duration: {video_meta.get('duration_formatted', 'Unknown')} ({video_meta.get('duration', 0):.1f}s)",
        f"- Resolution: {video_meta.get('width', 0)}x{video_meta.get('height', 0)} @ {video_meta.get('fps', 0)} FPS",
        "",
        "SYNCHRONIZED AUDIO & VISUAL TIMELINE:",
    ]

    for block in timeline_blocks[:100]:  # Limit context chunks for LLM
        time_str = f"[{block['start_timestamp']} -> {block['end_timestamp']}]"
        diag = block['dialogue'] if block['dialogue'] else "[No speech]"
        vis = block['visual_summary'] if block['visual_summary'] else "[No visual change]"
        context_lines.append(f"{time_str}")
        context_lines.append(f"  • Visual: {vis}")
        context_lines.append(f"  • Spoken Dialogue: {diag}")

    if len(full_transcript) > 0:
        context_lines.extend([
            "",
            "COMPLETE AUDIO TRANSCRIPTION EXCERPT:",
            full_transcript[:4000] + ("..." if len(full_transcript) > 4000 else ""),
        ])

    return "\n".join(context_lines)
