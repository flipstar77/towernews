"""Regenerate video from existing audio segments (no new TTS API calls)."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.tools.video_generator import VideoGenerator
import yaml

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Find the latest audio files
audio_dir = Path("output/2025-12-16/audio")
timestamps = set()
for f in audio_dir.glob("music_normalized_*.mp3"):
    ts = f.stem.replace("music_normalized_", "")
    timestamps.add(ts)

# Get latest timestamp
latest_ts = sorted(timestamps)[-1]
print(f"Using audio from timestamp: {latest_ts}")

# Build audio segments list
audio_segments = []

# Music
music_path = str(audio_dir / f"music_normalized_{latest_ts}.mp3")
if Path(music_path).exists():
    import subprocess, json
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", music_path],
        capture_output=True, text=True
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    audio_segments.append({"type": "music", "path": music_path, "duration": duration})
    print(f"  music: {duration:.1f}s")

# Intro
intro_path = str(audio_dir / f"intro_{latest_ts}.mp3")
if Path(intro_path).exists():
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", intro_path],
        capture_output=True, text=True
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    audio_segments.append({"type": "intro", "path": intro_path, "duration": duration})
    print(f"  intro: {duration:.1f}s")

# Stories
for i in range(1, 5):
    story_path = str(audio_dir / f"story_{i}_{latest_ts}.mp3")
    if Path(story_path).exists():
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", story_path],
            capture_output=True, text=True
        )
        duration = float(json.loads(result.stdout)["format"]["duration"])
        audio_segments.append({"type": "story", "path": story_path, "duration": duration})
        print(f"  story_{i}: {duration:.1f}s")

# Outro
outro_path = str(audio_dir / f"outro_{latest_ts}.mp3")
if Path(outro_path).exists():
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", outro_path],
        capture_output=True, text=True
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    audio_segments.append({"type": "outro", "path": outro_path, "duration": duration})
    print(f"  outro: {duration:.1f}s")

print(f"\nTotal segments: {len(audio_segments)}")

# Test article data for stats display
test_articles = [
    {"post_title": "Amazing AI Discovery", "score": 15420, "num_comments": 892},
    {"post_title": "Tech News Update", "score": 8750, "num_comments": 456},
    {"post_title": "Breaking Story", "score": 3200, "num_comments": 234},
]

# Generate video
video_gen = VideoGenerator(config)
ticker_text = ">>> Tower News Test - Video Regeneration Test <<<"

result = video_gen.generate_segmented_video(
    audio_segments=audio_segments,
    screenshots=[],  # No screenshots
    ticker_text=ticker_text,
    presenter_image=None,
    articles=test_articles
)

print(f"\nVideo generated: {result}")
