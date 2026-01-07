"""
Tower News Dashboard - Web Server
Dashboard for RAG chat and YouTube pipeline management.
"""

import os
import sys
import json
import subprocess
import threading
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from rag import UnifiedKnowledgeBase, QueryProcessor, RERANKER_AVAILABLE, create_reranker

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
HISTORY_FILE = PROJECT_ROOT / "data" / "post_history.json"
REACT_BUILD_DIR = PROJECT_ROOT / "web-ui" / "dist"

app = Flask(__name__, static_folder=str(REACT_BUILD_DIR), static_url_path='')
CORS(app)

# Initialize components
kb = UnifiedKnowledgeBase()
query_processor = QueryProcessor()
openai_client = OpenAI()

# Optional: Re-ranker for better result quality
reranker = None
if RERANKER_AVAILABLE:
    try:
        reranker = create_reranker(model_type="fast", use_cache=True)
        print("[Chat] Re-ranker enabled")
    except Exception as e:
        print(f"[Chat] Re-ranker not available: {e}")

SYSTEM_PROMPT = """You are a knowledgeable assistant for "The Tower" mobile idle game.

You have access to a knowledge base with Wiki content and Reddit community discussions.

GUIDELINES:
1. Use the provided context to answer questions - combine info from multiple sources
2. Wiki sources [Wiki] are authoritative for game mechanics and stats
3. Reddit posts/comments often have great tips, strategies, and community wisdom
4. If context is relevant but partial, summarize what you found and mention there may be more
5. For numbers/stats, prefer Wiki data. For strategies/tips, Reddit discussions are valuable
6. Be helpful and conversational, not robotic

When sources contain useful information, synthesize it into a helpful answer.
Only say "I don't have information" if the context is truly irrelevant to the question.

Be concise but helpful. Share the community's knowledge!"""


def search_context(query: str, limit: int = 8) -> tuple[str, list]:
    """Search knowledge base and return context + sources.
    Uses smart query detection and query preprocessing for abbreviation expansion."""

    # Preprocess query - expands abbreviations like DW -> Death Wave
    processed = query_processor.process(query)
    search_query = processed.expanded

    query_lower = query.lower()
    combined_results = []
    seen_ids = set()

    # Detect special queries
    is_top_query = any(word in query_lower for word in ["most upvoted", "top post", "best post", "highest score", "popular"])
    is_recent_query = any(word in query_lower for word in ["recent", "latest", "newest", "last"])
    is_announcement_query = "announcement" in query_lower or "monday" in query_lower

    # Handle special queries
    if is_top_query:
        # Get top posts by score
        top_results = kb.get_top_posts(limit=limit, doc_type="post")
        for r in top_results:
            post_id = r.metadata.get("post_id", "")
            if post_id not in seen_ids:
                seen_ids.add(post_id)
                combined_results.append(r)

    elif is_announcement_query:
        # Search specifically for announcements
        announcement_results = kb.search("Monday Morning Announcements", limit=limit)
        for r in announcement_results:
            post_id = r.metadata.get("post_id", "")
            if post_id not in seen_ids:
                seen_ids.add(post_id)
                combined_results.append(r)

    # Single search call - get more results, then prioritize (saves API calls)
    if len(combined_results) < limit:
        all_results = kb.search(search_query, limit=limit * 3)

        # Separate by type for prioritization
        wiki_results = [r for r in all_results if r.doc_type == "wiki"]
        post_results = [r for r in all_results if r.doc_type == "post"]
        other_results = [r for r in all_results if r.doc_type not in ("wiki", "post")]

        # Add in priority order: Wiki first (max 2), then posts, then others
        for r in wiki_results[:2]:
            post_id = r.metadata.get("post_id", "")
            if post_id not in seen_ids and len(combined_results) < limit:
                seen_ids.add(post_id)
                combined_results.append(r)

        for r in post_results:
            post_id = r.metadata.get("post_id", "")
            if post_id not in seen_ids and len(combined_results) < limit:
                seen_ids.add(post_id)
                combined_results.append(r)

        for r in other_results:
            post_id = r.metadata.get("post_id", "")
            if post_id not in seen_ids and len(combined_results) < limit:
                seen_ids.add(post_id)
                combined_results.append(r)

    if not combined_results:
        return "", []

    # Optional: Re-rank results for better relevance
    if reranker and len(combined_results) > 3:
        try:
            combined_results = reranker.rerank_search_results(
                query=search_query,
                results=combined_results,
                top_k=limit
            )
        except Exception as e:
            print(f"[Chat] Re-ranking failed: {e}")

    context_parts = []
    sources = []

    for i, result in enumerate(combined_results, 1):
        score = result.metadata.get("score", 0)
        post_type = result.metadata.get("post_type", "unknown")
        similarity = result.similarity
        title = result.metadata.get("title", result.metadata.get("parent_title", ""))

        # Mark Wiki content clearly
        type_label = "[Wiki]" if post_type == "wiki" else f"[{post_type}]"

        context_parts.append(
            f"[Source {i}] {type_label} (Score: {score})\nTitle: {title}\n{result.content}"
        )

        # Get URL for source
        reddit_url = result.metadata.get("reddit_url", "")
        permalink = result.metadata.get("permalink", "")
        wiki_url = result.metadata.get("url", "")  # Wiki pages have 'url'

        # Build full Reddit URL from permalink if needed
        if not reddit_url and permalink:
            reddit_url = f"https://reddit.com{permalink}"

        source_url = reddit_url or wiki_url or ""

        # Get flair
        flair = result.metadata.get("flair", result.metadata.get("parent_flair", ""))

        sources.append({
            "type": post_type,
            "title": title[:100] if title else "Community discussion",
            "flair": flair,
            "score": score,
            "similarity": round(similarity, 3),
            "url": source_url
        })

    return "\n\n---\n\n".join(context_parts), sources


@app.route("/")
def index():
    """Serve the React app."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/dashboard")
def dashboard_page():
    """Serve the React app for dashboard route."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/pipeline")
def pipeline_page():
    """Serve the Pipeline Builder page."""
    return render_template("pipeline.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat requests."""
    data = request.json
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Search for context
    context, sources = search_context(user_message, limit=5)

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add history (last 10 exchanges)
    for msg in history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add user message with context
    if context:
        user_content = f"""Context from community knowledge base:

{context}

---

User question: {user_message}"""
    else:
        user_content = f"User question: {user_message}\n\n(No relevant context found in knowledge base)"

    messages.append({"role": "user", "content": user_content})

    # Get response
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        assistant_message = response.choices[0].message.content

        return jsonify({
            "response": assistant_message,
            "sources": sources
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def search():
    """Direct search of knowledge base."""
    data = request.json
    query = data.get("query", "").strip()
    limit = data.get("limit", 10)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Preprocess query for abbreviation expansion
    processed = query_processor.process(query)
    results = kb.search(processed.expanded, limit=limit)

    return jsonify({
        "results": [
            {
                "content": r.content,
                "similarity": round(r.similarity, 3),
                "metadata": r.metadata
            }
            for r in results
        ]
    })


@app.route("/api/stats")
def stats():
    """Get knowledge base statistics."""
    return jsonify(kb.get_stats())


# ============== Pipeline Endpoints ==============

# Store pipeline status
pipeline_status = {
    "running": False,
    "progress": "",
    "last_run": None,
    "last_video": None,
    "error": None
}

def run_pipeline_async():
    """Run the pipeline in background."""
    global pipeline_status
    pipeline_status["running"] = True
    pipeline_status["progress"] = "Starting pipeline..."
    pipeline_status["error"] = None

    try:
        process = subprocess.Popen(
            ["python", str(PROJECT_ROOT / "main.py")],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        for line in process.stdout:
            line = line.strip()
            output_lines.append(line)
            # Update progress with last meaningful line
            if line and not line.startswith("="):
                pipeline_status["progress"] = line[-100:]

        process.wait()

        if process.returncode == 0:
            # Find the video URL in output
            for line in output_lines:
                if "youtube.com/watch" in line:
                    pipeline_status["last_video"] = line.split("https://")[-1]
                    pipeline_status["last_video"] = "https://" + pipeline_status["last_video"].split()[0]
                    break
            pipeline_status["progress"] = "Pipeline completed successfully!"
            pipeline_status["last_run"] = datetime.now().isoformat()
        else:
            pipeline_status["error"] = "Pipeline failed"
            pipeline_status["progress"] = "Pipeline failed"

    except Exception as e:
        pipeline_status["error"] = str(e)
        pipeline_status["progress"] = f"Error: {str(e)}"

    finally:
        pipeline_status["running"] = False


@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    """Start the video generation pipeline."""
    global pipeline_status

    if pipeline_status["running"]:
        return jsonify({"error": "Pipeline already running"}), 400

    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline_async)
    thread.start()

    return jsonify({"status": "started"})


@app.route("/api/pipeline/status")
def get_pipeline_status():
    """Get current pipeline status."""
    return jsonify(pipeline_status)


@app.route("/api/videos")
def get_videos():
    """Get list of generated videos."""
    videos = []

    if OUTPUT_DIR.exists():
        for date_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            if date_dir.is_dir() and date_dir.name[0].isdigit():
                # Find video and metadata
                for f in date_dir.iterdir():
                    if f.name.startswith("news_") and f.suffix == ".mp4":
                        # Try to find metadata
                        metadata_file = None
                        youtube_url = None
                        title = f"Video {date_dir.name}"

                        for mf in date_dir.iterdir():
                            if mf.name.startswith("metadata_") and mf.suffix == ".txt":
                                metadata_file = mf
                                break

                        if metadata_file and metadata_file.exists():
                            content = metadata_file.read_text(encoding='utf-8')
                            lines = content.split('\n')
                            for line in lines:
                                if line.startswith("Title:"):
                                    title = line.replace("Title:", "").strip()

                        videos.append({
                            "date": date_dir.name,
                            "title": title,
                            "path": str(f.relative_to(PROJECT_ROOT)),
                            "youtube_url": youtube_url
                        })

    return jsonify(videos[:20])  # Last 20 videos


@app.route("/api/history")
def get_history():
    """Get post history."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            posts = data.get("posts", {})
            # Convert to list sorted by date
            history = []
            for post_id, info in posts.items():
                history.append({
                    "id": post_id,
                    "title": info.get("title", "Unknown"),
                    "reported_at": info.get("reported_at", ""),
                    "video_date": info.get("video_date", "")
                })
            history.sort(key=lambda x: x.get("reported_at", ""), reverse=True)
            return jsonify(history[:50])

    return jsonify([])


@app.route("/api/recent-threads")
def get_recent_threads():
    """Get recent threads sorted by flair for banner display."""
    try:
        # Get recent posts by ingestion date (not search!)
        results = kb.get_recent_posts(limit=50, doc_type="post")

        # Group by flair and get recent ones
        threads = []
        for r in results:
            flair = r.metadata.get("flair", "General")
            if not flair:
                flair = "General"
            threads.append({
                "title": r.metadata.get("title", "")[:80],
                "flair": flair,
                "score": r.metadata.get("score", 0),
                "url": r.metadata.get("reddit_url", ""),
                "post_id": r.metadata.get("post_id", "")
            })

        # Sort by flair, then by score
        threads.sort(key=lambda x: (x["flair"], -x["score"]))

        # Return top 10
        return jsonify(threads[:10])
    except Exception as e:
        return jsonify([])


@app.route("/api/ingest/reddit", methods=["POST"])
def ingest_reddit():
    """Trigger Reddit ingestion."""
    # This would run in background ideally
    return jsonify({"status": "not_implemented", "message": "Use CLI for now: python ingest_massive.py"})


@app.route("/api/pipeline/segments")
@app.route("/api/pipeline/segments/<date>")
def get_pipeline_segments(date=None):
    """Get all pipeline segments/artifacts for a specific date."""
    if not date:
        # Find most recent date
        if OUTPUT_DIR.exists():
            dates = sorted([d.name for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()], reverse=True)
            if dates:
                date = dates[0]
            else:
                return jsonify({"error": "No output found"}), 404
        else:
            return jsonify({"error": "Output directory not found"}), 404

    date_dir = OUTPUT_DIR / date
    if not date_dir.exists():
        return jsonify({"error": f"No output for date {date}"}), 404

    result = {
        "date": date,
        "segments": [],
        "audio": [],
        "screenshots": [],
        "final_video": None,
        "metadata": None
    }

    # Get segments
    segments_dir = date_dir / "segments"
    if segments_dir.exists():
        for f in sorted(segments_dir.iterdir()):
            if f.suffix == ".mp4":
                # Parse segment type from filename
                seg_type = "unknown"
                if "music" in f.name:
                    seg_type = "music"
                elif "intro" in f.name:
                    seg_type = "intro"
                elif "story" in f.name:
                    seg_type = "story"
                elif "outro" in f.name:
                    seg_type = "outro"
                elif "concat" in f.name:
                    seg_type = "concat"

                result["segments"].append({
                    "name": f.name,
                    "type": seg_type,
                    "path": f"output/{date}/segments/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1)
                })

    # Get audio files
    audio_dir = date_dir / "audio"
    if audio_dir.exists():
        for f in sorted(audio_dir.iterdir()):
            if f.suffix == ".mp3":
                audio_type = "unknown"
                if "music" in f.name:
                    audio_type = "music"
                elif "intro" in f.name:
                    audio_type = "intro"
                elif "story" in f.name:
                    audio_type = "story"
                elif "outro" in f.name:
                    audio_type = "outro"
                elif "combined" in f.name:
                    audio_type = "combined"

                result["audio"].append({
                    "name": f.name,
                    "type": audio_type,
                    "path": f"output/{date}/audio/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1)
                })

    # Get screenshots
    screenshots_dir = date_dir / "screenshots"
    if screenshots_dir.exists():
        for f in sorted(screenshots_dir.iterdir()):
            if f.suffix in [".png", ".jpg", ".jpeg"]:
                result["screenshots"].append({
                    "name": f.name,
                    "path": f"output/{date}/screenshots/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1)
                })

    # Get final video
    for f in date_dir.iterdir():
        if f.name.startswith("news_") and f.suffix == ".mp4":
            result["final_video"] = {
                "name": f.name,
                "path": f"output/{date}/{f.name}",
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
            }
            break

    # Get metadata
    for f in date_dir.iterdir():
        if f.name.startswith("metadata_") and f.suffix == ".txt":
            content = f.read_text(encoding='utf-8')
            result["metadata"] = {
                "name": f.name,
                "path": f"output/{date}/{f.name}",
                "content": content
            }
            break

    return jsonify(result)


@app.route("/api/pipeline/dates")
def get_pipeline_dates():
    """Get all available pipeline output dates."""
    dates = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            if d.is_dir() and d.name[0].isdigit():
                # Check if it has content
                has_video = any(f.name.startswith("news_") and f.suffix == ".mp4" for f in d.iterdir())
                has_segments = (d / "segments").exists()

                dates.append({
                    "date": d.name,
                    "has_video": has_video,
                    "has_segments": has_segments
                })

    return jsonify(dates[:30])  # Last 30 dates


@app.route('/output/<path:filepath>')
def serve_output(filepath):
    """Serve files from output directory."""
    return send_from_directory(str(OUTPUT_DIR), filepath)


@app.route("/api/pipeline/build", methods=["POST"])
def build_custom_video():
    """Build a custom video from selected segments."""
    import subprocess
    from datetime import datetime

    data = request.json
    segments = data.get("segments", [])
    add_ticker = data.get("add_ticker", True)
    ticker_text = data.get("ticker_text", "Tower News")
    output_name = data.get("output_name", "custom")
    date = data.get("date")

    if not segments:
        return jsonify({"success": False, "error": "No segments provided"}), 400

    # Convert relative paths to absolute
    segment_paths = []
    for seg in segments:
        full_path = PROJECT_ROOT / seg
        if full_path.exists():
            segment_paths.append(str(full_path))
        else:
            return jsonify({"success": False, "error": f"Segment not found: {seg}"}), 400

    # Create output directory
    output_date = date or datetime.now().strftime("%Y-%m-%d")
    output_dir = OUTPUT_DIR / output_date
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create concat list file
    timestamp = datetime.now().strftime("%H%M%S")
    concat_list = output_dir / f"custom_concat_{timestamp}.txt"
    with open(concat_list, "w") as f:
        for path in segment_paths:
            f.write(f"file '{path}'\n")

    # Output video path
    output_video = output_dir / f"{output_name}_{timestamp}.mp4"

    try:
        # Step 1: Concatenate segments (re-encode to ensure audio works)
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_video)
        ]

        result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"FFmpeg concat failed: {result.stderr[-500:]}"
            }), 500

        # Step 2: Add ticker if requested
        if add_ticker and ticker_text:
            output_with_ticker = output_dir / f"{output_name}_ticker_{timestamp}.mp4"

            # Escape special characters in ticker text
            safe_ticker = ticker_text.replace("'", "\\'").replace(":", "\\:")

            ticker_cmd = [
                "ffmpeg", "-y",
                "-i", str(output_video),
                "-vf", f"drawtext=text='{safe_ticker}':fontsize=24:fontcolor=white:x=w-mod(t*100\\,w+tw):y=h-40:font=Arial",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(output_with_ticker)
            ]

            result = subprocess.run(ticker_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Use the ticker version
                output_video.unlink()  # Remove non-ticker version
                output_video = output_with_ticker

        return jsonify({
            "success": True,
            "video_path": str(output_video.relative_to(PROJECT_ROOT)),
            "size_mb": round(output_video.stat().st_size / (1024 * 1024), 2)
        })

    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Build timed out"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Catch-all route for React Router (SPA)
@app.errorhandler(404)
def not_found(e):
    """Serve React app for client-side routing."""
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == "__main__":
    print("Starting Tower News Dashboard (React UI)...")
    print(f"React build dir: {REACT_BUILD_DIR}")
    print(f"Knowledge base: {kb.get_stats()}")
    print(f"Open: http://localhost:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
