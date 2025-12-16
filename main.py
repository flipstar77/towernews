#!/usr/bin/env python3
"""
Tower News - Reddit News Shorts Pipeline
Main entry point for the application.
"""

import argparse
import schedule
import time
from datetime import datetime
from dotenv import load_dotenv

from src.pipeline import NewsPipeline, load_config


def run_once():
    """Run the pipeline once."""
    print(f"\n{'='*60}")
    print(f"Tower News Pipeline - Single Run")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    pipeline = NewsPipeline()
    result = pipeline.run_sync()

    if result.get("success"):
        print(f"\n[SUCCESS] Video generated successfully!")
        print(f"  Path: {result.get('video_path')}")
    else:
        print(f"\n[FAILED] Pipeline failed: {result.get('error')}")

    return result


def run_scheduled():
    """Run the pipeline on schedule."""
    config = load_config()
    schedule_config = config.get("schedule", {})

    run_time = schedule_config.get("run_time", "12:00")

    print(f"\n{'='*60}")
    print(f"Tower News Pipeline - Scheduled Mode")
    print(f"Will run daily at: {run_time}")
    print(f"{'='*60}\n")

    # Schedule the job
    schedule.every().day.at(run_time).do(run_once)

    print(f"Scheduler started. Waiting for {run_time}...")
    print("Press Ctrl+C to stop.\n")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def test_scraper():
    """Test the Reddit scraper."""
    from src.tools import RedditScraper

    config = load_config()
    scraper = RedditScraper(config)

    print("Testing Reddit Scraper...")
    result = scraper.run()

    print(f"\nFound {result.get('count')} posts from r/{result.get('subreddit')}:")
    for i, post in enumerate(result.get("posts", []), 1):
        print(f"\n{i}. {post.get('title')[:60]}...")
        print(f"   Score: {post.get('score')} | Comments: {post.get('num_comments')}")


def test_tts():
    """Test the TTS tool."""
    from src.tools import TTSTool

    config = load_config()
    tts = TTSTool(config)

    print("Testing TTS...")
    result = tts.run(text="Welcome to Tower News! This is a test of the text to speech system.")

    print(f"Audio saved to: {result.get('audio_path')}")


def main():
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Tower News - Reddit News Shorts Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run once
  python main.py --scheduled  # Run on schedule
  python main.py --test-scraper  # Test Reddit scraper
  python main.py --test-tts      # Test TTS
        """
    )

    parser.add_argument(
        "--scheduled", "-s",
        action="store_true",
        help="Run on schedule (daily at configured time)"
    )
    parser.add_argument(
        "--test-scraper",
        action="store_true",
        help="Test the Reddit scraper"
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Test the TTS system"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    try:
        if args.test_scraper:
            test_scraper()
        elif args.test_tts:
            test_tts()
        elif args.scheduled:
            run_scheduled()
        else:
            run_once()

    except KeyboardInterrupt:
        print("\n\nPipeline stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
