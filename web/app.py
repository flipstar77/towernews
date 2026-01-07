"""Streamlit Web UI for Tower Journey Pipeline."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import yaml
from datetime import datetime, timedelta
from PIL import Image, ImageGrab
from dotenv import load_dotenv
import io
import json

load_dotenv()

from src.journey import ImageAnalyzer, SegmentTracker, ScreenClassifier, BattleReportParser, TranscriptAnalyzer, JourneyVideoGenerator, KnowledgeBase, TOWER_GAME_KNOWLEDGE, ProgressTracker
from src.pipeline import NewsPipeline, load_config as load_news_config
from src.tools import RedditScraper
from src.ai import TowerAgent, BulkRedditScraper, TrainingDataExtractor


def load_config():
    """Load journey config."""
    config_path = Path("config/journey_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_tier_icon(tier_str):
    """Get tier icon based on tier string."""
    if "17" in str(tier_str) or "+" in str(tier_str):
        return "🏆"  # Gold trophy for high tiers
    elif "14" in str(tier_str) or "15" in str(tier_str) or "16" in str(tier_str):
        return "🥈"  # Silver for mid-high
    else:
        return "🎮"  # Default


def calculate_cells_per_hour(battle):
    """Calculate cells per hour from battle data if not already present."""
    summary = battle.get("summary", {})
    values = battle.get("values", {})

    # Check if we already have it
    existing = summary.get("cells_per_hour")
    if existing and existing != "?":
        return existing

    # Calculate from cells earned and real time
    cells_earned = values.get("cells_earned", {}).get("numeric", 0)
    real_time_str = values.get("real_time", {}).get("value", "")

    if cells_earned > 0 and real_time_str:
        import re
        hours = 0.0
        day_match = re.search(r'(\d+)d', real_time_str)
        if day_match:
            hours += int(day_match.group(1)) * 24
        hour_match = re.search(r'(\d+)h', real_time_str)
        if hour_match:
            hours += int(hour_match.group(1))
        min_match = re.search(r'(\d+)m', real_time_str)
        if min_match:
            hours += int(min_match.group(1)) / 60
        sec_match = re.search(r'(\d+)s', real_time_str)
        if sec_match:
            hours += int(sec_match.group(1)) / 3600

        if hours > 0:
            cells_per_hour = cells_earned / hours
            if cells_per_hour >= 1_000_000:
                return f"{cells_per_hour/1_000_000:.2f}M"
            elif cells_per_hour >= 1_000:
                return f"{cells_per_hour/1_000:.2f}K"
            else:
                return f"{cells_per_hour:.0f}"

    return "?"


def parse_battle_date(date_str):
    """Parse battle date string to datetime for sorting."""
    from datetime import datetime
    if not date_str or date_str == "Unknown":
        return datetime.min

    # Try common formats: "Dec 19, 2025 06:13"
    formats = [
        "%b %d, %Y %H:%M",  # Dec 19, 2025 06:13
        "%b %d, %Y",        # Dec 19, 2025
        "%Y-%m-%d %H:%M",   # 2025-12-19 06:13
        "%Y-%m-%d",         # 2025-12-19
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return datetime.min


def main():
    st.set_page_config(
        page_title="Tower Journey",
        page_icon="🗼",
        layout="wide"
    )

    # Custom CSS for battle cards
    st.markdown("""
    <style>
    .battle-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #0f3460;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }
    .battle-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .tier-badge {
        background: #e94560;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .wave-badge {
        color: #00d9ff;
        font-size: 1.2em;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid #0f3460;
    }
    .stat-label {
        color: #888;
    }
    .stat-value {
        color: #00d9ff;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🗼 Tower Journey")
    st.markdown("Track your Tower progress and generate journey videos!")

    # Load config and tools
    config = load_config()
    classifier = ScreenClassifier(config)
    tracker = SegmentTracker(config)
    battle_parser = BattleReportParser()

    # Sidebar - Quick Stats
    with st.sidebar:
        st.header("📊 Quick Stats")

        # Load battle history
        history_path = Path("data/journey/battle_history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                battle_history = json.load(f)
            st.metric("Total Battles", len(battle_history.get("battles", [])))
        else:
            battle_history = {"battles": []}
            st.info("No battles recorded yet")

        st.divider()

        if st.button("🗑️ Clear All History", type="secondary"):
            if history_path.exists():
                history_path.unlink()
            st.success("History cleared!")
            st.rerun()

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["📸 Add Battle", "📈 Battle History", "📊 Weekly Report", "🎬 Generate Video", "📝 Transcript Analysis", "📋 Video Template", "🧠 Knowledge Base", "📺 Tower News", "📈 Progress Tracker", "🤖 Tower AI"])

    # TAB 1: Add Battle Report
    with tab1:
        st.header("Add Battle Report")

        # Two input methods
        input_method = st.radio(
            "Input Method",
            ["📋 Paste Text", "📸 Paste Screenshot"],
            horizontal=True
        )

        if input_method == "📋 Paste Text":
            battle_text = st.text_area(
                "Paste your Battle Report here",
                height=400,
                placeholder="Copy and paste your entire Battle Report...\n\nBattle Report\nBattle Date\tDec 19, 2025 06:13\nGame Time\t1d 18h 55m 26s\nTier\t13\nWave\t6649\nCoins earned\t549.23T\n...",
                key="battle_text"
            )

            if battle_text and st.button("📊 Parse & Save Battle Report", type="primary"):
                with st.spinner("Parsing battle report..."):
                    result = battle_parser.parse(battle_text)

                    if result.get("values"):
                        # Save to battle history
                        history_path = Path("data/journey/battle_history.json")
                        history_path.parent.mkdir(parents=True, exist_ok=True)

                        if history_path.exists():
                            with open(history_path, 'r') as f:
                                battle_history = json.load(f)
                        else:
                            battle_history = {"battles": []}

                        # Add new battle with timestamp
                        summary = result.get("summary", {})
                        # Ensure all summary fields exist
                        summary.setdefault("tier", "?")
                        summary.setdefault("wave", "?")
                        summary.setdefault("coins_earned", "?")
                        summary.setdefault("coins_per_hour", "?")
                        summary.setdefault("cells_earned", "?")
                        summary.setdefault("cells_per_hour", "?")
                        summary.setdefault("battle_date", "Unknown")
                        summary.setdefault("is_tournament", False)

                        # Check for duplicates (same battle_date, tier, and wave)
                        is_duplicate = False
                        for existing in battle_history["battles"]:
                            ex_summary = existing.get("summary", {})
                            if (ex_summary.get("battle_date") == summary.get("battle_date") and
                                ex_summary.get("tier") == summary.get("tier") and
                                ex_summary.get("wave") == summary.get("wave")):
                                is_duplicate = True
                                break

                        if is_duplicate:
                            st.warning(f"⚠️ Duplicate! Battle from {summary.get('battle_date')} (Tier {summary.get('tier')}, Wave {summary.get('wave')}) already exists.")
                        else:
                            battle_entry = {
                                "id": len(battle_history["battles"]) + 1,
                                "added_at": datetime.now().isoformat(),
                                "summary": summary,
                                "values": result.get("values", {}),
                                "values_by_category": result.get("values_by_category", {})
                            }
                            battle_history["battles"].insert(0, battle_entry)  # Add to front

                            with open(history_path, 'w') as f:
                                json.dump(battle_history, f, indent=2)

                            st.success(f"✅ Battle saved! Tier {summary.get('tier', '?')} - Wave {summary.get('wave', '?')}")
                            st.rerun()
                    else:
                        st.error("Could not parse battle report. Make sure you copied the full report.")

        else:  # Screenshot paste
            st.info("📸 Use Ctrl+V to paste a screenshot from clipboard, or upload an image file.")

            # File uploader as fallback
            uploaded_file = st.file_uploader("Or upload a screenshot", type=["png", "jpg", "jpeg"])

            # Try to get image from clipboard
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📋 Paste from Clipboard", type="primary"):
                    try:
                        clipboard_image = ImageGrab.grabclipboard()
                        if clipboard_image:
                            st.session_state["clipboard_image"] = clipboard_image
                            st.success("Image pasted from clipboard!")
                            st.rerun()
                        else:
                            st.warning("No image found in clipboard. Copy a screenshot first!")
                    except Exception as e:
                        st.error(f"Could not access clipboard: {e}")

            # Display and process image
            image_to_process = None

            if "clipboard_image" in st.session_state:
                image_to_process = st.session_state["clipboard_image"]
                st.image(image_to_process, caption="Clipboard Image", use_container_width=True)
            elif uploaded_file:
                image_to_process = Image.open(uploaded_file)
                st.image(image_to_process, caption="Uploaded Image", use_container_width=True)

            if image_to_process:
                if st.button("🔍 Analyze Screenshot", type="primary"):
                    with st.spinner("Analyzing screenshot with AI..."):
                        # Save temp image
                        temp_path = Path("data/journey/temp_screenshot.png")
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        image_to_process.save(temp_path)

                        # Classify and extract
                        result = classifier.classify_and_extract(str(temp_path))

                        st.subheader(f"Screen Type: {result.get('screen_name', 'Unknown')}")
                        st.write(f"Confidence: {result.get('confidence', 0):.0%}")

                        if result.get("values"):
                            st.json(result["values"])

                        # Clear clipboard image after processing
                        if "clipboard_image" in st.session_state:
                            del st.session_state["clipboard_image"]

    # TAB 2: Battle History (Card View)
    with tab2:
        st.header("Battle History")

        history_path = Path("data/journey/battle_history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                battle_history = json.load(f)

            battles = battle_history.get("battles", [])

            # Sort battles chronologically by battle date (newest first)
            battles_sorted = sorted(
                battles,
                key=lambda b: parse_battle_date(b.get("summary", {}).get("battle_date", "")),
                reverse=True  # Newest first
            )

            if not battles_sorted:
                st.info("No battles recorded yet. Add your first battle report!")
            else:
                for i, battle in enumerate(battles_sorted):
                    summary = battle.get("summary", {})
                    tier = summary.get("tier", "?")
                    wave = summary.get("wave", "?")
                    coins = summary.get("coins_earned", "?")
                    coins_hr = summary.get("coins_per_hour", "?")
                    cells = summary.get("cells_earned", "?")
                    cells_hr = calculate_cells_per_hour(battle)  # Calculate if missing
                    date = summary.get("battle_date", "Unknown")

                    # Calculate golden bot %
                    values = battle.get("values", {})
                    total_enemies = values.get("total_enemies", {}).get("numeric", 0)
                    golden_bot_kills = values.get("golden_bot_kills", {}).get("numeric", 0)
                    golden_bot_pct = f"{(golden_bot_kills / total_enemies * 100):.1f}%" if total_enemies > 0 else "?"

                    # Check if tournament
                    is_tournament = summary.get("is_tournament", False)
                    run_type = "Tournament" if is_tournament else "Regular"

                    # Battle card container
                    if is_tournament:
                        st.markdown(f"### :trophy: **TOURNAMENT** · Tier {tier} · Wave {wave} · {date}")
                    else:
                        st.markdown(f"### Tier {tier} · Wave {wave} · {date}")

                    # Summary metrics (always visible)
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Coins", coins)
                    with col2:
                        st.metric("Coins/Hour", coins_hr)
                    with col3:
                        st.metric("Cells", cells)
                    with col4:
                        st.metric("Cells/Hour", cells_hr if cells_hr else "?")
                    with col5:
                        st.metric("GB Kills %", golden_bot_pct)

                    # Detailed stats in expander
                    values_by_cat = battle.get("values_by_category", {})

                    with st.expander("View Details", expanded=False):
                        if values_by_cat:
                            # Create tabs for each category
                            cat_names = list(battle_parser.CATEGORIES.keys())
                            cat_labels = [battle_parser.CATEGORIES.get(c, c) for c in cat_names]

                            detail_tabs = st.tabs(cat_labels)

                            for tab_idx, cat_key in enumerate(cat_names):
                                with detail_tabs[tab_idx]:
                                    cat_values = values_by_cat.get(cat_key, {})
                                    if cat_values:
                                        for field_name, field_data in cat_values.items():
                                            col1, col2 = st.columns([2, 1])
                                            with col1:
                                                st.text(field_data.get("label", field_name))
                                            with col2:
                                                st.text(field_data.get("value", "N/A"))
                                    else:
                                        st.caption("No data for this category")

                        # Delete button - use index + battle id for unique key
                        battle_id = battle.get("id")
                        if st.button(f"Delete Battle", key=f"delete_{i}_{battle_id}", type="secondary"):
                            # Find and remove by id from original list
                            battles[:] = [b for b in battles if b.get("id") != battle_id]
                            with open(history_path, 'w') as f:
                                json.dump(battle_history, f, indent=2)
                            st.rerun()

                    st.divider()
        else:
            st.info("No battles recorded yet. Add your first battle report!")

    # TAB 3: Weekly Report (Automated)
    with tab3:
        st.header("📊 Weekly Progress Report")
        st.markdown("*Automatically generate your weekly progress summary and video script.*")

        history_path = Path("data/journey/battle_history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                battle_history = json.load(f)
            battles = battle_history.get("battles", [])
        else:
            battles = []

        if not battles:
            st.warning("No battle data yet. Add some battle reports first!")
        else:
            # Date range selection
            st.subheader("1. Select Time Period")

            col1, col2 = st.columns(2)
            with col1:
                # Get date range from battles
                all_dates = []
                for b in battles:
                    date_str = b.get("summary", {}).get("battle_date", "")
                    parsed = parse_battle_date(date_str)
                    if parsed != datetime.min:
                        all_dates.append(parsed)

                if all_dates:
                    min_date = min(all_dates).date()
                    max_date = max(all_dates).date()
                else:
                    min_date = datetime.now().date()
                    max_date = datetime.now().date()

                week_start = st.date_input(
                    "Week Start",
                    value=max_date - timedelta(days=7),
                    max_value=max_date
                )
            with col2:
                week_end = st.date_input(
                    "Week End",
                    value=max_date,
                    max_value=max_date
                )

            # Filter battles by date range
            week_battles = []
            for b in battles:
                date_str = b.get("summary", {}).get("battle_date", "")
                battle_date = parse_battle_date(date_str)
                if battle_date != datetime.min:
                    if week_start <= battle_date.date() <= week_end:
                        week_battles.append(b)

            st.info(f"Found **{len(week_battles)}** battles in selected period")

            if week_battles:
                # Separate regular and tournament runs
                regular_runs = [b for b in week_battles if not b.get("summary", {}).get("is_tournament", False)]
                tournament_runs = [b for b in week_battles if b.get("summary", {}).get("is_tournament", False)]

                st.divider()
                st.subheader("2. Weekly Statistics")

                # Calculate aggregate stats for the week
                total_coins = 0
                total_cells = 0
                total_enemies = 0
                total_gb_kills = 0
                coins_per_hour_list = []
                cells_per_hour_list = []
                waves_list = []

                for run in regular_runs:
                    values = run.get("values", {})
                    summary = run.get("summary", {})

                    total_coins += values.get("coins_earned", {}).get("numeric", 0)
                    total_cells += values.get("cells_earned", {}).get("numeric", 0)
                    total_enemies += values.get("total_enemies", {}).get("numeric", 0)
                    total_gb_kills += values.get("golden_bot_kills", {}).get("numeric", 0)

                    cph = values.get("coins_per_hour", {}).get("numeric", 0)
                    if cph > 0:
                        coins_per_hour_list.append(cph)

                    cellph = values.get("cells_per_hour", {}).get("numeric", 0)
                    if cellph > 0:
                        cells_per_hour_list.append(cellph)

                    try:
                        wave_num = int(str(summary.get("wave", "0")).replace(",", ""))
                        waves_list.append(wave_num)
                    except:
                        pass

                # Format numbers
                def format_num(n):
                    if n >= 1e12:
                        return f"{n/1e12:.2f}T"
                    elif n >= 1e9:
                        return f"{n/1e9:.2f}B"
                    elif n >= 1e6:
                        return f"{n/1e6:.2f}M"
                    elif n >= 1e3:
                        return f"{n/1e3:.2f}K"
                    return f"{n:.0f}"

                avg_cph = sum(coins_per_hour_list) / len(coins_per_hour_list) if coins_per_hour_list else 0
                avg_cellph = sum(cells_per_hour_list) / len(cells_per_hour_list) if cells_per_hour_list else 0
                avg_wave = sum(waves_list) / len(waves_list) if waves_list else 0
                best_wave = max(waves_list) if waves_list else 0
                gb_pct = (total_gb_kills / total_enemies * 100) if total_enemies > 0 else 0

                # Display stats
                st.markdown("#### Regular Runs")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Runs This Week", len(regular_runs))
                with col2:
                    st.metric("Total Coins", format_num(total_coins))
                with col3:
                    st.metric("Total Cells", format_num(total_cells))
                with col4:
                    st.metric("Best Wave", f"{best_wave:,}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Coins/Hour", format_num(avg_cph))
                with col2:
                    st.metric("Avg Cells/Hour", format_num(avg_cellph))
                with col3:
                    st.metric("Avg Wave", f"{avg_wave:.0f}")
                with col4:
                    st.metric("GB Kill %", f"{gb_pct:.1f}%")

                if tournament_runs:
                    st.markdown("#### Tournament Runs")
                    for t_run in tournament_runs:
                        t_summary = t_run.get("summary", {})
                        st.markdown(f"🏆 **Tier {t_summary.get('tier', '?')}** - Wave {t_summary.get('wave', '?')} ({t_summary.get('battle_date', '')})")

                # Generate script
                st.divider()
                st.subheader("3. Auto-Generated Video Script")

                if st.button("Generate Weekly Script", type="primary"):
                    transcript_analyzer = TranscriptAnalyzer()

                    # Build weekly data summary
                    weekly_data = {
                        "period": f"{week_start} to {week_end}",
                        "regular_runs": len(regular_runs),
                        "tournament_runs": len(tournament_runs),
                        "total_coins": format_num(total_coins),
                        "total_cells": format_num(total_cells),
                        "avg_coins_per_hour": format_num(avg_cph),
                        "avg_cells_per_hour": format_num(avg_cellph),
                        "best_wave": best_wave,
                        "avg_wave": avg_wave,
                        "gb_kill_pct": f"{gb_pct:.1f}%"
                    }

                    # Generate template-based script
                    template = transcript_analyzer.generate_video_template()

                    script = f"""# Tower News Weekly Update
**Period:** {weekly_data['period']}

---

## INTRO (0:00 - 0:30)
*Hook & Greeting*

"What's up everyone, welcome back to Tower News! This week has been HUGE - let me show you what happened!"

---

## WEEKLY STATS OVERVIEW (0:30 - 2:00)
*Show progress this week*

**This Week's Numbers:**
- Regular runs completed: **{weekly_data['regular_runs']}**
- Tournament runs: **{weekly_data['tournament_runs']}**
- Total coins earned: **{weekly_data['total_coins']}**
- Total cells earned: **{weekly_data['total_cells']}**

**Efficiency:**
- Average coins/hour: **{weekly_data['avg_coins_per_hour']}**
- Average cells/hour: **{weekly_data['avg_cells_per_hour']}**
- Best wave reached: **{weekly_data['best_wave']:,}**
- Average wave: **{weekly_data['avg_wave']:.0f}**
- Golden Bot kill %: **{weekly_data['gb_kill_pct']}**

---

## HIGHLIGHTS (2:00 - 6:00)
*Cover the best moments*

"""
                    # Add best run details
                    if waves_list:
                        best_run_idx = waves_list.index(max(waves_list))
                        best_run = regular_runs[best_run_idx]
                        best_summary = best_run.get("summary", {})
                        script += f"""### Best Run of the Week
- **Tier:** {best_summary.get('tier', '?')}
- **Wave:** {best_summary.get('wave', '?')}
- **Coins:** {best_summary.get('coins_earned', '?')}
- **Coins/Hour:** {best_summary.get('coins_per_hour', '?')}
- **Date:** {best_summary.get('battle_date', '?')}

"""

                    # Add tournament section if applicable
                    if tournament_runs:
                        script += """---

## TOURNAMENT UPDATE (6:00 - 10:00)
*Tournament performance*

"""
                        for t_run in tournament_runs:
                            t_sum = t_run.get("summary", {})
                            t_vals = t_run.get("values", {})
                            script += f"""### Tournament Run
- **Tier:** {t_sum.get('tier', '?')}
- **Wave:** {t_sum.get('wave', '?')}
- **Coins Earned:** {t_sum.get('coins_earned', '?')}
- **Date:** {t_sum.get('battle_date', '?')}

"""

                    script += """---

## TIPS & TAKEAWAYS (10:00 - 12:00)
*What I learned this week*

- [Add your observations about what worked well]
- [Any strategy changes you made]
- [Plans for next week]

---

## OUTRO (12:00 - end)

"That's all for this week! Drop a comment and let me know how YOUR week went. Don't forget to like and subscribe, and I'll see you in the next one!"

---

🤖 Generated with Tower News Pipeline
"""

                    st.session_state["weekly_script"] = script
                    st.success("Script generated!")

                # Display generated script
                if "weekly_script" in st.session_state:
                    st.text_area(
                        "Generated Script",
                        st.session_state["weekly_script"],
                        height=500
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Script",
                            st.session_state["weekly_script"],
                            file_name=f"tower_news_weekly_{week_start}.md",
                            mime="text/markdown"
                        )
                    with col2:
                        if st.button("📋 Copy to Clipboard"):
                            st.info("Use Ctrl+A then Ctrl+C in the text area above")

        # Batch upload section
        st.divider()
        st.subheader("📸 Batch Upload Screenshots")
        st.markdown("*Drop multiple battle report screenshots at once*")

        uploaded_files = st.file_uploader(
            "Upload multiple screenshots",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files")

            if st.button("🔄 Process All Screenshots", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
                    progress_bar.progress((i + 1) / len(uploaded_files))

                    try:
                        # Save temp file
                        image = Image.open(file)
                        temp_path = Path(f"data/journey/temp_batch_{i}.png")
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(temp_path)

                        # Classify and extract
                        result = classifier.classify_and_extract(str(temp_path))

                        if result.get("screen_type") == "battle_report" and result.get("values"):
                            results.append({
                                "file": file.name,
                                "result": result,
                                "status": "success"
                            })
                        else:
                            results.append({
                                "file": file.name,
                                "result": result,
                                "status": "not_battle_report"
                            })

                        # Clean up
                        temp_path.unlink()

                    except Exception as e:
                        results.append({
                            "file": file.name,
                            "error": str(e),
                            "status": "error"
                        })

                progress_bar.empty()
                status_text.empty()

                # Show results
                success_count = sum(1 for r in results if r["status"] == "success")
                st.success(f"Processed {len(results)} files. {success_count} battle reports found.")

                # Save successful results
                if success_count > 0:
                    history_path = Path("data/journey/battle_history.json")
                    if history_path.exists():
                        with open(history_path, 'r') as f:
                            battle_history = json.load(f)
                    else:
                        battle_history = {"battles": []}

                    added_count = 0
                    for r in results:
                        if r["status"] == "success":
                            result = r["result"]
                            summary = result.get("summary", {})

                            # Check for duplicates
                            is_duplicate = False
                            for existing in battle_history["battles"]:
                                ex_summary = existing.get("summary", {})
                                if (ex_summary.get("battle_date") == summary.get("battle_date") and
                                    ex_summary.get("tier") == summary.get("tier") and
                                    ex_summary.get("wave") == summary.get("wave")):
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                battle_entry = {
                                    "id": len(battle_history["battles"]) + 1,
                                    "added_at": datetime.now().isoformat(),
                                    "summary": summary,
                                    "values": result.get("values", {}),
                                    "values_by_category": result.get("values_by_category", {})
                                }
                                battle_history["battles"].insert(0, battle_entry)
                                added_count += 1

                    if added_count > 0:
                        with open(history_path, 'w') as f:
                            json.dump(battle_history, f, indent=2)
                        st.success(f"Added {added_count} new battles to history!")

                # Show detailed results
                with st.expander("View Processing Results"):
                    for r in results:
                        if r["status"] == "success":
                            st.markdown(f"✅ **{r['file']}** - Battle Report parsed")
                        elif r["status"] == "not_battle_report":
                            st.markdown(f"⚠️ **{r['file']}** - Not a battle report")
                        else:
                            st.markdown(f"❌ **{r['file']}** - Error: {r.get('error', 'Unknown')}")

    # TAB 4: Generate Video
    with tab4:
        st.header("Generate Journey Video")

        history_path = Path("data/journey/battle_history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                battle_history = json.load(f)

            battles = battle_history.get("battles", [])

            if not battles:
                st.warning("No battle data. Add some battle reports first!")
            else:
                # Separate regular runs from tournament runs
                regular_runs = [b for b in battles if not b.get("summary", {}).get("is_tournament", False)]
                tournament_runs = [b for b in battles if b.get("summary", {}).get("is_tournament", False)]

                # --- REGULAR RUNS: Multi-select with aggregates ---
                st.subheader("📊 Regular Run Analysis")

                if regular_runs:
                    run_options = {
                        f"#{b['id']} - Tier {b['summary']['tier']} Wave {b['summary']['wave']} ({b['summary']['battle_date']})": b
                        for b in regular_runs
                    }

                    selected_run_labels = st.multiselect(
                        "Select runs to analyze (multi-select)",
                        options=list(run_options.keys()),
                        default=list(run_options.keys())[:min(5, len(run_options))]  # Default to last 5
                    )

                    if selected_run_labels:
                        selected_runs = [run_options[label] for label in selected_run_labels]

                        # Calculate aggregate statistics
                        total_runs = len(selected_runs)
                        total_coins = 0
                        total_cells = 0
                        total_enemies = 0
                        total_gb_kills = 0
                        coins_per_hour_list = []
                        cells_per_hour_list = []
                        waves_list = []

                        for run in selected_runs:
                            values = run.get("values", {})
                            summary = run.get("summary", {})

                            # Coins earned
                            coins_val = values.get("coins_earned", {}).get("numeric", 0)
                            total_coins += coins_val

                            # Cells earned
                            cells_val = values.get("cells_earned", {}).get("numeric", 0)
                            total_cells += cells_val

                            # Enemies and GB kills
                            enemies_val = values.get("total_enemies", {}).get("numeric", 0)
                            total_enemies += enemies_val
                            gb_kills_val = values.get("golden_bot_kills", {}).get("numeric", 0)
                            total_gb_kills += gb_kills_val

                            # Coins per hour (for average)
                            cph = values.get("coins_per_hour", {}).get("numeric", 0)
                            if cph > 0:
                                coins_per_hour_list.append(cph)

                            # Cells per hour (for average)
                            cellph = values.get("cells_per_hour", {}).get("numeric", 0)
                            if cellph > 0:
                                cells_per_hour_list.append(cellph)

                            # Wave reached
                            wave_str = summary.get("wave", "0")
                            try:
                                wave_num = int(str(wave_str).replace(",", ""))
                                waves_list.append(wave_num)
                            except:
                                pass

                        # Calculate averages
                        avg_coins_per_hour = sum(coins_per_hour_list) / len(coins_per_hour_list) if coins_per_hour_list else 0
                        avg_cells_per_hour = sum(cells_per_hour_list) / len(cells_per_hour_list) if cells_per_hour_list else 0
                        avg_wave = sum(waves_list) / len(waves_list) if waves_list else 0
                        gb_kill_pct = (total_gb_kills / total_enemies * 100) if total_enemies > 0 else 0

                        # Format large numbers
                        def format_number(n):
                            if n >= 1e12:
                                return f"{n/1e12:.2f}T"
                            elif n >= 1e9:
                                return f"{n/1e9:.2f}B"
                            elif n >= 1e6:
                                return f"{n/1e6:.2f}M"
                            elif n >= 1e3:
                                return f"{n/1e3:.2f}K"
                            else:
                                return f"{n:.0f}"

                        # Display aggregate stats
                        st.markdown("#### Aggregate Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Runs", total_runs)
                        with col2:
                            st.metric("Total Coins", format_number(total_coins))
                        with col3:
                            st.metric("Total Cells", format_number(total_cells))
                        with col4:
                            st.metric("Avg Wave", f"{avg_wave:.0f}")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Coins/Hour", format_number(avg_coins_per_hour))
                        with col2:
                            st.metric("Avg Cells/Hour", format_number(avg_cells_per_hour))
                        with col3:
                            st.metric("Total Enemies", format_number(total_enemies))
                        with col4:
                            st.metric("Avg GB Kill %", f"{gb_kill_pct:.1f}%")

                        st.divider()
                else:
                    st.info("No regular runs recorded yet.")

                # --- TOURNAMENT RUNS: Single comparison ---
                st.subheader("🏆 Tournament Comparison")

                if len(tournament_runs) >= 2:
                    tournament_options = [
                        f"#{b['id']} - Tier {b['summary']['tier']} Wave {b['summary']['wave']} ({b['summary']['battle_date']})"
                        for b in tournament_runs
                    ]

                    col1, col2 = st.columns(2)
                    with col1:
                        prev_tournament = st.selectbox("Previous Tournament", tournament_options, index=1 if len(tournament_runs) > 1 else 0)
                    with col2:
                        curr_tournament = st.selectbox("Current Tournament", tournament_options, index=0)

                    st.divider()
                elif tournament_runs:
                    st.info("Need at least 2 tournament runs for comparison.")
                else:
                    st.info("No tournament runs recorded yet.")

                # Video generation options
                st.subheader("🎬 Video Options")
                col1, col2, col3 = st.columns(3)
                with col1:
                    include_tts = st.checkbox("🎙️ Include Voiceover", value=True)
                with col2:
                    use_kb = st.checkbox("🧠 Use Knowledge Base", value=True, help="Enhance scripts with game tips and context from the Knowledge Base")
                with col3:
                    auto_upload = st.checkbox("📤 Upload to YouTube", value=False)

                if st.button("🎬 Generate Video", type="primary"):
                    with st.spinner("Generating your Tower Journey video..."):
                        try:
                            # Initialize knowledge base if enabled
                            kb = None
                            if use_kb:
                                kb = KnowledgeBase()
                                kb_stats = kb.get_stats()
                                if kb_stats.get('total_chunks', 0) > 0:
                                    st.info(f"📚 Using Knowledge Base with {kb_stats['total_chunks']} chunks")
                                else:
                                    st.warning("Knowledge Base is empty. Generate video without KB enhancement.")
                                    kb = None

                            # Initialize video generator with KB
                            video_gen = JourneyVideoGenerator(knowledge_base=kb)

                            # Use selected runs if available
                            if 'selected_runs' in dir() and selected_runs:
                                battles_to_use = selected_runs
                            else:
                                battles_to_use = battles

                            # Get screenshots from battles if available
                            screenshots = []
                            for b in battles_to_use:
                                img_path = b.get("image_path")
                                if img_path and Path(img_path).exists():
                                    screenshots.append(img_path)

                            # Generate video
                            result = video_gen.generate_video(
                                battles=battles_to_use,
                                screenshots=screenshots,
                                include_tts=include_tts,
                                use_knowledge_base=use_kb
                            )

                            if result.get("error"):
                                st.error(f"Error: {result['error']}")
                            else:
                                st.success("Video generated successfully!")

                                # Show script
                                script = result.get("script", {})
                                with st.expander("📝 Generated Script", expanded=True):
                                    if script.get('kb_enhanced'):
                                        st.success("🧠 Script enhanced with Knowledge Base insights")

                                    st.markdown(f"**Intro:** {script.get('intro', '')}")
                                    st.markdown("**Stories:**")
                                    for i, story in enumerate(script.get('stories', []), 1):
                                        st.markdown(f"{i}. {story}")
                                    st.markdown(f"**Outro:** {script.get('outro', '')}")

                                    # Show featured tip if available
                                    if script.get('featured_tip'):
                                        st.info(f"💡 **Featured Tip:** {script['featured_tip']}")

                                # Show stats
                                stats = result.get("stats", {})
                                with st.expander("📊 Video Statistics"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Runs", stats.get('total_runs', 0))
                                        st.metric("Total Coins", stats.get('total_coins_formatted', '0'))
                                    with col2:
                                        st.metric("Total Cells", stats.get('total_cells_formatted', '0'))
                                        st.metric("Highest Wave", stats.get('max_wave_formatted', '0'))
                                    with col3:
                                        st.metric("GB Kill %", f"{stats.get('gb_kill_pct', 0):.1f}%")
                                        st.metric("Duration", f"{result.get('total_duration', 0):.1f}s")

                                # Download video
                                video_path = result.get("video_path")
                                if video_path and Path(video_path).exists():
                                    st.video(video_path)

                                    with open(video_path, "rb") as f:
                                        st.download_button(
                                            "📥 Download Video",
                                            data=f.read(),
                                            file_name=f"tower_journey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                            mime="video/mp4"
                                        )

                                    # YouTube upload option
                                    if auto_upload:
                                        st.info("YouTube upload feature coming soon!")
                                else:
                                    st.warning("Video path not found. Check console for errors.")

                        except Exception as e:
                            st.error(f"Failed to generate video: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        else:
            st.warning("No battle data. Add some battle reports first!")

    # TAB 5: Transcript Analysis
    with tab5:
        st.header("YouTube Transcript Analysis")
        st.markdown("Analyze Tower content creator videos to understand their structure and create templates.")

        transcript_analyzer = TranscriptAnalyzer()

        # Show saved analyses in sidebar
        saved_analyses = transcript_analyzer.list_saved_analyses()
        if saved_analyses:
            st.sidebar.subheader("Saved Topic Analyses")
            for analysis_name in saved_analyses:
                if st.sidebar.button(f"Load: {analysis_name}", key=f"load_{analysis_name}"):
                    loaded = transcript_analyzer.load_topic_analysis(analysis_name)
                    if loaded:
                        st.session_state["topic_analysis"] = loaded
                        st.session_state["channel_name"] = analysis_name
                        st.rerun()

        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["📺 Channel URL (Auto-fetch all videos)", "📋 Manual Video URLs"],
            horizontal=True
        )

        if input_method == "📺 Channel URL (Auto-fetch all videos)":
            st.subheader("Fetch Videos from Channel")

            channel_url = st.text_input(
                "YouTube Channel URL",
                placeholder="https://www.youtube.com/@AllClouded",
                help="Enter a YouTube channel URL. All videos will be fetched automatically."
            )

            # Row 1: Basic settings
            col1, col2, col3 = st.columns(3)
            with col1:
                max_videos = st.number_input("Max Videos to Fetch", min_value=10, max_value=200, value=50, help="Fetch up to this many videos before filtering")
            with col2:
                preferred_lang = st.selectbox("Preferred Language", ["en", "de", "es", "fr"], index=0, key="lang_channel")
            with col3:
                filter_mode = st.selectbox(
                    "Content Filter",
                    ["🎯 Tower Only (Keywords)", "🤖 Tower Only (AI)", "📺 All Videos"],
                    index=0,
                    help="Filter to only Tower-related content"
                )

            # Row 2: Filtering options (shown when filtering is enabled)
            if filter_mode != "📺 All Videos":
                col1, col2 = st.columns(2)
                with col1:
                    min_confidence = st.slider(
                        "Min Confidence",
                        min_value=0.3,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        help="Minimum confidence that video is Tower-related (higher = stricter)"
                    )
                with col2:
                    st.caption("🎯 Keywords: Fast, based on title matching")
                    st.caption("🤖 AI: Slower but more accurate")

            # Row 3: Whisper settings
            col1, col2 = st.columns(2)
            with col1:
                use_whisper = st.checkbox("Use Whisper Fallback", value=True, help="Use local Whisper transcription if YouTube captions unavailable")
            with col2:
                whisper_model = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium"], index=1, key="whisper_channel", help="Larger models are more accurate but slower")

            fetch_channel_btn = st.button("Fetch Channel Videos", type="primary")

            if fetch_channel_btn and channel_url:
                with st.spinner("Fetching video list from channel..."):
                    status_text = st.empty()

                    def update_status(msg):
                        status_text.text(msg)

                    # Fetch videos
                    videos = transcript_analyzer.get_channel_videos(
                        channel_url,
                        max_videos=max_videos,
                        progress_callback=update_status
                    )

                    if videos and "error" not in videos[0]:
                        total_fetched = len(videos)

                        # Apply filtering if enabled
                        if filter_mode != "📺 All Videos":
                            use_ai = "AI" in filter_mode
                            status_text.text(f"Filtering {total_fetched} videos for Tower content...")

                            videos = transcript_analyzer.filter_tower_videos(
                                videos,
                                min_confidence=min_confidence,
                                use_ai=use_ai,
                                progress_callback=update_status
                            )

                            st.success(f"Found {len(videos)} Tower-related videos (from {total_fetched} total)")

                            # Show filtering stats
                            if videos:
                                excluded = total_fetched - len(videos)
                                if excluded > 0:
                                    st.info(f"Filtered out {excluded} non-Tower videos")
                        else:
                            st.success(f"Found {len(videos)} videos!")

                        # Extract channel name from URL for saving
                        channel_name = channel_url.split("@")[-1].split("/")[0] if "@" in channel_url else "channel"
                        st.session_state["channel_name"] = channel_name

                        # Store in session state for selection
                        st.session_state["channel_videos"] = videos
                    else:
                        error_msg = videos[0].get("error", "Unknown error") if videos else "No videos found"
                        st.error(f"Error fetching channel: {error_msg}")

            # Show video selection if we have channel videos
            if "channel_videos" in st.session_state and st.session_state["channel_videos"]:
                videos = st.session_state["channel_videos"]

                st.subheader(f"Select Videos to Analyze ({len(videos)} available)")

                # Show relevance info if available
                has_relevance = any(v.get('relevance') for v in videos)
                if has_relevance:
                    with st.expander("📊 Relevance Details", expanded=False):
                        for v in videos:
                            rel = v.get('relevance', {})
                            if rel:
                                conf = rel.get('confidence', 0)
                                conf_bar = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
                                keywords = ", ".join(rel.get('matched_keywords', [])[:5])
                                st.caption(f"{conf_bar} **{v['title'][:40]}...** - {rel.get('reason', 'N/A')} {f'({keywords})' if keywords else ''}")

                # Multi-select for videos with confidence indicators
                video_options = {}
                for v in videos:
                    if v.get('video_id'):
                        rel = v.get('relevance', {})
                        conf = rel.get('confidence', 0)
                        if rel and conf > 0:
                            conf_icon = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
                            label = f"{conf_icon} {v['title'][:45]}... ({v['duration_formatted']})"
                        else:
                            label = f"{v['title'][:50]}... ({v['duration_formatted']})"
                        video_options[label] = v

                selected_labels = st.multiselect(
                    "Select videos",
                    options=list(video_options.keys()),
                    default=list(video_options.keys())  # All videos selected by default
                )

                if selected_labels and st.button("Analyze Selected Videos", type="primary"):
                    selected_videos = [video_options[label] for label in selected_labels]

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_transcript_status(msg):
                        status_text.text(msg)

                    success_count = 0
                    for i, video in enumerate(selected_videos):
                        vid = video['video_id']
                        status_text.text(f"Fetching transcript {i+1}/{len(selected_videos)}: {video['title'][:30]}...")
                        progress_bar.progress((i + 1) / len(selected_videos))

                        transcript = transcript_analyzer.fetch_transcript_with_fallback(
                            vid,
                            languages=[preferred_lang, 'en'],
                            use_whisper=use_whisper,
                            whisper_model=whisper_model,
                            progress_callback=update_transcript_status
                        )

                        if "error" not in transcript:
                            # Add title to transcript
                            transcript["title"] = video.get("title", "Unknown")
                            analysis = transcript_analyzer.analyze_structure(transcript)

                            if "transcripts" not in st.session_state:
                                st.session_state["transcripts"] = {}
                            st.session_state["transcripts"][vid] = {
                                "transcript": transcript,
                                "analysis": analysis
                            }
                            success_count += 1
                            # Show transcription method
                            method = transcript.get("transcription_method", "unknown")
                            if method == "whisper":
                                st.info(f"📝 Transcribed with Whisper: {video['title'][:40]}...")
                        else:
                            st.warning(f"No transcript: {video['title'][:40]}...")

                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"Successfully analyzed {success_count}/{len(selected_videos)} videos!")
                    st.rerun()

        else:  # Manual video URLs
            st.subheader("Add Videos Manually")

            video_input = st.text_area(
                "Enter YouTube URLs or Video IDs (one per line)",
                height=150,
                placeholder="https://www.youtube.com/watch?v=xxxxx\nhttps://youtu.be/yyyyy\nzzzzzzzzzzz",
                help="Paste YouTube URLs or video IDs. The transcript will be automatically fetched."
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                preferred_lang = st.selectbox("Preferred Language", ["en", "de", "es", "fr"], index=0, key="lang_manual")
            with col2:
                use_whisper_manual = st.checkbox("Use Whisper Fallback", value=True, key="whisper_manual_check", help="Use local Whisper transcription if YouTube captions unavailable")
            with col3:
                whisper_model_manual = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium"], index=1, key="whisper_manual", help="Larger models are more accurate but slower")
            with col4:
                analyze_btn = st.button("Fetch & Analyze Transcripts", type="primary")

            if analyze_btn and video_input:
                video_ids = [line.strip() for line in video_input.strip().split('\n') if line.strip()]

                if video_ids:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_manual_status(msg):
                        status_text.text(msg)

                    all_transcripts = []
                    all_analyses = []

                    for i, vid in enumerate(video_ids):
                        status_text.text(f"Fetching transcript {i+1}/{len(video_ids)}: {vid[:30]}...")
                        progress_bar.progress((i + 1) / len(video_ids))

                        transcript = transcript_analyzer.fetch_transcript_with_fallback(
                            vid,
                            languages=[preferred_lang, 'en'],
                            use_whisper=use_whisper_manual,
                            whisper_model=whisper_model_manual,
                            progress_callback=update_manual_status
                        )

                        if "error" not in transcript:
                            all_transcripts.append(transcript)
                            analysis = transcript_analyzer.analyze_structure(transcript)
                            all_analyses.append(analysis)

                            # Store in session state
                            if "transcripts" not in st.session_state:
                                st.session_state["transcripts"] = {}
                            st.session_state["transcripts"][transcript["video_id"]] = {
                                "transcript": transcript,
                                "analysis": analysis
                            }

                            # Show transcription method
                            method = transcript.get("transcription_method", "unknown")
                            if method == "whisper":
                                st.info(f"📝 Transcribed with Whisper: {vid[:40]}...")
                        else:
                            st.warning(f"Could not fetch: {vid} - {transcript.get('error', 'Unknown error')}")

                    progress_bar.empty()
                    status_text.empty()

                    if all_analyses:
                        st.success(f"Successfully analyzed {len(all_analyses)} video(s)!")

        # Display cached transcripts
        st.divider()
        st.subheader("Analyzed Videos")

        if "transcripts" in st.session_state and st.session_state["transcripts"]:
            for video_id, data in st.session_state["transcripts"].items():
                transcript = data["transcript"]
                analysis = data["analysis"]

                title = transcript.get("title", video_id)
                method = transcript.get("transcription_method", "youtube_api")
                method_icon = "🎙️" if method == "whisper" else "📺"
                with st.expander(f"{method_icon} {title} ({transcript.get('duration_formatted', '?')})"):
                    # Quick stats
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Duration", transcript.get("duration_formatted", "?"))
                    with col2:
                        st.metric("Words", transcript.get("word_count", 0))
                    with col3:
                        wpm = analysis.get("estimated_words_per_minute", 0)
                        st.metric("Words/Min", f"{wpm:.0f}")
                    with col4:
                        st.metric("Segments", len(transcript.get("segments", [])))
                    with col5:
                        method_label = "Whisper" if method == "whisper" else "YouTube"
                        st.metric("Source", method_label)

                    # Structure analysis
                    st.markdown("#### Video Structure")
                    structure = analysis.get("structure", {})

                    struct_col1, struct_col2, struct_col3 = st.columns(3)
                    with struct_col1:
                        has_hook = structure.get("has_intro_hook", False)
                        st.markdown(f"**Intro Hook:** {'Yes' if has_hook else 'No'}")
                    with struct_col2:
                        has_cta = structure.get("has_call_to_action", False)
                        st.markdown(f"**Call to Action:** {'Yes' if has_cta else 'No'}")
                    with struct_col3:
                        has_outro = structure.get("has_outro", False)
                        st.markdown(f"**Outro:** {'Yes' if has_outro else 'No'}")

                    # Tower keywords
                    keywords = analysis.get("tower_keywords", [])
                    if keywords:
                        st.markdown("#### Tower Keywords Mentioned")
                        kw_text = ", ".join([f"**{k['keyword']}** ({k['count']})" for k in keywords[:10]])
                        st.markdown(kw_text)

                    # Intro preview
                    if structure.get("intro_preview"):
                        st.markdown("#### Intro Preview (first 60s)")
                        st.text_area("", structure["intro_preview"], height=100, disabled=True, key=f"intro_{video_id}")

                    # Full transcript
                    with st.expander("View Full Transcript"):
                        st.text_area("", transcript.get("full_text", ""), height=300, disabled=True, key=f"full_{video_id}")

                    # Segments
                    with st.expander("View Segments (by minute)"):
                        for seg in transcript.get("segments", []):
                            st.markdown(f"**[{seg['start_formatted']}]** {seg['text'][:200]}...")

            # Comparison section
            if len(st.session_state["transcripts"]) >= 2:
                st.divider()
                st.subheader("Video Comparison & Template Extraction")

                if st.button("Compare All Videos", type="secondary"):
                    all_analyses = [d["analysis"] for d in st.session_state["transcripts"].values()]

                    # Aggregate stats
                    avg_wpm = sum(a.get("estimated_words_per_minute", 0) for a in all_analyses) / len(all_analyses)
                    avg_words = sum(a.get("word_count", 0) for a in all_analyses) / len(all_analyses)

                    intro_hook_pct = sum(1 for a in all_analyses if a.get("structure", {}).get("has_intro_hook")) / len(all_analyses) * 100
                    cta_pct = sum(1 for a in all_analyses if a.get("structure", {}).get("has_call_to_action")) / len(all_analyses) * 100
                    outro_pct = sum(1 for a in all_analyses if a.get("structure", {}).get("has_outro")) / len(all_analyses) * 100

                    st.markdown("#### Aggregate Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Words/Min", f"{avg_wpm:.0f}")
                    with col2:
                        st.metric("Avg Word Count", f"{avg_words:.0f}")
                    with col3:
                        st.metric("Videos Analyzed", len(all_analyses))

                    st.markdown("#### Common Patterns")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("% with Intro Hook", f"{intro_hook_pct:.0f}%")
                    with col2:
                        st.metric("% with CTA", f"{cta_pct:.0f}%")
                    with col3:
                        st.metric("% with Outro", f"{outro_pct:.0f}%")

                    # Aggregate keywords
                    all_keywords = {}
                    for a in all_analyses:
                        for kw in a.get("tower_keywords", []):
                            all_keywords[kw["keyword"]] = all_keywords.get(kw["keyword"], 0) + kw["count"]

                    if all_keywords:
                        st.markdown("#### Most Common Keywords Across All Videos")
                        top_kw = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:15]
                        kw_text = ", ".join([f"**{k}** ({v})" for k, v in top_kw])
                        st.markdown(kw_text)

                    # Template suggestion
                    st.markdown("#### Suggested Video Template")
                    template = f"""
**Based on {len(all_analyses)} analyzed videos:**

1. **Intro Hook** (0:00 - 0:15)
   - {'Most videos use a hook' if intro_hook_pct > 50 else 'Many videos skip the hook'}
   - Typical: "Hey everyone, today we're going to..."

2. **Main Content** (0:15 - end-1:00)
   - Average speaking pace: {avg_wpm:.0f} words per minute
   - Focus keywords: {', '.join([k for k, v in top_kw[:5]])}

3. **Call to Action** (throughout or end)
   - {cta_pct:.0f}% of videos include subscribe/like prompts

4. **Outro** (last ~30 seconds)
   - {'Common pattern' if outro_pct > 50 else 'Often skipped'}: Thanks for watching, see you next time
"""
                    st.markdown(template)

                # AI Topic Analysis section
                st.divider()
                st.subheader("AI Topic Analysis")
                st.markdown("*Use GPT to extract actual topics, themes, and content categories from the videos.*")

                if st.button("Analyze Topics with AI", type="primary", key="ai_topic_btn"):
                    with st.spinner("Analyzing video topics with AI..."):
                        all_transcripts = [d["transcript"] for d in st.session_state["transcripts"].values()]

                        # Get titles if available from channel videos
                        video_titles = []
                        channel_videos = st.session_state.get("channel_videos", [])
                        video_title_map = {v.get("video_id", ""): v.get("title", "") for v in channel_videos if "video_id" in v}

                        for t in all_transcripts:
                            vid = t.get("video_id", "")
                            title = video_title_map.get(vid, t.get("title", vid))
                            video_titles.append(title)

                        progress_text = st.empty()

                        def update_progress(msg):
                            progress_text.text(msg)

                        topic_analysis = transcript_analyzer.analyze_channel_topics(
                            all_transcripts,
                            video_titles,
                            progress_callback=update_progress
                        )

                        progress_text.empty()

                        # Store in session state
                        st.session_state["topic_analysis"] = topic_analysis

                        # Save to disk with channel name
                        channel_name = st.session_state.get("channel_name", "unknown_channel")
                        transcript_analyzer.save_topic_analysis(channel_name, topic_analysis)
                        st.success(f"Topic analysis saved to data/journey/topic_analysis/")

                # Display topic analysis results
                if "topic_analysis" in st.session_state:
                    topic_data = st.session_state["topic_analysis"]

                    if "error" not in topic_data:
                        st.success(f"AI analyzed {topic_data.get('videos_analyzed', 0)} videos!")

                        # Content Types Distribution
                        st.markdown("#### Content Types")
                        content_types = topic_data.get("content_types", [])
                        if content_types:
                            ct_cols = st.columns(min(len(content_types), 4))
                            for i, (ct, count) in enumerate(content_types[:4]):
                                with ct_cols[i]:
                                    st.metric(ct, count)

                        # Target Audiences
                        st.markdown("#### Target Audiences")
                        audiences = topic_data.get("target_audiences", {})
                        if audiences:
                            aud_text = ", ".join([f"**{k}**: {v}" for k, v in audiences.items()])
                            st.markdown(aud_text)

                        # Top Topics
                        st.markdown("#### Most Common Topics Across Videos")
                        top_topics = topic_data.get("top_topics", [])
                        if top_topics:
                            topic_list = ""
                            for topic, count in top_topics[:15]:
                                topic_list += f"- **{topic.title()}** (mentioned in {count} videos)\n"
                            st.markdown(topic_list)

                        # Game Features Covered
                        st.markdown("#### Game Features/Mechanics Covered")
                        features = topic_data.get("game_features", [])
                        if features:
                            feature_text = ", ".join([f"**{f[0].title()}** ({f[1]})" for f in features[:12]])
                            st.markdown(feature_text)

                        # Sample Tips
                        st.markdown("#### Sample Tips from Videos")
                        tips = topic_data.get("sample_tips", [])
                        if tips:
                            for tip in tips[:10]:
                                st.markdown(f"- {tip}")

                        # Individual Video Analyses
                        st.markdown("#### Individual Video Topic Breakdown")
                        individual = topic_data.get("individual_analyses", [])
                        for analysis in individual:
                            title = analysis.get("video_title", analysis.get("video_id", "Unknown"))
                            with st.expander(f"{title[:60]}..."):
                                st.markdown(f"**Content Type:** {analysis.get('content_type', '?')}")
                                st.markdown(f"**Target Audience:** {analysis.get('target_audience', '?')}")
                                st.markdown(f"**Main Topic:** {analysis.get('main_topic', '?')}")

                                topics = analysis.get("topics", [])
                                if topics:
                                    st.markdown("**Topics:**")
                                    for t in topics:
                                        st.markdown(f"- {t}")

                                takeaways = analysis.get("key_takeaways", [])
                                if takeaways:
                                    st.markdown("**Key Takeaways:**")
                                    for t in takeaways:
                                        st.markdown(f"- {t}")

                                questions = analysis.get("questions_answered", [])
                                if questions:
                                    st.markdown("**Questions Answered:**")
                                    for q in questions:
                                        st.markdown(f"- {q}")

        else:
            st.info("No videos analyzed yet. Add some YouTube URLs above to get started!")

        # Clear cache button
        if st.button("Clear Transcript Cache", type="secondary"):
            st.session_state["transcripts"] = {}
            cache_dir = Path("data/journey/transcripts")
            if cache_dir.exists():
                for f in cache_dir.glob("*.json"):
                    f.unlink()
            st.success("Cache cleared!")
            st.rerun()

    # TAB 6: Video Template Generator
    with tab6:
        st.header("Video Template Generator")
        st.markdown("Generate improved video templates based on analyzed creator content patterns.")

        transcript_analyzer = TranscriptAnalyzer()

        # Check for saved analyses
        saved_analyses = transcript_analyzer.list_saved_analyses()

        if saved_analyses:
            st.success(f"Found {len(saved_analyses)} analyzed channel(s): {', '.join(saved_analyses)}")

            # Select which analysis to use
            selected_analysis = st.selectbox(
                "Select channel analysis to use",
                options=saved_analyses,
                index=0
            )

            if st.button("Generate Video Template", type="primary"):
                with st.spinner("Generating template from analyzed patterns..."):
                    # Load the selected analysis
                    topic_analysis = transcript_analyzer.load_topic_analysis(selected_analysis)

                    if topic_analysis:
                        template = transcript_analyzer.generate_video_template(topic_analysis)
                        st.session_state["video_template"] = template
                        st.success("Template generated!")

            # Display the template
            if "video_template" in st.session_state:
                template = st.session_state["video_template"]

                st.divider()
                st.subheader(template.get("template_name", "Tower Progress Video"))
                st.caption(f"Based on: {template.get('based_on', 'analyzed videos')}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recommended Duration", template.get("recommended_duration", "15-20 min"))
                with col2:
                    st.metric("Sections", len(template.get("sections", [])))

                # Display sections
                st.markdown("---")
                st.markdown("### Video Sections")

                for section in template.get("sections", []):
                    with st.expander(f"**{section.get('name', 'Section')}** ({section.get('duration', '')})"):
                        if section.get("purpose"):
                            st.markdown(f"**Purpose:** {section['purpose']}")

                        if section.get("content_suggestions"):
                            st.markdown("**Content to Cover:**")
                            for item in section["content_suggestions"]:
                                if isinstance(item, dict):
                                    st.markdown(f"- {item.get('name', item)}")
                                else:
                                    st.markdown(f"- {item}")

                        if section.get("tips_to_include"):
                            st.markdown("**Tips to Include:**")
                            for tip in section["tips_to_include"]:
                                st.markdown(f"- 💡 {tip}")

                        if section.get("example"):
                            st.markdown("**Example Script:**")
                            st.info(section["example"])

                # Content Pillars
                st.markdown("---")
                st.markdown("### Content Pillars")
                pillars = template.get("content_pillars", [])
                if pillars:
                    cols = st.columns(len(pillars))
                    for i, pillar in enumerate(pillars):
                        with cols[i]:
                            st.markdown(f"**{pillar.get('name', 'Pillar')}**")
                            st.caption(f"*{pillar.get('frequency', '')}*")
                            for element in pillar.get("elements", []):
                                st.markdown(f"- {element}")

                # Recurring Topics
                st.markdown("---")
                st.markdown("### Recurring Topics")
                recurring = template.get("recurring_topics", [])
                if recurring:
                    for topic_group in recurring[:6]:
                        st.markdown(f"**{topic_group.get('name', 'Topic')}** ({topic_group.get('suggested_duration', '')})")
                        topics = topic_group.get("topics", [])
                        if topics:
                            st.caption(", ".join(topics))

                # Engagement Elements
                st.markdown("---")
                st.markdown("### Engagement Elements")
                engagement = template.get("engagement_elements", [])
                if engagement:
                    for elem in engagement:
                        st.markdown(f"- ✨ {elem}")

                # Visual Suggestions
                st.markdown("---")
                st.markdown("### Visual Suggestions")
                visuals = template.get("visual_suggestions", [])
                if visuals:
                    for vis in visuals:
                        st.markdown(f"- 🎥 {vis}")

                # Generate Script Outline
                st.markdown("---")
                st.markdown("### Generate Script Outline")

                # Option to include battle data
                history_path = Path("data/journey/battle_history.json")
                battle_data = None
                if history_path.exists():
                    with open(history_path, 'r') as f:
                        battle_history = json.load(f)
                    battles = battle_history.get("battles", [])
                    if battles:
                        include_battle = st.checkbox("Include latest battle data in script", value=True)
                        if include_battle:
                            battle_data = battles[0]

                if st.button("Generate Full Script Outline", type="secondary"):
                    outline = transcript_analyzer.generate_script_outline(battle_data, template)
                    st.markdown("---")
                    st.markdown("### Script Outline")
                    st.markdown(outline)

                    # Download button
                    st.download_button(
                        label="Download Script Outline",
                        data=outline,
                        file_name="tower_video_script.md",
                        mime="text/markdown"
                    )

        else:
            st.warning("No channel analysis data found. Please analyze some creator videos first in the 'Transcript Analysis' tab.")

            st.markdown("""
            ### How to use:
            1. Go to the **Transcript Analysis** tab
            2. Enter a YouTube channel URL (e.g., @AllClouded, @EthanDX)
            3. Analyze the videos to extract topics and patterns
            4. Return here to generate a template based on the analysis
            """)

    # TAB 7: Knowledge Base
    with tab7:
        st.header("Tower Game Knowledge Base")
        st.markdown("Ask questions about The Tower game mechanics, strategies, and terminology.")

        # Initialize knowledge base
        kb = KnowledgeBase()

        # Sidebar: KB Stats
        stats = kb.get_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chunks", stats['total_chunks'])
        with col2:
            st.metric("Embeddings", stats['total_embeddings'])
        with col3:
            st.metric("Sources", len(stats.get('sources', [])))
        with col4:
            status = "Ready" if stats['embeddings_complete'] else "Needs Rebuild"
            st.metric("Status", status)

        st.divider()

        # Question input
        st.subheader("Ask a Question")
        question = st.text_input(
            "Your question about The Tower game",
            placeholder="How do I improve my coins per hour?",
            key="kb_question"
        )

        if st.button("Ask", type="primary") and question:
            with st.spinner("Searching knowledge base..."):
                result = kb.ask(question)

                st.markdown("### Answer")
                st.markdown(result['answer'])

                confidence = result.get('confidence', 0)
                if confidence > 0:
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

                # Show sources
                sources = result.get('sources', [])
                if sources:
                    with st.expander(f"Sources ({len(sources)} relevant chunks)"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**{i}. {source['source']}** (similarity: {source['similarity']:.2%})")
                            st.markdown(f"*Type: {source['type']}*")
                            st.text(source['excerpt'])
                            st.divider()

        st.divider()

        # Knowledge Base Management
        st.subheader("Manage Knowledge Base")

        # Row 1: Basic imports
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Import from Video Analyses"):
                with st.spinner("Importing from topic analyses..."):
                    count = kb.import_from_topic_analyses()
                    st.success(f"Imported {count} chunks from video analyses")
                    st.rerun()

        with col2:
            if st.button("Add Default Tower Knowledge"):
                with st.spinner("Adding game knowledge..."):
                    count = kb.add_manual_knowledge(TOWER_GAME_KNOWLEDGE)
                    st.success(f"Added {count} knowledge chunks")
                    st.rerun()

        with col3:
            if st.button("Build Embeddings"):
                with st.spinner("Building embeddings (this may take a moment)..."):
                    count = kb.build_embeddings()
                    st.success(f"Created {count} embeddings")
                    st.rerun()

        # Row 2: External sources
        st.markdown("#### Import from External Sources")
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Reddit r/TheTowerGame"):
                reddit_limit = st.slider("Number of posts", 10, 200, 50, key="reddit_limit")
                reddit_sort = st.selectbox("Sort by", ["top", "hot", "new"], key="reddit_sort")
                reddit_time = st.selectbox("Time filter", ["all", "year", "month", "week"], key="reddit_time")
                include_comments = st.checkbox("Include comments", value=True, key="reddit_comments")

                if st.button("Import from Reddit"):
                    with st.spinner(f"Fetching {reddit_limit} posts from Reddit..."):
                        count = kb.import_from_reddit(
                            sort=reddit_sort,
                            time_filter=reddit_time,
                            limit=reddit_limit,
                            include_comments=include_comments
                        )
                        st.success(f"Imported {count} chunks from Reddit")
                        st.rerun()

        with col2:
            with st.expander("Fandom Wiki"):
                import_all_fandom = st.checkbox("Import ALL pages (entire wiki)", key="import_all_fandom")

                if not import_all_fandom:
                    fandom_limit = st.slider("Number of pages", 10, 500, 100, key="fandom_limit")
                else:
                    fandom_limit = 2000  # High limit for full wiki
                    st.info("Will import all pages including subpages")

                if st.button("Import from Fandom"):
                    with st.spinner(f"Fetching {'ALL' if import_all_fandom else fandom_limit} pages from Fandom wiki..."):
                        count = kb.import_from_fandom(limit=fandom_limit)
                        st.success(f"Imported {count} chunks from Fandom wiki")
                        st.rerun()

        # Row 3: Notion (requires Playwright)
        with st.expander("Notion Wiki (requires Playwright)"):
            st.info("Notion pages require JavaScript rendering. Install Playwright: `pip install playwright && playwright install chromium`")
            notion_url = st.text_input(
                "Notion page URL",
                value="https://the-tower.notion.site/",
                key="notion_url"
            )

            if st.button("Import from Notion"):
                with st.spinner("Fetching from Notion wiki..."):
                    urls = [notion_url] if notion_url else None
                    count = kb.import_from_notion(urls=urls)
                    if count > 0:
                        st.success(f"Imported {count} chunks from Notion")
                    else:
                        st.warning("No content found. Make sure Playwright is installed.")
                    st.rerun()

        # Show chunks by type
        if stats['total_chunks'] > 0:
            st.subheader("Knowledge by Type")
            for doc_type, count in stats.get('chunks_by_type', {}).items():
                st.markdown(f"- **{doc_type}**: {count} chunks")

        # Add custom knowledge
        st.subheader("Add Custom Knowledge")
        with st.expander("Add your own knowledge"):
            custom_title = st.text_input("Title (optional)", key="kb_custom_title")
            custom_type = st.selectbox(
                "Type",
                ["tips", "mechanics", "strategy", "terminology", "other"],
                key="kb_custom_type"
            )
            custom_content = st.text_area(
                "Content",
                placeholder="Enter knowledge about The Tower game...",
                height=150,
                key="kb_custom_content"
            )

            if st.button("Add to Knowledge Base"):
                if custom_content:
                    count = kb.add_manual_knowledge([{
                        "title": custom_title,
                        "type": custom_type,
                        "content": custom_content
                    }])
                    if count > 0:
                        st.success("Knowledge added! Remember to rebuild embeddings.")
                    else:
                        st.warning("Content was too short or already exists.")
                else:
                    st.warning("Please enter some content.")

        # Danger zone
        with st.expander("Danger Zone"):
            st.warning("These actions cannot be undone!")
            if st.button("Clear All Knowledge", type="secondary"):
                kb.clear()
                st.success("Knowledge base cleared.")
                st.rerun()

    # TAB 8: Tower News Pipeline
    with tab8:
        st.header("📺 Tower News Pipeline")
        st.markdown("Generate daily news videos from Reddit r/TheTowerGame posts.")

        # Load news config
        try:
            news_config = load_news_config()
        except Exception as e:
            st.error(f"Could not load news config: {e}")
            news_config = {}

        # Show current config
        col1, col2, col3 = st.columns(3)
        with col1:
            reddit_config = news_config.get("sources", {}).get("reddit", {})
            st.metric("Subreddit", f"r/{reddit_config.get('subreddits', ['TheTowerGame'])[0]}")
        with col2:
            st.metric("Min Upvotes", reddit_config.get("min_upvotes", 5))
        with col3:
            st.metric("Min Comments", reddit_config.get("min_comments", 3))

        st.divider()

        # Preview Reddit Posts section
        st.subheader("📰 Preview Reddit Posts")
        st.markdown("See what posts are currently available for the news video.")

        if st.button("Fetch Latest Posts", key="fetch_reddit"):
            with st.spinner("Fetching posts from Reddit..."):
                try:
                    scraper = RedditScraper(news_config)
                    result = scraper.run()
                    posts = result.get("posts", [])

                    if posts:
                        st.success(f"Found {len(posts)} posts that meet criteria")
                        st.session_state["news_posts"] = posts

                        for i, post in enumerate(posts, 1):
                            with st.expander(f"{i}. {post.get('title', 'No title')[:60]}...", expanded=i == 1):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Score", post.get("score", 0))
                                with col2:
                                    st.metric("Comments", post.get("num_comments", 0))
                                with col3:
                                    st.metric("Flair", post.get("flair", "None"))

                                st.markdown(f"**URL:** [Reddit Link]({post.get('url', '#')})")

                                if post.get("selftext"):
                                    st.markdown("**Content:**")
                                    st.text(post.get("selftext", "")[:500] + "..." if len(post.get("selftext", "")) > 500 else post.get("selftext", ""))

                                if post.get("image_url"):
                                    st.markdown(f"**Image:** [Link]({post.get('image_url')})")

                                if post.get("top_comments"):
                                    st.markdown("**Top Comments:**")
                                    for j, comment in enumerate(post.get("top_comments", [])[:3], 1):
                                        st.caption(f"{j}. {comment[:200]}...")
                    else:
                        st.warning("No posts found. Try adjusting the filter settings.")

                except Exception as e:
                    st.error(f"Error fetching posts: {e}")

        st.divider()

        # Generate News Video section
        st.subheader("🎬 Generate News Video")

        col1, col2 = st.columns(2)
        with col1:
            auto_upload = st.checkbox(
                "Upload to YouTube after generation",
                value=news_config.get("youtube", {}).get("auto_upload", False),
                key="news_auto_upload"
            )
        with col2:
            privacy_status = st.selectbox(
                "YouTube Privacy",
                ["public", "unlisted", "private"],
                index=["public", "unlisted", "private"].index(
                    news_config.get("youtube", {}).get("privacy_status", "public")
                ),
                key="news_privacy"
            )

        # History info
        history_path = Path("data/news_history.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                news_history = json.load(f)
            reported_count = len(news_history.get("reported_posts", []))
            st.info(f"📋 {reported_count} posts already reported (will be skipped)")

        if st.button("🚀 Generate Tower News Video", type="primary", key="generate_news"):
            # Initialize progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.container()

            with st.spinner("Generating Tower News video..."):
                try:
                    import asyncio
                    import sys
                    from io import StringIO

                    # Capture print output
                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()

                    status_text.text("Stage 1: Initializing pipeline...")
                    progress_bar.progress(10)

                    # Create pipeline
                    pipeline = NewsPipeline()

                    # Override YouTube settings based on UI
                    pipeline.config["youtube"]["auto_upload"] = auto_upload
                    pipeline.config["youtube"]["privacy_status"] = privacy_status

                    status_text.text("Stage 2: Running pipeline...")
                    progress_bar.progress(20)

                    # Run the pipeline
                    result = asyncio.run(pipeline.run())

                    # Restore stdout and get output
                    sys.stdout = old_stdout
                    log_output = captured_output.getvalue()

                    progress_bar.progress(100)

                    if result.get("success"):
                        status_text.text("✅ Video generated successfully!")
                        st.success("Tower News video generated!")

                        # Show results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 📹 Video Generated")
                            video_path = result.get("video_path")
                            if video_path and Path(video_path).exists():
                                st.video(video_path)

                                # Download button
                                with open(video_path, "rb") as f:
                                    st.download_button(
                                        "📥 Download Video",
                                        data=f.read(),
                                        file_name=f"tower_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                        mime="video/mp4"
                                    )

                        with col2:
                            st.markdown("### 📊 Pipeline Results")

                            # Show stages
                            stages = result.get("stages", {})
                            for stage_name, stage_data in stages.items():
                                success = stage_data.get("success", False)
                                icon = "✅" if success else "❌"
                                st.markdown(f"{icon} **{stage_name.replace('_', ' ').title()}**")

                            # YouTube URL if uploaded
                            if result.get("youtube_url"):
                                st.markdown("### 📺 YouTube")
                                st.markdown(f"[Watch on YouTube]({result.get('youtube_url')})")
                                st.code(result.get("youtube_url"))

                        # Show metadata
                        metadata_path = result.get("metadata_path")
                        if metadata_path and Path(metadata_path).exists():
                            with st.expander("📝 Video Metadata"):
                                with open(metadata_path, 'r', encoding='utf-8', errors='replace') as f:
                                    st.text(f.read())

                        # Show log
                        with st.expander("📋 Pipeline Log"):
                            st.text(log_output)

                    else:
                        status_text.text("❌ Pipeline failed")
                        st.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")

                        # Show partial results
                        with st.expander("Pipeline Details"):
                            st.json(result)

                        # Show log
                        with st.expander("📋 Pipeline Log"):
                            st.text(log_output)

                except Exception as e:
                    sys.stdout = old_stdout
                    progress_bar.progress(100)
                    status_text.text("❌ Error occurred")
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        st.divider()

        # Recent Videos section
        st.subheader("📁 Recent Tower News Videos")

        output_dir = Path("output")
        if output_dir.exists():
            # Find recent video directories
            video_dirs = sorted(output_dir.glob("*/"), key=lambda x: x.name, reverse=True)[:7]

            if video_dirs:
                for video_dir in video_dirs:
                    # Find video files
                    video_files = list(video_dir.glob("news_*.mp4"))
                    if video_files:
                        latest_video = sorted(video_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                        file_size_mb = latest_video.stat().st_size / (1024 * 1024)

                        with st.expander(f"📅 {video_dir.name} - {latest_video.name} ({file_size_mb:.1f} MB)"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.video(str(latest_video))
                            with col2:
                                # Check for metadata
                                metadata_files = list(video_dir.glob("metadata_*.txt"))
                                if metadata_files:
                                    with open(metadata_files[0], 'r', encoding='utf-8', errors='replace') as f:
                                        metadata_content = f.read()
                                    st.text_area("Metadata", metadata_content, height=200, key=f"meta_{video_dir.name}")

                                # Download button
                                with open(latest_video, "rb") as f:
                                    st.download_button(
                                        "📥 Download",
                                        data=f.read(),
                                        file_name=latest_video.name,
                                        mime="video/mp4",
                                        key=f"dl_{video_dir.name}"
                                    )
            else:
                st.info("No Tower News videos generated yet.")
        else:
            st.info("No output directory found.")

        st.divider()

        # Configuration section
        with st.expander("⚙️ Configuration"):
            st.markdown("Current configuration from `config/config.yaml`:")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Reddit Settings:**")
                st.json(news_config.get("sources", {}).get("reddit", {}))

            with col2:
                st.markdown("**YouTube Settings:**")
                youtube_config = news_config.get("youtube", {})
                st.json({
                    "auto_upload": youtube_config.get("auto_upload", False),
                    "privacy_status": youtube_config.get("privacy_status", "private")
                })

            st.markdown("**Channel:**")
            st.json(news_config.get("channel", {}))

            st.caption("Edit `config/config.yaml` to change these settings.")


    # TAB 9: Progress Tracker
    with tab9:
        st.header("Progress Tracker")
        st.markdown("Track your Tower game progress over time by uploading screenshots. Compare values and see how you've developed!")

        # Initialize progress tracker
        progress_tracker = ProgressTracker()

        # Show current progress summary in sidebar-like section
        summary = progress_tracker.get_overall_progress_summary()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Snapshots", summary.get("total_snapshots", 0))
        with col2:
            st.metric("Categories Tracked", len(summary.get("categories_tracked", [])))
        with col3:
            if summary.get("first_snapshot"):
                st.metric("First Snapshot", summary["first_snapshot"][:10])
            else:
                st.metric("First Snapshot", "None")
        with col4:
            if summary.get("last_snapshot"):
                st.metric("Last Snapshot", summary["last_snapshot"][:10])
            else:
                st.metric("Last Snapshot", "None")

        st.divider()

        # Two columns: Upload and History
        upload_col, history_col = st.columns([1, 1])

        with upload_col:
            st.subheader("Upload Screenshot")

            # Category selector
            category_options = {
                "auto": "Auto-detect",
                **{k: v["name"] for k, v in ProgressTracker.PROGRESS_CATEGORIES.items()}
            }
            selected_category = st.selectbox(
                "Category (optional)",
                options=list(category_options.keys()),
                format_func=lambda x: category_options[x],
                help="Select a category or leave as auto-detect"
            )

            # File uploader
            uploaded_file = st.file_uploader(
                "Upload a screenshot",
                type=["png", "jpg", "jpeg"],
                key="progress_upload"
            )

            # Clipboard paste
            if st.button("Paste from Clipboard", key="progress_paste"):
                try:
                    clipboard_image = ImageGrab.grabclipboard()
                    if clipboard_image:
                        st.session_state["progress_clipboard_image"] = clipboard_image
                        st.success("Image pasted from clipboard!")
                        st.rerun()
                    else:
                        st.warning("No image found in clipboard.")
                except Exception as e:
                    st.error(f"Could not access clipboard: {e}")

            # Display and process image
            image_to_process = None

            if "progress_clipboard_image" in st.session_state:
                image_to_process = st.session_state["progress_clipboard_image"]
                st.image(image_to_process, caption="Clipboard Image", use_container_width=True)
            elif uploaded_file:
                image_to_process = Image.open(uploaded_file)
                st.image(image_to_process, caption="Uploaded Image", use_container_width=True)

            # Notes input
            notes = st.text_input("Notes (optional)", placeholder="e.g., After grinding for 2 hours...")

            if image_to_process:
                if st.button("Analyze & Save Progress", type="primary", key="analyze_progress"):
                    with st.spinner("Analyzing screenshot..."):
                        # Save temp image
                        temp_path = Path("data/journey/temp_progress.png")
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        image_to_process.save(temp_path)

                        # Analyze and save
                        category_hint = None if selected_category == "auto" else selected_category
                        result = progress_tracker.add_snapshot(
                            str(temp_path),
                            category=category_hint,
                            notes=notes,
                            save_screenshot=True
                        )

                        # Clear clipboard image
                        if "progress_clipboard_image" in st.session_state:
                            del st.session_state["progress_clipboard_image"]

                        st.success(f"Progress saved! Category: {result.get('screen_name', result.get('category', 'Unknown'))}")

                        # Show result
                        st.markdown("### Detected Values")
                        values = result.get("values", {})
                        if values:
                            for key, val in values.items():
                                st.markdown(f"- **{val.get('label', key)}:** {val.get('value', 'N/A')}")

                        # Show comparison if available
                        comparison = result.get("comparison")
                        if comparison:
                            st.markdown("### Comparison to Previous")
                            st.info(f"Time since last snapshot: {comparison.get('time_elapsed', 'Unknown')}")

                            improvements = comparison.get("improvements", [])
                            if improvements:
                                st.markdown("**Improvements:**")
                                for imp in improvements:
                                    st.markdown(f"- **{imp['metric']}:** +{imp['gain']} ({imp['percentage']}%)")
                            else:
                                st.info("No improvements detected (or first snapshot for this category)")

                        st.rerun()

        with history_col:
            st.subheader("Progress History")

            # Category filter
            filter_category = st.selectbox(
                "Filter by Category",
                options=["all"] + list(ProgressTracker.PROGRESS_CATEGORIES.keys()),
                format_func=lambda x: "All Categories" if x == "all" else ProgressTracker.PROGRESS_CATEGORIES[x]["name"],
                key="history_filter"
            )

            # Get snapshots
            filter_cat = None if filter_category == "all" else filter_category
            snapshots = progress_tracker.get_all_snapshots(filter_cat)

            if not snapshots:
                st.info("No progress snapshots yet. Upload your first screenshot!")
            else:
                for snapshot in snapshots[:10]:  # Show last 10
                    cat_name = ProgressTracker.PROGRESS_CATEGORIES.get(
                        snapshot.get("category", "other"), {}
                    ).get("name", snapshot.get("category", "Unknown"))

                    date_str = snapshot.get("date", "")[:16].replace("T", " ")

                    with st.expander(f"{cat_name} - {date_str}", expanded=False):
                        # Show screenshot if available
                        screenshot_path = snapshot.get("screenshot_path")
                        if screenshot_path and Path(screenshot_path).exists():
                            st.image(screenshot_path, caption="Screenshot", use_container_width=True)

                        # Values
                        st.markdown("**Values:**")
                        for key, val in snapshot.get("values", {}).items():
                            st.markdown(f"- {val.get('label', key)}: **{val.get('value', 'N/A')}**")

                        # Notes
                        if snapshot.get("notes"):
                            st.markdown(f"**Notes:** {snapshot['notes']}")

                        # Comparison
                        comparison = snapshot.get("comparison")
                        if comparison:
                            st.markdown("---")
                            st.markdown(f"**vs Previous ({comparison.get('time_elapsed', '?')}):**")

                            improvements = comparison.get("improvements", [])
                            if improvements:
                                for imp in improvements[:5]:
                                    direction = "+" if "+" not in str(imp['gain']) else ""
                                    st.markdown(f"- {imp['metric']}: {direction}{imp['gain']} ({imp['percentage']}%)")

                        # Delete button
                        if st.button("Delete", key=f"del_snap_{snapshot.get('id')}"):
                            progress_tracker.delete_snapshot(snapshot.get("id"))
                            st.rerun()

        st.divider()

        # Progress Timeline Section
        st.subheader("Progress Timeline")

        if summary.get("categories_tracked"):
            timeline_category = st.selectbox(
                "Select category for timeline",
                options=summary.get("categories_tracked"),
                format_func=lambda x: ProgressTracker.PROGRESS_CATEGORIES[x]["name"],
                key="timeline_category"
            )

            timeline = progress_tracker.get_progress_timeline(timeline_category)

            if len(timeline) >= 2:
                st.markdown(f"**{len(timeline)} snapshots recorded**")

                # Create a simple comparison table
                st.markdown("### Value Changes Over Time")

                # Get all unique metrics across timeline
                all_metrics = set()
                for point in timeline:
                    for key in point.get("values", {}).keys():
                        all_metrics.add(key)

                # Show first vs last comparison
                first = timeline[0]
                last = timeline[-1]

                comparison_data = []
                for metric in all_metrics:
                    first_val = first.get("values", {}).get(metric, {})
                    last_val = last.get("values", {}).get(metric, {})

                    if first_val and last_val:
                        first_num = first_val.get("numeric", 0)
                        last_num = last_val.get("numeric", 0)
                        diff = last_num - first_num

                        if first_num > 0:
                            pct = ((last_num - first_num) / first_num) * 100
                        else:
                            pct = 100 if last_num > 0 else 0

                        comparison_data.append({
                            "Metric": first_val.get("label", metric),
                            "First": first_val.get("value", str(first_num)),
                            "Current": last_val.get("value", str(last_num)),
                            "Change": progress_tracker._format_number(abs(diff)) if diff >= 0 else f"-{progress_tracker._format_number(abs(diff))}",
                            "Growth %": f"{pct:.1f}%"
                        })

                if comparison_data:
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True)

                    # Calculate overall growth
                    total_growth = sum([
                        float(d["Growth %"].replace("%", ""))
                        for d in comparison_data
                        if d["Growth %"] != "100.0%"  # Exclude new metrics
                    ])
                    avg_growth = total_growth / len(comparison_data) if comparison_data else 0

                    st.metric("Average Growth", f"{avg_growth:.1f}%")

            elif len(timeline) == 1:
                st.info("Need at least 2 snapshots to show progress timeline.")
            else:
                st.info("No snapshots for this category yet.")
        else:
            st.info("No progress data yet. Upload some screenshots to start tracking!")

        # Generate Report Section
        st.divider()
        st.subheader("Generate Progress Report")

        if st.button("Generate Full Report", key="gen_report"):
            report = progress_tracker.generate_progress_report()
            st.markdown(report)

            st.download_button(
                "Download Report",
                data=report,
                file_name=f"tower_progress_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

        # Danger Zone
        with st.expander("Danger Zone"):
            st.warning("These actions cannot be undone!")

            clear_cat = st.selectbox(
                "Category to clear",
                options=["all"] + list(ProgressTracker.PROGRESS_CATEGORIES.keys()),
                format_func=lambda x: "All Categories" if x == "all" else ProgressTracker.PROGRESS_CATEGORIES[x]["name"],
                key="clear_category"
            )

            if st.button("Clear History", type="secondary", key="clear_progress"):
                clear_cat_val = None if clear_cat == "all" else clear_cat
                progress_tracker.clear_history(clear_cat_val)
                st.success("History cleared!")
                st.rerun()

    # TAB 10: Tower AI
    with tab10:
        st.header("Tower AI - Game Strategy Agent")
        st.markdown("Train an AI on Reddit data to create game progression strategies.")

        # Initialize components
        bulk_scraper = BulkRedditScraper()
        training_extractor = TrainingDataExtractor()

        # Sub-tabs for AI functionality
        ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs(["📥 Scrape Reddit", "🔬 Extract Training Data", "🎯 Strategy Agent", "💬 Ask AI"])

        # AI TAB 1: Reddit Scraping
        with ai_tab1:
            st.subheader("Bulk Reddit Scraping")
            st.markdown("Scrape thousands of posts from r/TheTowerGame for AI training.")

            # Show existing cache files
            cache_files = bulk_scraper.list_cache_files()
            if cache_files:
                st.info(f"Found {len(cache_files)} cached scrape files")
                with st.expander("View Cached Files"):
                    for cf in cache_files:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.text(cf["filename"])
                        with col2:
                            st.text(f"{cf['post_count']} posts")
                        with col3:
                            st.text(cf["scraped_at"][:10])

            st.divider()

            # Scraping options
            col1, col2, col3 = st.columns(3)
            with col1:
                target_posts = st.number_input(
                    "Target Posts",
                    min_value=100,
                    max_value=10000,
                    value=5000,
                    step=500,
                    help="Reddit allows ~1000 posts per sort type"
                )
            with col2:
                sort_method = st.selectbox(
                    "Sort Method",
                    ["top", "hot", "new", "controversial"],
                    index=0
                )
            with col3:
                timeframe = st.selectbox(
                    "Timeframe (for 'top')",
                    ["all", "year", "month", "week", "day"],
                    index=0
                )

            col1, col2 = st.columns(2)
            with col1:
                include_comments = st.checkbox("Include Comments", value=True)
            with col2:
                comments_per_post = st.slider("Comments per Post", 5, 20, 10) if include_comments else 0

            if st.button("Start Scraping", type="primary", key="start_scrape"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(status, current, total):
                    if total > 0:
                        progress_bar.progress(min(current / total, 1.0))
                    status_text.text(status)

                with st.spinner("Scraping Reddit (this may take a while)..."):
                    posts = bulk_scraper.scrape_subreddit(
                        subreddit="TheTowerGame",
                        target_count=target_posts,
                        sort=sort_method,
                        timeframe=timeframe,
                        include_comments=include_comments,
                        comments_per_post=comments_per_post,
                        progress_callback=update_progress
                    )

                    if posts:
                        # Save to cache
                        cache_path = bulk_scraper.save_to_cache(posts)

                        # Show stats
                        stats = bulk_scraper.get_stats(posts)
                        progress_bar.empty()
                        status_text.empty()

                        st.success(f"Scraped {len(posts)} posts!")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Posts", stats["total_posts"])
                        with col2:
                            st.metric("Avg Score", f"{stats['avg_score']:.1f}")
                        with col3:
                            st.metric("Avg Comments", f"{stats['avg_comments']:.1f}")
                        with col4:
                            st.metric("Comments Fetched", stats["fetched_comments"])

                        st.markdown("**Top Flairs:**")
                        flair_text = ", ".join([f"{f}: {c}" for f, c in stats["top_flairs"]])
                        st.caption(flair_text)

                        st.session_state["scraped_posts"] = posts
                    else:
                        st.error("Scraping failed. Check console for errors.")

        # AI TAB 2: Training Data Extraction
        with ai_tab2:
            st.subheader("Extract Training Data")
            st.markdown("Process scraped posts to extract game strategies and knowledge.")

            # Select source data
            cache_files = bulk_scraper.list_cache_files()

            if not cache_files:
                st.warning("No scraped data available. Go to 'Scrape Reddit' first.")
            else:
                source_file = st.selectbox(
                    "Select Source Data",
                    options=[cf["filename"] for cf in cache_files],
                    format_func=lambda x: f"{x} ({[cf for cf in cache_files if cf['filename'] == x][0]['post_count']} posts)"
                )

                col1, col2 = st.columns(2)
                with col1:
                    min_score = st.slider("Minimum Post Score", 1, 50, 5)
                with col2:
                    min_relevance = st.slider("Minimum Relevance", 0.3, 1.0, 0.5, 0.1)

                if st.button("Extract Training Data", type="primary", key="extract_data"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(status, current, total):
                        if total > 0:
                            progress_bar.progress(min(current / total, 1.0))
                        status_text.text(status)

                    with st.spinner("Extracting and categorizing content..."):
                        # Load posts
                        posts = bulk_scraper.load_from_cache(source_file)

                        if posts:
                            # Extract content
                            status_text.text("Extracting content from posts...")
                            extracted = training_extractor.extract_from_posts(posts, min_score)

                            # Categorize with AI
                            status_text.text("Categorizing with AI...")
                            categorized = training_extractor.categorize_content_batch(
                                extracted["content"],
                                progress_callback=update_progress
                            )

                            # Extract strategies
                            status_text.text("Extracting strategies...")
                            strategies = training_extractor.extract_strategies(
                                categorized,
                                min_relevance=min_relevance,
                                progress_callback=update_progress
                            )

                            # Build knowledge base
                            kb = training_extractor.build_knowledge_base(strategies)

                            # Save
                            kb_path = training_extractor.save_training_data(kb, "knowledge_base.json")

                            progress_bar.empty()
                            status_text.empty()

                            st.success("Training data extracted!")

                            # Show results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Categories", len(kb["categories"]))
                            with col2:
                                st.metric("Total Strategies", kb["meta"]["total_strategies"])

                            st.markdown("### Extracted Strategies by Category")
                            for cat_key, cat_data in kb.get("categories", {}).items():
                                with st.expander(f"{cat_data['name']} ({cat_data['strategy_count']} strategies)"):
                                    for strat in cat_data.get("strategies", []):
                                        st.markdown(f"**{strat.get('title', 'Unnamed')}**")
                                        st.caption(f"Priority: {strat.get('priority', '?')} | Stage: {strat.get('game_stage', 'all')}")
                                        st.markdown(strat.get("description", ""))
                                        if strat.get("tips"):
                                            st.markdown("Tips: " + ", ".join(strat["tips"][:3]))
                                        st.divider()

                            st.session_state["knowledge_base"] = kb

            # Show existing training data
            st.divider()
            training_files = training_extractor.list_training_files()
            if training_files:
                st.markdown("### Saved Training Data")
                for tf in training_files:
                    st.caption(f"{tf['filename']} - {tf['type']} ({tf['created_at'][:10]})")

        # AI TAB 3: Strategy Agent
        with ai_tab3:
            st.subheader("Strategy Planning Agent")
            st.markdown("Let AI create a personalized progression plan for your game state.")

            # Check for knowledge base
            kb_path = Path("data/ai/training_data/knowledge_base.json")
            if kb_path.exists():
                agent = TowerAgent(str(kb_path))
                st.success("Knowledge base loaded!")
            else:
                agent = TowerAgent()
                st.warning("No knowledge base found. Agent will use general knowledge only.")

            st.markdown("### Enter Your Current Game State")

            col1, col2, col3 = st.columns(3)
            with col1:
                tier = st.number_input("Current Tier", 1, 20, 10)
                highest_wave = st.number_input("Highest Wave", 100, 50000, 5000, step=500)
            with col2:
                total_coins = st.text_input("Total Coins", "100B")
                coins_per_hour = st.text_input("Coins per Hour", "5B")
            with col3:
                labs_completed = st.number_input("Labs Completed", 0, 100, 20)
                cards_count = st.number_input("Total Cards", 0, 100, 40)

            # Ultimate weapons selection
            uw_options = ["smart_missiles", "poison_swamp", "death_ray", "chain_lightning",
                          "golden_tower", "black_hole", "chrono_field", "spotlight"]
            selected_uw = st.multiselect("Ultimate Weapons Unlocked", uw_options)

            # Goals
            goal_options = ["reach_next_tier", "improve_coins_per_hour", "unlock_more_uw",
                           "improve_tournament", "max_workshop", "collect_all_cards"]
            selected_goals = st.multiselect("Your Goals", goal_options, default=["reach_next_tier"])

            # Time horizon
            time_horizon = st.selectbox("Plan Timeframe", ["day", "week", "month"], index=1)

            if st.button("Generate Progression Plan", type="primary", key="gen_plan"):
                # Set game state
                game_state = {
                    "tier": tier,
                    "highest_wave": highest_wave,
                    "total_coins": total_coins,
                    "coins_per_hour": coins_per_hour,
                    "labs_completed": labs_completed,
                    "cards": {"total": cards_count},
                    "ultimate_weapons": selected_uw,
                    "goals": selected_goals
                }

                agent.set_current_state(game_state)

                with st.spinner("AI is analyzing your game state and creating a plan..."):
                    # Generate plan
                    plan = agent.create_progression_plan(time_horizon=time_horizon)

                    if "error" in plan:
                        st.error(plan["error"])
                    else:
                        st.success(f"Plan created: {plan.get('plan_name', 'Progression Plan')}")

                        # Executive summary
                        st.markdown("### Executive Summary")
                        st.info(plan.get("executive_summary", ""))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Tier", plan.get("current_tier", tier))
                        with col2:
                            st.metric("Target Tier", plan.get("target_tier", "?"))

                        # Immediate actions
                        st.markdown("### Immediate Actions")
                        for i, action in enumerate(plan.get("immediate_actions", [])[:5], 1):
                            with st.expander(f"{i}. {action.get('action', 'Action')}", expanded=i==1):
                                st.markdown(f"**Category:** {action.get('category', 'general')}")
                                st.markdown(f"**Priority:** {action.get('priority', '?')}")
                                st.markdown(f"**Impact:** {action.get('expected_impact', 'Unknown')}")
                                st.markdown(f"**Details:** {action.get('details', '')}")

                        # Daily routine
                        routine = plan.get("daily_routine", {})
                        if routine:
                            st.markdown("### Daily Routine")
                            for activity in routine.get("active_play", []):
                                st.markdown(f"- **{activity.get('activity', '')}** ({activity.get('duration', '')})")

                            afk = routine.get("afk_strategy", {})
                            if afk:
                                st.markdown(f"**AFK Strategy:** Farm Tier {afk.get('tier', '?')} up to wave {afk.get('wave_target', '?')}")

                        # Upgrade priorities
                        st.markdown("### Upgrade Priorities")
                        upgrades = plan.get("upgrade_priorities", {})

                        up_col1, up_col2 = st.columns(2)
                        with up_col1:
                            if upgrades.get("workshop"):
                                st.markdown("**Workshop:**")
                                for up in upgrades["workshop"][:5]:
                                    st.caption(f"- {up.get('stat', '?')}: {up.get('current', '?')} -> {up.get('target', '?')}")

                            if upgrades.get("cards"):
                                st.markdown("**Cards:**")
                                for card in upgrades["cards"][:3]:
                                    st.caption(f"- {card}")

                        with up_col2:
                            if upgrades.get("ultimate_weapons"):
                                st.markdown("**Ultimate Weapons:**")
                                for uw in upgrades["ultimate_weapons"][:3]:
                                    st.caption(f"- {uw}")

                            if upgrades.get("labs"):
                                st.markdown("**Labs:**")
                                for lab in upgrades["labs"][:3]:
                                    st.caption(f"- {lab}")

                        # Warnings
                        warnings = plan.get("warnings", [])
                        if warnings:
                            st.markdown("### Warnings")
                            for warn in warnings:
                                st.warning(warn)

                        # Save plan
                        plan_path = agent.save_plan()
                        st.success(f"Plan saved to {plan_path}")

                        st.session_state["current_plan"] = plan
                        st.session_state["tower_agent"] = agent

        # AI TAB 4: Ask AI
        with ai_tab4:
            st.subheader("Ask Tower AI")
            st.markdown("Ask questions about The Tower game strategy.")

            # Load agent if available
            if "tower_agent" in st.session_state:
                agent = st.session_state["tower_agent"]
            else:
                kb_path = Path("data/ai/training_data/knowledge_base.json")
                agent = TowerAgent(str(kb_path)) if kb_path.exists() else TowerAgent()

            question = st.text_area(
                "Your Question",
                placeholder="e.g., What's the best way to improve coins per hour at Tier 12?",
                height=100
            )

            if st.button("Ask AI", type="primary", key="ask_ai") and question:
                with st.spinner("Thinking..."):
                    answer = agent.ask_question(question)
                    st.markdown("### Answer")
                    st.markdown(answer)

            # Quick questions
            st.divider()
            st.markdown("### Quick Questions")

            quick_questions = [
                "What should I focus on in early game?",
                "How do I improve coins per hour?",
                "What's the best workshop upgrade order?",
                "Which Ultimate Weapons should I prioritize?",
                "How do I prepare for tournaments?",
                "What's the best AFK farming strategy?"
            ]

            cols = st.columns(2)
            for i, q in enumerate(quick_questions):
                with cols[i % 2]:
                    if st.button(q, key=f"quick_{i}"):
                        with st.spinner("Thinking..."):
                            answer = agent.ask_question(q)
                            st.session_state["last_answer"] = {"q": q, "a": answer}

            if "last_answer" in st.session_state:
                st.markdown(f"### Q: {st.session_state['last_answer']['q']}")
                st.markdown(st.session_state["last_answer"]["a"])


if __name__ == "__main__":
    main()
