"""Progress Tracker - Track and compare game progress over time using screenshots."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import base64
from openai import OpenAI


class ProgressTracker:
    """Tracks Tower game progress over time by analyzing screenshots."""

    # Categories of progress we can track
    PROGRESS_CATEGORIES = {
        "workshop": {
            "name": "Workshop",
            "description": "Workshop upgrade levels",
            "key_metrics": ["total_levels", "highest_upgrade", "max_level_count"]
        },
        "cards": {
            "name": "Cards",
            "description": "Card collection and levels",
            "key_metrics": ["total_cards", "max_level_cards", "rare_cards", "epic_cards", "legendary_cards"]
        },
        "masteries": {
            "name": "Masteries",
            "description": "Mastery unlocks and levels",
            "key_metrics": ["total_unlocked", "total_levels", "max_mastery"]
        },
        "coins": {
            "name": "Coins",
            "description": "Total coins and earnings",
            "key_metrics": ["total_coins", "coins_per_hour"]
        },
        "stats": {
            "name": "Alltime Stats",
            "description": "Overall game statistics",
            "key_metrics": ["highest_wave", "total_kills", "total_damage", "play_time"]
        },
        "labs": {
            "name": "Laboratory",
            "description": "Lab research progress",
            "key_metrics": ["labs_completed", "current_research"]
        },
        "ultimate_weapons": {
            "name": "Ultimate Weapons",
            "description": "UW unlocks and levels",
            "key_metrics": ["uw_unlocked", "total_uw_levels"]
        },
        "perks": {
            "name": "Perks",
            "description": "Perk unlocks and levels",
            "key_metrics": ["total_perks", "perk_levels"]
        }
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.history_path = Path("data/journey/progress_history.json")
        self.screenshots_path = Path("data/journey/progress_screenshots")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.screenshots_path.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = self.config.get("llm", {}).get("model", "gpt-4o")
        self._load_history()

    def _load_history(self):
        """Load progress history from JSON file."""
        if self.history_path.exists():
            with open(self.history_path, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        else:
            self.history = {
                "snapshots": [],  # List of progress snapshots over time
                "latest": {}  # Quick access to most recent values per category
            }

    def _save_history(self):
        """Save history to JSON file."""
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_screenshot(self, image_path: str, category: str = None) -> Dict[str, Any]:
        """
        Analyze a screenshot and extract progress values.

        Args:
            image_path: Path to the screenshot
            category: Optional category hint (workshop, cards, etc.)

        Returns:
            dict with 'category', 'values', 'raw_response'
        """
        base64_image = self.encode_image(image_path)

        category_context = ""
        if category and category in self.PROGRESS_CATEGORIES:
            cat_info = self.PROGRESS_CATEGORIES[category]
            category_context = f"\nThis appears to be a {cat_info['name']} screenshot. Key metrics to find: {', '.join(cat_info['key_metrics'])}"

        prompt = f"""Analyze this Tower (idle tower defense game) screenshot and extract ALL visible progress values.

1. IDENTIFY the screen type:
   - Workshop: Shows upgrade levels for various stats
   - Cards: Shows card collection with levels
   - Masteries: Shows mastery unlocks and levels
   - Alltime Stats: Shows total coins, highest wave, play time
   - Laboratory: Shows lab research progress
   - Ultimate Weapons: Shows UW unlocks and levels
   - Perks: Shows perk unlocks and levels
   - Other: Any other game screen
{category_context}

2. EXTRACT all visible numerical values with their labels.

Respond in this exact JSON format:
{{
    "category": "workshop|cards|masteries|stats|labs|ultimate_weapons|perks|other",
    "screen_name": "Human readable name",
    "confidence": 0.0-1.0,
    "values": {{
        "metric_name": {{
            "value": "displayed value string",
            "numeric": 123456,
            "label": "Original label from screen",
            "unit": "optional unit like 'levels', 'cards', etc."
        }}
    }},
    "summary": {{
        "total_items": "if applicable, total count of items",
        "key_achievement": "notable achievement if visible"
    }}
}}

For numbers with suffixes (K, M, B, T, q), include both displayed value and full numeric.
Example: "1.5B" -> {{"value": "1.5B", "numeric": 1500000000, "label": "Total Coins"}}

Extract ALL visible stats and values, not just the main ones."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result

    def add_snapshot(self,
                     image_path: str,
                     category: str = None,
                     notes: str = "",
                     save_screenshot: bool = True) -> Dict[str, Any]:
        """
        Add a new progress snapshot from a screenshot.

        Args:
            image_path: Path to the screenshot
            category: Optional category hint
            notes: Optional user notes
            save_screenshot: Whether to save a copy of the screenshot

        Returns:
            dict with snapshot data and comparison to previous
        """
        # Analyze the screenshot
        analysis = self.analyze_screenshot(image_path, category)

        category = analysis.get("category", "other")
        now = datetime.now().isoformat()

        # Save screenshot copy if requested
        saved_path = None
        if save_screenshot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = Path(image_path).suffix
            saved_path = str(self.screenshots_path / f"{category}_{timestamp}{ext}")
            import shutil
            shutil.copy2(image_path, saved_path)

        # Get previous snapshot for comparison
        previous = self.get_latest(category)

        # Create snapshot entry
        snapshot = {
            "id": len(self.history.get("snapshots", [])) + 1,
            "category": category,
            "screen_name": analysis.get("screen_name", category),
            "date": now,
            "values": analysis.get("values", {}),
            "summary": analysis.get("summary", {}),
            "confidence": analysis.get("confidence", 0),
            "notes": notes,
            "screenshot_path": saved_path
        }

        # Calculate comparison if we have previous data
        if previous:
            comparison = self._calculate_comparison(previous, snapshot)
            snapshot["comparison"] = comparison
        else:
            snapshot["comparison"] = None

        # Add to history
        if "snapshots" not in self.history:
            self.history["snapshots"] = []
        self.history["snapshots"].append(snapshot)

        # Update latest
        if "latest" not in self.history:
            self.history["latest"] = {}
        self.history["latest"][category] = {
            "snapshot_id": snapshot["id"],
            "date": now,
            "values": analysis.get("values", {}),
            "summary": analysis.get("summary", {})
        }

        self._save_history()
        return snapshot

    def _calculate_comparison(self, previous: Dict, current: Dict) -> Dict[str, Any]:
        """Calculate the difference between two snapshots."""
        comparison = {
            "time_elapsed": self._calculate_time_elapsed(previous.get("date"), current.get("date")),
            "changes": {},
            "improvements": [],
            "total_improvement_score": 0
        }

        prev_values = previous.get("values", {})
        curr_values = current.get("values", {})

        improvement_count = 0

        for key, curr_data in curr_values.items():
            if key in prev_values:
                prev_data = prev_values[key]
                prev_num = prev_data.get("numeric", 0)
                curr_num = curr_data.get("numeric", 0)

                if prev_num != 0:
                    diff = curr_num - prev_num
                    pct = ((curr_num - prev_num) / prev_num) * 100
                else:
                    diff = curr_num
                    pct = 100 if curr_num > 0 else 0

                direction = "up" if diff > 0 else ("down" if diff < 0 else "same")

                comparison["changes"][key] = {
                    "previous": prev_data.get("value", str(prev_num)),
                    "current": curr_data.get("value", str(curr_num)),
                    "difference": diff,
                    "difference_formatted": self._format_number(abs(diff)),
                    "percentage": round(pct, 1),
                    "direction": direction,
                    "label": curr_data.get("label", key)
                }

                if diff > 0:
                    improvement_count += 1
                    comparison["improvements"].append({
                        "metric": curr_data.get("label", key),
                        "gain": self._format_number(diff),
                        "percentage": round(pct, 1)
                    })
            else:
                # New metric that wasn't tracked before
                comparison["changes"][key] = {
                    "previous": None,
                    "current": curr_data.get("value"),
                    "is_new": True,
                    "label": curr_data.get("label", key)
                }

        comparison["total_improvement_score"] = improvement_count
        return comparison

    def _calculate_time_elapsed(self, start_date: str, end_date: str) -> str:
        """Calculate human-readable time elapsed."""
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            delta = end - start

            days = delta.days
            hours = delta.seconds // 3600

            if days > 0:
                return f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"
            elif hours > 0:
                minutes = (delta.seconds % 3600) // 60
                return f"{hours} hour{'s' if hours != 1 else ''}, {minutes} minute{'s' if minutes != 1 else ''}"
            else:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''}"
        except:
            return "Unknown"

    def _format_number(self, value: float) -> str:
        """Format a large number with appropriate suffix."""
        if value >= 1_000_000_000_000_000:
            return f"{value / 1_000_000_000_000_000:.2f}q"
        elif value >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.2f}T"
        elif value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.2f}K"
        else:
            return str(int(value))

    def get_latest(self, category: str) -> Optional[Dict]:
        """Get the most recent snapshot for a category."""
        latest = self.history.get("latest", {}).get(category)
        if latest:
            # Find the full snapshot
            snapshot_id = latest.get("snapshot_id")
            for snapshot in self.history.get("snapshots", []):
                if snapshot.get("id") == snapshot_id:
                    return snapshot
        return None

    def get_all_snapshots(self, category: str = None) -> List[Dict]:
        """Get all snapshots, optionally filtered by category."""
        snapshots = self.history.get("snapshots", [])
        if category:
            snapshots = [s for s in snapshots if s.get("category") == category]
        return sorted(snapshots, key=lambda x: x.get("date", ""), reverse=True)

    def get_progress_timeline(self, category: str, metric: str = None) -> List[Dict]:
        """
        Get a timeline of progress for a specific category.

        Args:
            category: Category to get timeline for
            metric: Optional specific metric to track

        Returns:
            List of data points with dates and values
        """
        snapshots = self.get_all_snapshots(category)
        timeline = []

        for snapshot in reversed(snapshots):  # Oldest first
            point = {
                "date": snapshot.get("date"),
                "snapshot_id": snapshot.get("id")
            }

            if metric:
                values = snapshot.get("values", {})
                if metric in values:
                    point["value"] = values[metric].get("numeric", 0)
                    point["formatted"] = values[metric].get("value", "")
            else:
                # Include all values
                point["values"] = snapshot.get("values", {})
                point["summary"] = snapshot.get("summary", {})

            timeline.append(point)

        return timeline

    def get_overall_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of progress across all categories."""
        summary = {
            "categories_tracked": [],
            "total_snapshots": len(self.history.get("snapshots", [])),
            "first_snapshot": None,
            "last_snapshot": None,
            "category_summaries": {}
        }

        snapshots = self.history.get("snapshots", [])
        if snapshots:
            sorted_snapshots = sorted(snapshots, key=lambda x: x.get("date", ""))
            summary["first_snapshot"] = sorted_snapshots[0].get("date")
            summary["last_snapshot"] = sorted_snapshots[-1].get("date")

        # Get latest for each category
        for category in self.PROGRESS_CATEGORIES:
            cat_snapshots = [s for s in snapshots if s.get("category") == category]
            if cat_snapshots:
                summary["categories_tracked"].append(category)

                first = min(cat_snapshots, key=lambda x: x.get("date", ""))
                last = max(cat_snapshots, key=lambda x: x.get("date", ""))

                summary["category_summaries"][category] = {
                    "name": self.PROGRESS_CATEGORIES[category]["name"],
                    "snapshot_count": len(cat_snapshots),
                    "first_date": first.get("date"),
                    "last_date": last.get("date"),
                    "latest_values": last.get("values", {}),
                    "has_comparison": last.get("comparison") is not None
                }

        return summary

    def compare_snapshots(self, snapshot_id_1: int, snapshot_id_2: int) -> Optional[Dict]:
        """Compare two specific snapshots by ID."""
        snapshot_1 = None
        snapshot_2 = None

        for snapshot in self.history.get("snapshots", []):
            if snapshot.get("id") == snapshot_id_1:
                snapshot_1 = snapshot
            if snapshot.get("id") == snapshot_id_2:
                snapshot_2 = snapshot

        if snapshot_1 and snapshot_2:
            return self._calculate_comparison(snapshot_1, snapshot_2)
        return None

    def generate_progress_report(self, category: str = None, num_snapshots: int = 5) -> str:
        """Generate a text progress report."""
        summary = self.get_overall_progress_summary()

        report = "# Tower Progress Report\n\n"
        report += f"**Total Snapshots:** {summary['total_snapshots']}\n"
        report += f"**Categories Tracked:** {', '.join(summary['categories_tracked']) or 'None'}\n"

        if summary['first_snapshot'] and summary['last_snapshot']:
            report += f"**Tracking Period:** {summary['first_snapshot'][:10]} to {summary['last_snapshot'][:10]}\n"

        report += "\n---\n\n"

        categories_to_show = [category] if category else summary['categories_tracked']

        for cat in categories_to_show:
            cat_summary = summary.get("category_summaries", {}).get(cat)
            if cat_summary:
                cat_name = self.PROGRESS_CATEGORIES.get(cat, {}).get("name", cat)
                report += f"## {cat_name}\n\n"
                report += f"**Snapshots:** {cat_summary['snapshot_count']}\n\n"

                # Latest values
                report += "### Current Values\n"
                for key, val in cat_summary.get("latest_values", {}).items():
                    report += f"- **{val.get('label', key)}:** {val.get('value', 'N/A')}\n"

                # Get recent comparisons
                snapshots = self.get_all_snapshots(cat)[:num_snapshots]
                if len(snapshots) > 1:
                    report += "\n### Recent Changes\n"
                    for snap in snapshots:
                        if snap.get("comparison"):
                            comp = snap["comparison"]
                            report += f"\n**{snap['date'][:10]}** (vs previous, {comp['time_elapsed']} ago)\n"
                            for imp in comp.get("improvements", [])[:5]:
                                report += f"- {imp['metric']}: +{imp['gain']} ({imp['percentage']}%)\n"

                report += "\n---\n\n"

        return report

    def clear_history(self, category: str = None):
        """Clear history, optionally only for a specific category."""
        if category:
            self.history["snapshots"] = [
                s for s in self.history.get("snapshots", [])
                if s.get("category") != category
            ]
            if category in self.history.get("latest", {}):
                del self.history["latest"][category]
        else:
            self.history = {"snapshots": [], "latest": {}}

        self._save_history()

    def delete_snapshot(self, snapshot_id: int):
        """Delete a specific snapshot by ID."""
        self.history["snapshots"] = [
            s for s in self.history.get("snapshots", [])
            if s.get("id") != snapshot_id
        ]

        # Rebuild latest references
        self.history["latest"] = {}
        for snapshot in self.history.get("snapshots", []):
            cat = snapshot.get("category")
            if cat:
                existing = self.history["latest"].get(cat)
                if not existing or snapshot.get("date", "") > existing.get("date", ""):
                    self.history["latest"][cat] = {
                        "snapshot_id": snapshot["id"],
                        "date": snapshot["date"],
                        "values": snapshot.get("values", {}),
                        "summary": snapshot.get("summary", {})
                    }

        self._save_history()
