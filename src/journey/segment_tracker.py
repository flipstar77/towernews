"""Segment Tracker - Stores and retrieves journey progress history."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class SegmentTracker:
    """Tracks journey segment values over time."""

    def __init__(self, config: dict):
        self.config = config
        self.history_path = Path("data/journey/history.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Load history from JSON file."""
        if self.history_path.exists():
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                "entries": [],
                "latest": {}  # Quick access to most recent values per segment
            }

    def _save_history(self):
        """Save history to JSON file."""
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_latest(self, segment_type: str) -> Optional[dict]:
        """
        Get the most recent value for a segment.

        Returns:
            dict with 'value', 'formatted', 'date' or None if no history
        """
        return self.history.get("latest", {}).get(segment_type)

    def get_previous(self, segment_type: str) -> Optional[dict]:
        """
        Get the previous (second most recent) value for a segment.

        Returns:
            dict with 'value', 'formatted', 'date' or None if no history
        """
        entries = [e for e in self.history.get("entries", [])
                   if e.get("segment") == segment_type]

        if len(entries) >= 2:
            # Sort by date descending, get second
            sorted_entries = sorted(entries, key=lambda x: x.get("date", ""), reverse=True)
            return sorted_entries[1]

        return None

    def add_entry(self, segment_type: str, value: float, formatted: str, notes: str = "") -> dict:
        """
        Add a new entry to the history.

        Args:
            segment_type: Type of segment (coins, labs, highest_wave)
            value: Numeric value
            formatted: Display string (e.g., "1.5B")
            notes: Optional user notes

        Returns:
            dict with entry data and comparison to previous
        """
        now = datetime.now().isoformat()

        # Get previous for comparison
        previous = self.get_latest(segment_type)

        entry = {
            "segment": segment_type,
            "value": value,
            "formatted": formatted,
            "notes": notes,
            "date": now,
            "episode": self._get_episode_number(segment_type)
        }

        # Calculate comparison
        if previous:
            diff = value - previous.get("value", 0)
            if previous.get("value", 0) > 0:
                pct = ((value - previous["value"]) / previous["value"]) * 100
            else:
                pct = 100 if value > 0 else 0

            entry["comparison"] = {
                "previous_value": previous.get("value"),
                "previous_formatted": previous.get("formatted"),
                "difference": diff,
                "percentage": round(pct, 1),
                "direction": "up" if diff > 0 else ("down" if diff < 0 else "same")
            }
        else:
            entry["comparison"] = None

        # Add to entries
        self.history["entries"].append(entry)

        # Update latest
        if "latest" not in self.history:
            self.history["latest"] = {}
        self.history["latest"][segment_type] = {
            "value": value,
            "formatted": formatted,
            "date": now
        }

        self._save_history()

        return entry

    def _get_episode_number(self, segment_type: str) -> int:
        """Get the next episode number for a segment."""
        entries = [e for e in self.history.get("entries", [])
                   if e.get("segment") == segment_type]
        return len(entries) + 1

    def get_all_entries(self, segment_type: str = None) -> list:
        """Get all entries, optionally filtered by segment type."""
        entries = self.history.get("entries", [])
        if segment_type:
            entries = [e for e in entries if e.get("segment") == segment_type]
        return sorted(entries, key=lambda x: x.get("date", ""), reverse=True)

    def get_journey_summary(self) -> dict:
        """Get a summary of all segments for video generation."""
        summary = {}
        segments = self.config.get("segments", {})

        for segment_type in segments.keys():
            latest = self.get_latest(segment_type)
            if latest:
                # Find the entry with comparison data
                entries = [e for e in self.history.get("entries", [])
                           if e.get("segment") == segment_type]
                if entries:
                    most_recent = sorted(entries, key=lambda x: x.get("date", ""), reverse=True)[0]
                    summary[segment_type] = most_recent

        return summary

    def clear_history(self, segment_type: str = None):
        """Clear history, optionally only for a specific segment."""
        if segment_type:
            self.history["entries"] = [e for e in self.history.get("entries", [])
                                        if e.get("segment") != segment_type]
            if segment_type in self.history.get("latest", {}):
                del self.history["latest"][segment_type]
        else:
            self.history = {"entries": [], "latest": {}}

        self._save_history()
