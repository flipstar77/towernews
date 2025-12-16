"""Pipeline - Main orchestration of the news generation pipeline."""

import yaml
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from .agents import PlannerAgent


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class NewsPipeline:
    """Main pipeline for generating news videos."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to configuration file
        """
        load_dotenv()  # Load environment variables

        self.config = load_config(config_path)
        self.planner = PlannerAgent(self.config)

    async def run(self, **kwargs) -> dict:
        """
        Run the complete news pipeline.

        Args:
            **kwargs: Override parameters

        Returns:
            Pipeline results
        """
        print("=" * 50)
        print(f"News Pipeline Started - {datetime.now()}")
        print("=" * 50)

        result = await self.planner.run(kwargs)

        print("=" * 50)
        if result.get("success"):
            print(f"Pipeline SUCCEEDED")
            print(f"Video: {result.get('video_path')}")
        else:
            print(f"Pipeline FAILED")
            print(f"Error: {result.get('error')}")
        print("=" * 50)

        return result

    def run_sync(self, **kwargs) -> dict:
        """
        Synchronous wrapper for run().

        Args:
            **kwargs: Override parameters

        Returns:
            Pipeline results
        """
        return asyncio.run(self.run(**kwargs))


def run_daily_pipeline():
    """Run the daily news pipeline."""
    pipeline = NewsPipeline()
    return pipeline.run_sync()


if __name__ == "__main__":
    run_daily_pipeline()
