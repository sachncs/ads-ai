import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def ensure_output_dir(base_dir: str = "outputs") -> Path:
    """Creates a timestamped output directory.

    Args:
        base_dir: The root directory for all outputs.

    Returns:
        The path to the newly created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directory: %s", run_dir)
    return run_dir

def save_json(data: Any, filepath: Path) -> None:
    """Serializes data to a JSON file.

    Args:
        data: The object to serialize (must be JSON-compatible or have .model_dump()).
        filepath: The full path to the destination file.
    """
    try:
        if hasattr(data, "model_dump"):
            serialized = data.model_dump(mode="json")
        elif hasattr(data, "dict"):
            serialized = data.dict()
        else:
            serialized = data

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save JSON to %s: %s", filepath, e)
