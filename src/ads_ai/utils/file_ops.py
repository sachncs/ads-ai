"""File operations for run artifact management."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class FileOperationError(Exception):
    """Raised when a file write or serialization operation fails."""


@runtime_checkable
class ModelDumper(Protocol):
    """An object that exposes ``model_dump`` for serialization."""

    def model_dump(self, mode: str = ...) -> dict[str, Any]: ...


@runtime_checkable
class LegacyDumper(Protocol):
    """An object that exposes ``dict`` for serialization (Pydantic v1)."""

    def dict(self, **kwargs: Any) -> dict[str, Any]: ...


JsonScalar = str | int | float | bool
"""JSON-native scalar types."""


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


def save_json(
    data: ModelDumper | LegacyDumper | dict[str, Any] | list[Any] | JsonScalar,
    filepath: Path,
) -> None:
    """Serializes data to a JSON file.

    Args:
        data: The object to serialize. Accepts Pydantic v2 models
            (with ``model_dump``), Pydantic v1 models (with ``dict``),
            dicts, lists, or JSON-native scalars.
        filepath: The full path to the destination file.

    Raises:
        FileOperationError: If serialization or file writing fails.
    """
    start = time.perf_counter()
    exc: Exception | None = None
    try:
        if isinstance(data, dict):
            serialized: Any = data
        elif isinstance(data, list):
            serialized = data
        elif isinstance(data, str | int | float | bool):
            serialized = data
        elif hasattr(data, "model_dump"):
            serialized = data.model_dump(mode="json")
        elif hasattr(data, "dict"):
            serialized = data.dict()  # pragma: no cover
        else:
            raise TypeError(
                f"data must be a dict, list, JSON scalar, or an object "
                f"with model_dump/dict; got {type(data).__name__}"
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        exc = e
    finally:
        elapsed = time.perf_counter() - start
        logger.debug("save_json(%s) completed in %.3fs", filepath.name, elapsed)
    if exc is not None:
        raise FileOperationError(
            f"Failed to save JSON to {filepath}: {exc}"
        ) from exc
