"""Shared timing utilities for agent operations."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@contextmanager
def agent_timing(
    logger: logging.Logger,
    operation_name: str,
    *args: object,
) -> Iterator[list[tuple[str, object]]]:
    """Context manager for timing agent operations with structured logging.

    Args:
        logger: The logger to use for logging.
        operation_name: The name of the operation being timed (e.g., "create_brief").
        *args: Key-value pairs to include in log metadata (format: key=value).

    Yields:
        A list that can be appended with extra metadata before logging.

    Example:
        with agent_timing(logger, "create_brief", product="Widget", goal="sales") as meta:
            meta.append(("platforms", ["tiktok", "instagram"]))
            result = do_work()
        # Logs: "create_brief completed product=Widget goal=sales elapsed=0.123s"
    """
    start = time.perf_counter()
    exc: Exception | None = None
    metadata: list[tuple[str, object]] = []
    for i in range(0, len(args), 2):
        if i + 1 < len(args):
            metadata.append((str(args[i]), args[i + 1]))
    try:
        yield metadata
    except Exception as e:
        exc = e
        raise
    finally:
        elapsed = time.perf_counter() - start
        if exc is None:
            logger.info(
                "%s completed %s elapsed=%.3fs",
                operation_name,
                " ".join(f"{k}={v}" for k, v in metadata),
                elapsed,
            )
        else:
            logger.exception(
                "%s failed %s elapsed=%.3fs",
                operation_name,
                " ".join(f"{k}={v}" for k, v in metadata),
                elapsed,
            )
