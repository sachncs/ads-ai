# ads_ai.utils

File operations and utilities for run artifact management.

## Overview

Provides tools for creating output directories, serializing run data, and managing artifacts produced during pipeline execution.

## Available Functions

### `ensure_output_dir(base_dir: str = "outputs") -> Path`

Creates a timestamped output directory for run artifacts.

```python
from ads_ai.utils import ensure_output_dir

run_dir = ensure_output_dir("outputs")
# Returns: outputs/run_20260428_143022/
```

### `save_json(data, filepath: Path) -> None`

Serializes data to a JSON file. Supports:
- Python dicts and lists
- JSON-native scalars (str, int, float, bool)
- Pydantic v2 models (via `model_dump`)
- Pydantic v1 models (via `dict`)

```python
from ads_ai.utils import save_json

save_json({"key": "value"}, Path("output.json"))
```

## Exceptions

### `FileOperationError`

Raised when a file write or serialization operation fails.

## Architecture

The module uses runtime Protocol classes (`ModelDumper`, `LegacyDumper`) for duck-typing serialization support, avoiding hard dependency on Pydantic version.