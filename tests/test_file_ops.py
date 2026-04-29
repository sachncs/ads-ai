"""Tests for file operation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ads_ai.utils.file_ops import (
    FileOperationError,
    ensure_output_dir,
    save_json,
)


class TestEnsureOutputDir:
    """Tests for output directory creation."""

    def test_creates_timestamped_directory(self, tmp_path: Path) -> None:
        """Should create a directory with a run_YYYYMMDD_HHMMSS prefix."""
        run_dir = ensure_output_dir(str(tmp_path))
        assert run_dir.exists()
        assert run_dir.name.startswith("run_")
        assert run_dir.parent == tmp_path


class TestSaveJson:
    """Tests for JSON serialization."""

    def test_save_dict(self, tmp_path: Path) -> None:
        """Should write a plain dict to a JSON file."""
        filepath = tmp_path / "data.json"
        data = {"key": "value", "nested": {"a": 1}}
        save_json(data, filepath)

        with open(filepath, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_list(self, tmp_path: Path) -> None:
        """Should write a list to a JSON file."""
        filepath = tmp_path / "list.json"
        data = [1, 2, {"a": "b"}]
        save_json(data, filepath)

        with open(filepath, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_string(self, tmp_path: Path) -> None:
        """Should write a plain string to a JSON file."""
        filepath = tmp_path / "str.json"
        save_json("hello", filepath)

        with open(filepath, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == "hello"

    def test_save_pydantic_v2_model(self, tmp_path: Path) -> None:
        """Should serialize an object with model_dump (Pydantic v2)."""
        filepath = tmp_path / "model.json"
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"field": "value"}
        save_json(mock_model, filepath)

        with open(filepath, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == {"field": "value"}
        mock_model.model_dump.assert_called_once_with(mode="json")

    def test_save_pydantic_v1_model(self, tmp_path: Path) -> None:
        """Should serialize an object with dict() (Pydantic v1 fallback)."""
        filepath = tmp_path / "v1_model.json"
        v1_model = MagicMock()
        del v1_model.model_dump
        v1_model.dict.return_value = {"legacy": True}
        save_json(v1_model, filepath)

        with open(filepath, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == {"legacy": True}
        v1_model.dict.assert_called_once()

    def test_save_unsupported_type_raises(self, tmp_path: Path) -> None:
        """Should raise FileOperationError for unsupported types."""
        filepath = tmp_path / "bad.json"
        with pytest.raises(FileOperationError, match="Failed to save JSON"):
            save_json(object(), filepath)

    def test_os_error_raises_file_operation_error(self, tmp_path: Path) -> None:
        """Should wrap OSError during file write."""
        filepath = tmp_path / "readonly.json"
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(FileOperationError, match="denied"):
                save_json({"a": 1}, filepath)

    def test_json_error_raises_file_operation_error(self, tmp_path: Path) -> None:
        """Should wrap JSON serialization errors."""
        filepath = tmp_path / "bad.json"
        circular: dict[str, Any] = {}
        circular["self"] = circular
        with pytest.raises(FileOperationError, match="Failed to save JSON"):
            save_json(circular, filepath)
