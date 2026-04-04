"""Base agent class and shared infrastructure."""

from __future__ import annotations

import logging
import time
from typing import Any

from google import genai
from pydantic import BaseModel

from ads_ai.config import settings

logger = logging.getLogger(__name__)


class BaseAgent:
    """Abstract base providing a shared LLM generation interface.

    Attributes:
        client: An authenticated ``genai.Client`` instance.
        model_name: The model identifier used for content generation.
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str | None = None,
    ) -> None:
        """Initializes the agent with a GenAI client and model.

        Args:
            client: The Google GenAI client.
            model_name: Model identifier. Falls back to the configured
                default when ``None``.
        """
        self.client = client
        self.model_name = model_name or settings.default_text_model

    def _to_json_dict(self, obj: Any) -> Any:
        """Recursively converts Pydantic models/lists into JSON-serializable dicts.

        Args:
            obj: The object to convert. This can be a Pydantic model, a list,
                a dictionary, or a primitive type.

        Returns:
            A JSON-serializable representation of the object (dict, list, or
            primitive).
        """
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, list):
            return [self._to_json_dict(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._to_json_dict(v) for k, v in obj.items()}
        return obj

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> Any:
        """Sends a prompt to the LLM and validates the response.

        Args:
            prompt: The text prompt to send.
            response_schema: A Pydantic model class used to parse and
                validate the JSON response.

        Returns:
            A validated instance of ``response_schema`` if provided,
            otherwise the raw response text string.

        Raises:
            ValueError: If response parsing or validation fails.
            Exception: On upstream API failures or connectivity issues.
        """
        config: dict[str, Any] = {"response_mime_type": "application/json"}
        if response_schema:
            schema_dict = response_schema.model_json_schema()

            def _clean_schema(d: dict[str, Any]) -> None:
                """Removes 'additionalProperties' from the JSON schema recursively.

                Args:
                    d: The dictionary representing the JSON schema or sub-schema.
                """
                if "additionalProperties" in d:
                    del d["additionalProperties"]
                for v in d.values():
                    if isinstance(v, dict):
                        _clean_schema(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                _clean_schema(item)

            _clean_schema(schema_dict)
            config["response_schema"] = schema_dict

        # Transient error retry logic (503 Service Unavailable)
        max_retries = settings.max_retries
        retry_delay = settings.retry_delay

        response_text = ""
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                response_text = response.text
                break
            except Exception as e:
                # Check for 503 in the error message or type
                if "503" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        "Gemini API 503 (attempt %d/%d). Retrying in %ds...",
                        attempt + 1,
                        max_retries,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                raise e

        if response_schema:
            return response_schema.model_validate_json(response_text)
        return response_text
