"""Base agent class and shared infrastructure."""

from __future__ import annotations

import logging
import time
from typing import Any, TypeVar, overload

from google import genai  # type: ignore[attr-defined]
from pydantic import BaseModel

from ads_ai.config import settings

T = TypeVar("T", bound=BaseModel)

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

        Raises:
            ValueError: If the resolved model name is empty.
        """
        self.client = client
        self.model_name = model_name or settings.default_text_model

    def to_json_dict(self, obj: Any) -> Any:
        """Recursively converts Pydantic models/lists into JSON-serializable dicts.

        Args:
            obj: The object to convert. This can be a Pydantic model, a list,
                a dictionary, or a primitive type.

        Returns:
            A JSON-serializable dict, list, or primitive.
        """
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, list):
            return [self.to_json_dict(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self.to_json_dict(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        """Determines whether an upstream error warrants a retry.

        Checks common SDK attributes (``code``, ``status_code``) before
        falling back to a conservative string search for ``503``.

        Args:
            err: The exception raised by the upstream API call.

        Returns:
            ``True`` if the error looks like a transient service error.
        """
        if hasattr(err, "code") and getattr(err, "code", None) == 503:
            return True
        if hasattr(err, "status_code") and getattr(err, "status_code", None) == 503:
            return True
        return "503" in str(err)

    @overload
    def generate(
        self,
        prompt: str,
        response_schema: type[T],
    ) -> T:
        ...

    @overload
    def generate(
        self,
        prompt: str,
        response_schema: None = None,
    ) -> str:
        ...

    def generate(
        self,
        prompt: str,
        response_schema: type[T] | None = None,
    ) -> T | str:
        """Sends a prompt to the LLM and validates the response.

        Args:
            prompt: The text prompt to send.
            response_schema: A Pydantic model class used to parse and
                validate the JSON response. When provided, the return type
                is the schema type; when ``None``, returns raw text.

        Returns:
            A validated instance of ``response_schema`` if provided,
            otherwise the raw response text string.

        Raises:
            ValueError: If JSON parsing or Pydantic validation fails.
            Exception: Propagates from the upstream API on non-retryable errors.
        """
        llm_config: dict[str, Any] = {"response_mime_type": "application/json"}
        response_schema_name: str | None = None
        if response_schema:
            response_schema_dict = response_schema.model_json_schema()
            response_schema_name = response_schema.__name__

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

            _clean_schema(response_schema_dict)
            llm_config["response_schema"] = response_schema_dict

        max_retries = settings.max_retries
        backoff_delay = settings.retry_delay

        raw_response_text = ""
        call_start = time.perf_counter()
        for retry_attempt in range(max_retries):
            try:
                logger.debug(
                    "generate started agent=%s model=%s attempt=%d schema=%s",
                    self.__class__.__name__,
                    self.model_name,
                    retry_attempt + 1,
                    response_schema_name or "raw",
                )
                api_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=llm_config,
                )
                raw_response_text = api_response.text
                break
            except Exception as e:
                if self._is_retryable(e) and retry_attempt < max_retries - 1:
                    logger.warning(
                        "Gemini API 503 (attempt %d/%d). Retrying in %ds. "
                        "agent=%s model=%s error=%s",
                        retry_attempt + 1,
                        max_retries,
                        backoff_delay,
                        self.__class__.__name__,
                        self.model_name,
                        str(e)[:100],
                    )
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                    continue
                logger.exception(
                    "generate failed agent=%s model=%s attempt=%d error=%s",
                    self.__class__.__name__,
                    self.model_name,
                    retry_attempt + 1,
                    str(e)[:200],
                )
                raise

        elapsed = time.perf_counter() - call_start
        logger.debug(
            "generate completed agent=%s model=%s schema=%s elapsed=%.3fs",
            self.__class__.__name__,
            self.model_name,
            response_schema_name or "raw",
            elapsed,
        )

        if response_schema:
            return response_schema.model_validate_json(raw_response_text)
        return raw_response_text
