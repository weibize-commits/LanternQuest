import json
import re
import time
from copy import deepcopy
from typing import Any

from lanternquest.llm import LLMAdapterError, LLMRequest, LLMResponse


class OpenAIResponsesAdapter:
    """Structured-output adapter for the OpenAI Responses API."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int = 2000,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        structured_output_mode: str = "json_schema",
        profile_id: str | None = None,
        max_input_characters: int | None = None,
        max_retries: int | None = None,
        structured_output_retries: int = 0,
        chat_template_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        minimum_call_interval_seconds: float = 0.0,
        api_style: str = "responses",
        system_instruction_suffix: str = "",
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("An explicit LLM model identifier is required")
        if max_output_tokens < 128:
            raise ValueError("max_output_tokens must be at least 128")
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if structured_output_mode not in {"json_schema", "json_object"}:
            raise ValueError(
                "structured_output_mode must be json_schema or json_object"
            )
        if max_input_characters is not None and max_input_characters < 1000:
            raise ValueError("max_input_characters must be at least 1000")
        if max_retries is not None and max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if structured_output_retries < 0:
            raise ValueError("structured_output_retries cannot be negative")
        if minimum_call_interval_seconds < 0:
            raise ValueError("minimum_call_interval_seconds cannot be negative")
        if api_style not in {"responses", "chat_completions"}:
            raise ValueError("api_style must be responses or chat_completions")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "OpenAI SDK is not installed; install the project with the llm extra"
                ) from error
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            if max_retries is not None:
                client_kwargs["max_retries"] = max_retries
            client = OpenAI(**client_kwargs)

        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.structured_output_mode = structured_output_mode
        self.profile_id = profile_id
        self.max_input_characters = max_input_characters
        self.max_retries = max_retries
        self.structured_output_retries = structured_output_retries
        self.chat_template_kwargs = chat_template_kwargs
        self.extra_body = extra_body
        self.minimum_call_interval_seconds = minimum_call_interval_seconds
        self._last_provider_call_monotonic: float | None = None
        self.api_style = api_style
        self.system_instruction_suffix = system_instruction_suffix
        self.client = client

    @staticmethod
    def _bounded_copy(
        value: Any,
        *,
        string_limit: int,
        list_limit: int,
        parent_key: str = "",
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: OpenAIResponsesAdapter._bounded_copy(
                    item,
                    string_limit=string_limit,
                    list_limit=list_limit,
                    parent_key=str(key),
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            configured_limit = {
                "candidate_actions": 12,
                "valid_actions": 12,
                "training_skills": 3,
                "active_train_memory": 3,
                "train_memory": 3,
                "replacement_train_memory": 3,
                "observed_facts": 32,
                "procedure_actions": 24,
                "actions": 24,
            }.get(parent_key, list_limit)
            key_limit = min(configured_limit, list_limit)
            return [
                OpenAIResponsesAdapter._bounded_copy(
                    item,
                    string_limit=string_limit,
                    list_limit=list_limit,
                    parent_key=parent_key,
                )
                for item in value[: min(len(value), key_limit)]
            ]
        if isinstance(value, str):
            key_limit = {
                "task_description": 1600,
                "observation": 2400,
                "current_observation": 2400,
                "inventory": 1200,
                "action": 512,
            }.get(parent_key, string_limit)
            if len(value) <= key_limit:
                return value
            marker = " ...[context clipped]... "
            head = max(1, (key_limit - len(marker)) * 3 // 4)
            tail = max(1, key_limit - len(marker) - head)
            return value[:head] + marker + value[-tail:]
        return deepcopy(value)

    def _serialize_payload(self, payload: dict[str, Any]) -> tuple[str, bool]:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if self.max_input_characters is None or len(serialized) <= self.max_input_characters:
            return serialized, False

        for string_limit, list_limit in (
            (2000, 24),
            (1200, 16),
            (700, 12),
            (400, 8),
            (240, 4),
        ):
            compacted = self._bounded_copy(
                payload,
                string_limit=string_limit,
                list_limit=list_limit,
            )
            serialized = json.dumps(compacted, ensure_ascii=False, sort_keys=True)
            if len(serialized) <= self.max_input_characters:
                return serialized, True
        raise ValueError(
            "Structured request still exceeds max_input_characters after deterministic "
            "compaction"
        )

    @staticmethod
    def _decode_json_object(output_text: str) -> tuple[dict[str, Any], str | None]:
        try:
            content = json.loads(output_text)
            if not isinstance(content, dict):
                raise ValueError("OpenAI response JSON must be an object")
            return content, None
        except json.JSONDecodeError as original_error:
            stripped = output_text.strip()
            if stripped.startswith("```") and stripped.endswith("```"):
                lines = stripped.splitlines()
                if len(lines) >= 3:
                    stripped = "\n".join(lines[1:-1]).strip()
            start = stripped.find("{")
            if start >= 0:
                try:
                    content, _ = json.JSONDecoder().raw_decode(stripped[start:])
                    if isinstance(content, dict):
                        return content, "extracted_json_object"
                except json.JSONDecodeError:
                    pass
            preview = output_text[:240].replace("\r", " ").replace("\n", " ")
            raise ValueError(
                f"OpenAI response was not valid JSON; prefix={preview!r}"
            ) from original_error

    def complete(self, request: LLMRequest) -> LLMResponse:
        schema_name = re.sub(r"[^a-zA-Z0-9_-]", "_", request.purpose)[:64]
        instructions = request.system_instruction
        if self.structured_output_mode == "json_object":
            instructions += (
                " Return only one JSON object containing values that conform to this "
                "JSON Schema. Do not return or describe the schema itself: "
                + json.dumps(request.response_schema, ensure_ascii=False, sort_keys=True)
            )
            text_format: dict[str, Any] = {"type": "json_object"}
        else:
            text_format = {
                "type": "json_schema",
                "name": f"lanternquest_{schema_name}"[:64],
                "schema": request.response_schema,
                "strict": True,
            }
        if self.system_instruction_suffix:
            instructions = f"{instructions} {self.system_instruction_suffix.strip()}"
        serialized_payload, input_was_compacted = self._serialize_payload(
            request.payload
        )
        if self.api_style == "responses":
            request_kwargs: dict[str, Any] = {
                "model": self.model_id,
                "instructions": instructions,
                "input": serialized_payload,
                "max_output_tokens": self.max_output_tokens,
                "store": False,
                "text": {"format": text_format},
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature
            if self.reasoning_effort:
                request_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            response_format = text_format
            if text_format["type"] == "json_schema":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": text_format["name"],
                        "schema": text_format["schema"],
                        "strict": text_format["strict"],
                    },
                }
            request_kwargs = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": serialized_payload},
                ],
                "max_tokens": self.max_output_tokens,
                "response_format": response_format,
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature
            if self.reasoning_effort:
                request_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p
        extra_body = dict(self.extra_body or {})
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        provider_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        response = None
        content = None
        output_normalization = None
        last_error: ValueError | None = None
        for attempt in range(self.structured_output_retries + 1):
            if self._last_provider_call_monotonic is not None:
                elapsed = time.monotonic() - self._last_provider_call_monotonic
                remaining = self.minimum_call_interval_seconds - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            self._last_provider_call_monotonic = time.monotonic()
            provider_calls += 1
            if self.api_style == "responses":
                response = self.client.responses.create(**request_kwargs)
                output_text = response.output_text
            else:
                response = self.client.chat.completions.create(**request_kwargs)
                output_text = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            total_input_tokens += (
                getattr(usage, "input_tokens", None)
                or getattr(usage, "prompt_tokens", None)
                or 0
            )
            total_output_tokens += (
                getattr(usage, "output_tokens", None)
                or getattr(usage, "completion_tokens", None)
                or 0
            )
            if not output_text:
                last_error = ValueError(
                    "OpenAI response did not contain structured output text"
                )
            else:
                try:
                    content, output_normalization = self._decode_json_object(output_text)
                    break
                except ValueError as error:
                    last_error = error
            if attempt < self.structured_output_retries:
                continue
        if content is None or response is None:
            assert last_error is not None
            raise LLMAdapterError(
                str(last_error),
                provider_calls=provider_calls,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ) from last_error
        normalizations = []
        if input_was_compacted:
            normalizations.append("input_context_compacted")
        if output_normalization:
            normalizations.append(output_normalization)
        if provider_calls > 1:
            normalizations.append("structured_output_retry")
        required_fields = request.response_schema.get("required", [])
        wrapped_properties = content.get("properties") if isinstance(content, dict) else None
        if (
            isinstance(required_fields, list)
            and isinstance(wrapped_properties, dict)
            and not all(field in content for field in required_fields)
            and all(field in wrapped_properties for field in required_fields)
        ):
            content = wrapped_properties
            normalizations.append("unwrapped_schema_properties")

        return LLMResponse(
            model_id=str(getattr(response, "model", self.model_id)),
            content=content,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            response_id=getattr(response, "id", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            normalization_applied=("+".join(normalizations) if normalizations else None),
            provider_calls=provider_calls,
        )
