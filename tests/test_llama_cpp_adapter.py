import json
from pathlib import Path

import pytest

from lanternquest.llama_cpp_adapter import LlamaCppAdapter
from lanternquest.llm import LLMAdapterError, LLMRequest


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs = None

    def create_chat_completion(self, **kwargs):
        self.kwargs = kwargs
        return {
            "id": "local-1",
            "model": "local-test",
            "choices": [{"message": {"content": self.content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        }


def request() -> LLMRequest:
    return LLMRequest(
        purpose="plan_generation",
        system_instruction="Return a decision.",
        payload={"case_id": "case-1"},
        response_schema={
            "type": "object",
            "properties": {"decision_type": {"type": "string"}},
            "required": ["decision_type"],
        },
    )


def test_local_adapter_uses_schema_and_reports_model_fingerprint() -> None:
    client = FakeClient(json.dumps({"decision_type": "clarify"}))
    adapter = LlamaCppAdapter(
        Path("unused.gguf"),
        model_id="qwen-local",
        model_sha256="a" * 64,
        client=client,
    )

    response = adapter.complete(request())

    assert response.content == {"decision_type": "clarify"}
    assert response.input_tokens == 12
    assert response.output_tokens == 5
    assert response.system_fingerprint == f"sha256:{'a' * 64}"
    assert client.kwargs["response_format"]["schema"] == request().response_schema


def test_local_adapter_rejects_non_json_output() -> None:
    adapter = LlamaCppAdapter(
        Path("unused.gguf"),
        model_id="qwen-local",
        model_sha256="b" * 64,
        client=FakeClient("not json"),
    )

    with pytest.raises(LLMAdapterError):
        adapter.complete(request())
