import json
from types import SimpleNamespace

import pytest

from lanternquest.llm import LLMAdapterError, LLMRequest
from lanternquest.openai_adapter import OpenAIResponsesAdapter


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            id="resp-test",
            output_text=json.dumps({"answer": "ok"}),
            model="test-model-snapshot",
            system_fingerprint="fp-test",
            usage=SimpleNamespace(input_tokens=11, output_tokens=4),
        )


class FakeChatCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            id="chat-test",
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"answer": "ok"}'))
            ],
            model="test-chat-model",
            system_fingerprint="chat-fp",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
        )


def make_request() -> LLMRequest:
    return LLMRequest(
        purpose="grounded_explanation",
        system_instruction="Use only the supplied decision.",
        payload={"decision": {"decision_type": "clarify"}},
        response_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
    )


def test_openai_adapter_uses_structured_outputs_without_storage() -> None:
    fake_responses = FakeResponses()
    client = SimpleNamespace(responses=fake_responses)
    adapter = OpenAIResponsesAdapter(
        "test-model",
        temperature=0.0,
        top_p=0.95,
        reasoning_effort="none",
        client=client,
    )

    result = adapter.complete(make_request())

    assert result.content == {"answer": "ok"}
    assert result.model_id == "test-model-snapshot"
    assert result.input_tokens == 11
    assert result.output_tokens == 4
    assert result.response_id == "resp-test"
    assert result.system_fingerprint == "fp-test"
    assert fake_responses.kwargs["store"] is False
    assert fake_responses.kwargs["temperature"] == 0.0
    assert fake_responses.kwargs["top_p"] == 0.95
    assert fake_responses.kwargs["reasoning"] == {"effort": "none"}
    text_config = fake_responses.kwargs["text"]
    assert isinstance(text_config, dict)
    assert text_config["format"]["type"] == "json_schema"
    assert text_config["format"]["strict"] is True


def test_openai_adapter_passes_chat_template_controls() -> None:
    fake_responses = FakeResponses()
    adapter = OpenAIResponsesAdapter(
        "test-model",
        chat_template_kwargs={"enable_thinking": False},
        client=SimpleNamespace(responses=fake_responses),
    )

    adapter.complete(make_request())

    assert fake_responses.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_openai_adapter_supports_chat_completions_style() -> None:
    completions = FakeChatCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    adapter = OpenAIResponsesAdapter(
        "test-model",
        api_style="chat_completions",
        chat_template_kwargs={"enable_thinking": False},
        extra_body={"thinking": {"type": "disabled"}},
        system_instruction_suffix="/no_think",
        client=client,
    )

    result = adapter.complete(make_request())

    assert result.content == {"answer": "ok"}
    assert result.input_tokens == 12
    assert result.output_tokens == 5
    assert completions.kwargs["messages"][1]["role"] == "user"
    assert completions.kwargs["messages"][0]["content"].endswith("/no_think")
    assert completions.kwargs["response_format"]["type"] == "json_schema"
    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }


def test_openai_adapter_requires_explicit_model() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAIResponsesAdapter("", client=SimpleNamespace())


def test_openai_adapter_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        OpenAIResponsesAdapter("test-model", temperature=2.1, client=SimpleNamespace())


def test_openai_adapter_omits_unsupported_temperature() -> None:
    fake_responses = FakeResponses()
    adapter = OpenAIResponsesAdapter(
        "test-model",
        temperature=None,
        client=SimpleNamespace(responses=fake_responses),
    )

    adapter.complete(make_request())

    assert "temperature" not in fake_responses.kwargs


def test_openai_adapter_rejects_invalid_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        OpenAIResponsesAdapter("test-model", top_p=0.0, client=SimpleNamespace())


def test_openai_adapter_accepts_provider_compatible_base_url_with_client() -> None:
    fake_responses = FakeResponses()
    adapter = OpenAIResponsesAdapter(
        "compatible-model",
        base_url="https://example.invalid/v1",
        client=SimpleNamespace(responses=fake_responses),
    )

    result = adapter.complete(make_request())

    assert result.content == {"answer": "ok"}


def test_openai_adapter_rejects_non_json_output() -> None:
    responses = FakeResponses()

    def invalid_create(**kwargs: object) -> SimpleNamespace:
        responses.kwargs = kwargs
        return SimpleNamespace(output_text="not-json", model="test", usage=None)

    responses.create = invalid_create  # type: ignore[method-assign]
    adapter = OpenAIResponsesAdapter(
        "test-model", client=SimpleNamespace(responses=responses)
    )

    with pytest.raises(ValueError, match="valid JSON.*not-json"):
        adapter.complete(make_request())


def test_openai_adapter_unwraps_schema_shaped_provider_output() -> None:
    responses = FakeResponses()

    def wrapped_create(**kwargs: object) -> SimpleNamespace:
        responses.kwargs = kwargs
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": "ok"},
                }
            ),
            model="test-model",
            usage=None,
        )

    responses.create = wrapped_create  # type: ignore[method-assign]
    adapter = OpenAIResponsesAdapter(
        "test-model", client=SimpleNamespace(responses=responses)
    )

    result = adapter.complete(make_request())

    assert result.content == {"answer": "ok"}
    assert result.normalization_applied == "unwrapped_schema_properties"


def test_openai_adapter_supports_json_object_mode_with_schema_instruction() -> None:
    responses = FakeResponses()
    adapter = OpenAIResponsesAdapter(
        "test-model",
        structured_output_mode="json_object",
        client=SimpleNamespace(responses=responses),
    )

    adapter.complete(make_request())

    assert responses.kwargs["text"] == {"format": {"type": "json_object"}}
    assert "Do not return or describe the schema itself" in str(
        responses.kwargs["instructions"]
    )


def test_openai_adapter_compacts_large_structured_inputs() -> None:
    responses = FakeResponses()
    adapter = OpenAIResponsesAdapter(
        "test-model",
        max_input_characters=1200,
        client=SimpleNamespace(responses=responses),
    )
    request = make_request().model_copy(
        update={
            "payload": {
                "task_description": "test task",
                "training_skills": [
                    {"procedure_actions": ["look around"] * 100, "notes": "x" * 4000}
                    for _ in range(8)
                ],
            }
        }
    )

    result = adapter.complete(request)

    assert len(str(responses.kwargs["input"])) <= 1200
    assert result.normalization_applied == "input_context_compacted"


def test_openai_adapter_extracts_json_from_fenced_provider_output() -> None:
    responses = FakeResponses()

    def fenced_create(**kwargs: object) -> SimpleNamespace:
        responses.kwargs = kwargs
        return SimpleNamespace(
            output_text='```json\n{"answer": "ok"}\n```',
            model="test-model",
            usage=None,
        )

    responses.create = fenced_create  # type: ignore[method-assign]
    adapter = OpenAIResponsesAdapter(
        "test-model", client=SimpleNamespace(responses=responses)
    )

    result = adapter.complete(make_request())

    assert result.content == {"answer": "ok"}
    assert result.normalization_applied == "extracted_json_object"


def test_openai_adapter_retries_empty_structured_output_and_counts_calls() -> None:
    responses = FakeResponses()
    call_count = 0

    def retry_create(**kwargs: object) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        responses.kwargs = kwargs
        return SimpleNamespace(
            output_text="" if call_count == 1 else '{"answer": "ok"}',
            model="test-model",
            usage=SimpleNamespace(input_tokens=11, output_tokens=4),
        )

    responses.create = retry_create  # type: ignore[method-assign]
    adapter = OpenAIResponsesAdapter(
        "test-model",
        structured_output_retries=1,
        client=SimpleNamespace(responses=responses),
    )

    result = adapter.complete(make_request())

    assert result.provider_calls == 2
    assert result.input_tokens == 22
    assert result.output_tokens == 8
    assert result.normalization_applied == "structured_output_retry"


def test_openai_adapter_reports_exhausted_structured_output_retries() -> None:
    responses = FakeResponses()

    def empty_create(**kwargs: object) -> SimpleNamespace:
        responses.kwargs = kwargs
        return SimpleNamespace(
            output_text="",
            model="test-model",
            usage=SimpleNamespace(input_tokens=7, output_tokens=0),
        )

    responses.create = empty_create  # type: ignore[method-assign]
    adapter = OpenAIResponsesAdapter(
        "test-model",
        structured_output_retries=1,
        client=SimpleNamespace(responses=responses),
    )

    with pytest.raises(LLMAdapterError) as error:
        adapter.complete(make_request())

    assert error.value.provider_calls == 2
    assert error.value.input_tokens == 14


def test_openai_adapter_paces_structured_output_retries(monkeypatch) -> None:
    responses = FakeResponses()
    call_count = 0

    def retry_create(**kwargs: object) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(
            output_text="" if call_count == 1 else '{"answer": "ok"}',
            model="test-model",
            usage=None,
        )

    responses.create = retry_create  # type: ignore[method-assign]
    clock = iter((100.0, 102.0, 105.0))
    sleeps: list[float] = []
    monkeypatch.setattr("lanternquest.openai_adapter.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("lanternquest.openai_adapter.time.sleep", sleeps.append)
    adapter = OpenAIResponsesAdapter(
        "test-model",
        structured_output_retries=1,
        minimum_call_interval_seconds=5.0,
        client=SimpleNamespace(responses=responses),
    )

    result = adapter.complete(make_request())

    assert result.provider_calls == 2
    assert sleeps == [3.0]
