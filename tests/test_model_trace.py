from pathlib import Path

import pytest

from lanternquest.llm import LLMRequest, LLMResponse
from lanternquest.model_trace import ModelTraceAdapter, ModelTraceDocument, ModelTraceError


class FakeAdapter:
    model_id = "trace-test-model"
    profile_id = "trace-test-profile"

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            model_id=self.model_id,
            content={"purpose": request.purpose, "value": len(self.calls)},
            input_tokens=10,
            output_tokens=3,
            response_id=f"response-{len(self.calls)}",
        )


def make_request(purpose: str = "plan_generation", *, value: int = 1) -> LLMRequest:
    return LLMRequest(
        purpose=purpose,
        system_instruction="Return the requested structured object.",
        payload={"value": value},
        response_schema={"type": "object"},
    )


def record_trace(path: Path) -> ModelTraceDocument:
    adapter = ModelTraceAdapter(FakeAdapter(), mode="record")
    adapter.begin_episode(path)
    adapter.complete(make_request(value=1))
    adapter.complete(make_request("environment_action", value=2))
    return adapter.end_episode()


def test_record_and_exact_prefix_replay_do_not_call_delegate(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source = record_trace(source_path)
    delegate = FakeAdapter()
    replay = ModelTraceAdapter(delegate, mode="replay_prefix")
    replay.begin_episode(tmp_path / "replayed.json", source_trace_path=source_path)

    first = replay.complete(make_request(value=1))
    second = replay.complete(make_request("environment_action", value=2))
    result = replay.end_episode()

    assert source.prefix_complete is True
    assert delegate.calls == []
    assert first.response_id == "response-1"
    assert second.response_id == "response-2"
    assert result.prefix_complete is True
    assert result.source_consumed_count == 2
    assert [entry.source for entry in result.entries] == ["replay", "replay"]


def test_replay_allows_residual_repair_only_after_complete_prefix(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    record_trace(source_path)
    delegate = FakeAdapter()
    replay = ModelTraceAdapter(delegate, mode="replay_prefix")
    replay.begin_episode(tmp_path / "branched.json", source_trace_path=source_path)
    replay.complete(make_request(value=1))
    replay.complete(make_request("environment_action", value=2))

    repair = replay.complete(make_request("residual_repair", value=3))
    result = replay.end_episode()

    assert repair.content["purpose"] == "residual_repair"
    assert len(delegate.calls) == 1
    assert result.divergence_purpose == "residual_repair"
    assert [entry.source for entry in result.entries] == [
        "replay",
        "replay",
        "live",
    ]


def test_replay_rejects_request_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    record_trace(source_path)
    replay = ModelTraceAdapter(FakeAdapter(), mode="replay_prefix")
    replay.begin_episode(tmp_path / "mismatch.json", source_trace_path=source_path)

    with pytest.raises(ModelTraceError, match="request mismatch"):
        replay.complete(make_request(value=99))

    result = replay.end_episode()
    assert result.mismatch_count == 1
    assert result.prefix_complete is False


def test_replay_rejects_early_residual_repair(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    record_trace(source_path)
    replay = ModelTraceAdapter(FakeAdapter(), mode="replay_prefix")
    replay.begin_episode(tmp_path / "early.json", source_trace_path=source_path)

    with pytest.raises(ModelTraceError, match="before the recorded prefix"):
        replay.complete(make_request("residual_repair"))

    replay.end_episode()


def test_replay_hash_ignores_scienceworld_presentation_order(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_adapter = ModelTraceAdapter(FakeAdapter(), mode="record")
    source_adapter.begin_episode(source_path)
    source_adapter.complete(
        LLMRequest(
            purpose="plan_generation",
            system_instruction="Plan.",
            payload={
                "current_observation": "Room.\n\ta blue cup\n\ta red cup",
                "legal_actions": ["take red cup", "take blue cup"],
                "observed_facts": [
                    {"subject": "red cup", "relation": "visible_in"},
                    {"subject": "blue cup", "relation": "visible_in"},
                ],
            },
            response_schema={"type": "object"},
        )
    )
    source_adapter.end_episode()

    delegate = FakeAdapter()
    replay = ModelTraceAdapter(delegate, mode="replay_prefix")
    replay.begin_episode(tmp_path / "replay.json", source_trace_path=source_path)
    replay.complete(
        LLMRequest(
            purpose="plan_generation",
            system_instruction="Plan.",
            payload={
                "current_observation": "Room.\n\ta red cup\n\ta blue cup",
                "legal_actions": ["take blue cup", "take red cup"],
                "observed_facts": [
                    {"subject": "blue cup", "relation": "visible_in"},
                    {"subject": "red cup", "relation": "visible_in"},
                ],
            },
            response_schema={"type": "object"},
        )
    )
    result = replay.end_episode()

    assert delegate.calls == []
    assert result.prefix_complete is True
    assert result.mismatch_count == 0


def test_replay_until_divergence_reuses_shared_prefix_then_stays_live(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    record_trace(source_path)
    delegate = FakeAdapter()
    replay = ModelTraceAdapter(delegate, mode="replay_until_divergence")
    replay.begin_episode(tmp_path / "branched.json", source_trace_path=source_path)

    shared = replay.complete(make_request(value=1))
    changed = replay.complete(make_request("environment_action", value=99))
    later = replay.complete(make_request("plan_generation", value=3))
    result = replay.end_episode()

    assert shared.response_id == "response-1"
    assert changed.response_id == "response-1"
    assert later.response_id == "response-2"
    assert len(delegate.calls) == 2
    assert result.divergence_purpose == "environment_action"
    assert result.mismatch_count == 0
    assert result.source_consumed_count == 1
    assert result.prefix_complete is False
    assert [entry.source for entry in result.entries] == ["replay", "live", "live"]
