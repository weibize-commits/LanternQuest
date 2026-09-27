import hashlib
from pathlib import Path

import pytest

from lanternquest.llm import LLMResponse
from lanternquest.model_trace import ModelTraceAdapter
from scripts.scienceworld_ensr_components import FactPattern, parse_observation_facts
from scripts.scienceworld_ensr_harness import (
    ENSRBudget,
    ENSRHarness,
    ENSRMemory,
    HierarchicalPlan,
    PlannerSubgoal,
    _canonicalize_model_payload,
    _conditional_conductivity_boxes,
    _conductivity_conditioned_move_action,
    _conductivity_detector_result,
    _conductivity_probe_action,
    _conductivity_probe_components,
    _facts_satisfied,
    _failed_devices_from_observation,
    _filter_irreversible_focus_actions,
    _focus_target_stages,
    _goal_directed_action,
    _local_replan_has_budget,
    _measurement_conditioned_focus_action,
    _measurement_decision_temperature,
    _measurement_poll_action,
    _next_procedure_action,
    _procedure_action_domain,
    _procedure_task_target,
    _repair_expected_locations,
    _semantic_acquisition_action,
    _semantic_destination,
    _semantic_focus_actions,
    _semantic_move_to_box_action,
    _semantic_task_kind,
    _state_change_workflow_action,
    _systematic_exploration_action,
    _task_target_acquisition_action,
    _update_entity_locations,
)

CLOSED = """This room is called the hallway. In it, you see:
\tthe agent
You also see:
\tA door to the kitchen (that is closed)
"""
OPEN = CLOSED.replace("that is closed", "that is open")
INVENTORY = "In your inventory, you see:\n\tnothing\n"


def _memory() -> ENSRMemory:
    return ENSRMemory(
        [
            {
                "fragment_id": "fragment-open-door",
                "task_id": "1-1",
                "task_name": "boil",
                "variation": 0,
                "step_index": 1,
                "action_template": "open door to kitchen",
                "preconditions": [],
                "effects": [
                    {
                        "subject": "door to the kitchen",
                        "relation": "has_state",
                        "object": "open",
                        "polarity": True,
                    }
                ],
                "outcome": {"reward": 0, "score": 0, "completed": False},
            }
        ]
    )


def _plan_payload() -> dict:
    return {
        "task_summary": "Open the kitchen door.",
        "subgoals": [
            {
                "subgoal_id": "sg-open",
                "description": "Open the kitchen door.",
                "expected_facts": [
                    {
                        "subject": "door to the kitchen",
                        "relation": "has_state",
                        "object": "open",
                        "polarity": True,
                    }
                ],
                "completion_test": "The kitchen door has_state open.",
                "retry_limit": 3,
            }
        ],
    }


def _open_action_payload() -> dict:
    return {
        "action": "open door to kitchen",
        "expected_fact": {
            "subject": "door to the kitchen",
            "relation": "has_state",
            "object": "open",
            "polarity": True,
        },
        "expected_progress": "The door becomes open.",
    }


class FakeAdapter:
    model_id = "fake-model"
    profile_id = "fake-profile"

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            model_id=self.model_id,
            content=self.payloads.pop(0),
            input_tokens=10,
            output_tokens=5,
            system_fingerprint="fake-fingerprint",
        )


class FailingAdapter(FakeAdapter):
    def complete(self, request):
        self.requests.append(request)
        raise ValueError("malformed provider JSON")


class CandidateRankAdapter(FakeAdapter):
    def complete(self, request):
        self.requests.append(request)
        if request.purpose == "candidate_ranking":
            content = {
                "candidate_id": request.payload["verified_candidates"][0][
                    "candidate_id"
                ]
            }
        else:
            content = self.payloads.pop(0)
        return LLMResponse(
            model_id=self.model_id,
            content=content,
            input_tokens=10,
            output_tokens=5,
            system_fingerprint="fake-fingerprint",
        )


class OpenDoorEnvironment:
    def load(self, *args, **kwargs) -> None:
        return None

    def reset(self):
        return CLOSED, {
            "taskDesc": "Open the kitchen door.",
            "look": CLOSED,
            "inv": INVENTORY,
            "valid": ["look around", "open door to kitchen"],
            "score": 0,
        }

    def step(self, action: str):
        assert action == "open door to kitchen"
        return "You open the door.", 100, True, {
            "look": OPEN,
            "inv": INVENTORY,
            "valid": ["look around", "close door to kitchen"],
            "score": 100,
        }


class RecoveringEnvironment(OpenDoorEnvironment):
    def __init__(self) -> None:
        self.steps = 0

    def step(self, action: str):
        self.steps += 1
        if self.steps == 1:
            assert action == "look around"
            return CLOSED, 0, False, {
                "look": CLOSED,
                "inv": INVENTORY,
                "valid": ["look around", "open door to kitchen"],
                "score": 0,
            }
        return super().step(action)


class TwoStageEnvironment(OpenDoorEnvironment):
    def __init__(self) -> None:
        self.steps = 0

    def step(self, action: str):
        self.steps += 1
        if self.steps == 1:
            assert action == "open door to kitchen"
            return "You open the door.", 50, False, {
                "look": OPEN,
                "inv": INVENTORY,
                "valid": ["go to kitchen"],
                "score": 50,
            }
        assert action == "go to kitchen"
        kitchen = "This room is called the kitchen. In it, you see:\n\tthe agent\n"
        return "You enter the kitchen.", 50, True, {
            "look": kitchen,
            "inv": INVENTORY,
            "valid": ["look around"],
            "score": 100,
        }


def _run(adapter: FakeAdapter, environment):
    return ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
    ).run_episode(
        environment,
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )


def test_ensr_harness_verifies_an_expected_transition() -> None:
    result = _run(FakeAdapter([_plan_payload(), _open_action_payload()]), OpenDoorEnvironment())

    assert result.status == "terminal_success"
    assert result.task_success
    assert result.model_calls == 2
    assert result.mechanism_metrics["verified_transition_count"] == 1
    assert result.completed_subgoal_ids == ["sg-open"]


def test_hierarchy_ablation_collapses_a_generated_plan_to_one_task_subgoal() -> None:
    payload = _plan_payload()
    payload["subgoals"].insert(
        0,
        {
            "subgoal_id": "sg-inspect",
            "description": "Inspect the hallway.",
            "expected_facts": [
                {
                    "subject": "agent",
                    "relation": "located_in",
                    "object": "hallway",
                    "polarity": True,
                }
            ],
            "completion_test": "The hallway has been inspected.",
            "retry_limit": 2,
        },
    )
    harness = ENSRHarness(
        _memory(),
        FakeAdapter([]),
        enable_hierarchical_subgoals=False,
    )

    collapsed = harness._apply_hierarchy_policy(
        HierarchicalPlan.model_validate(payload),
        task_description="Open the kitchen door.",
    )

    assert len(collapsed.subgoals) == 1
    assert collapsed.subgoals[0].subgoal_id == "sg-task"
    assert collapsed.subgoals[0].description == "Complete the task: Open the kitchen door."
    assert collapsed.subgoals[0].expected_facts[0].object == "open"
    assert collapsed.subgoals[0].retry_limit == 3


def test_invalid_fragment_memory_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="replace or accumulate"):
        ENSRHarness(_memory(), FakeAdapter([]), fragment_memory_policy="invalid")


def test_transition_verifier_ablation_records_no_verified_transition() -> None:
    result = ENSRHarness(
        _memory(),
        FakeAdapter([_plan_payload(), _open_action_payload()]),
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_symbolic_transition_verifier=False,
    ).run_episode(
        OpenDoorEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.mechanism_metrics["verified_transition_count"] == 0
    assert result.mechanism_metrics["symbolic_transition_verifier_enabled"] == 0


def test_ensr_harness_retrieves_and_replans_after_no_progress() -> None:
    local_replan = {
        "diagnosis": "Looking again did not change the state.",
        "replacement_subgoal": _plan_payload()["subgoals"][0],
    }
    adapter = FakeAdapter(
        [
            _plan_payload(),
            {
                "action": "look around",
                "expected_fact": None,
                "expected_progress": "Inspect the current state.",
            },
            local_replan,
            _open_action_payload(),
        ]
    )
    result = _run(adapter, RecoveringEnvironment())

    assert result.status == "terminal_success"
    assert result.retrieval_calls == 2
    assert result.mechanism_metrics["obligation_count"] == 1
    assert result.obligations[0].source == "no_progress"
    assert result.mechanism_metrics["local_replan_count"] == 1
    assert result.mechanism_metrics["local_replan_recovery_count"] == 1


def test_obligation_retrieval_ablation_keeps_replanning_without_retrieval() -> None:
    local_replan = {
        "diagnosis": "Looking again did not change the state.",
        "replacement_subgoal": _plan_payload()["subgoals"][0],
    }
    adapter = FakeAdapter(
        [
            _plan_payload(),
            {
                "action": "look around",
                "expected_fact": None,
                "expected_progress": "Inspect the current state.",
            },
            local_replan,
            _open_action_payload(),
        ]
    )
    result = ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_obligation_conditioned_retrieval=False,
    ).run_episode(
        RecoveringEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.retrieval_calls == 1
    assert all(event.trigger != "obligation" for event in result.retrieval_events)
    assert result.mechanism_metrics["obligation_count"] == 1
    assert result.mechanism_metrics["local_replan_count"] == 1
    assert result.mechanism_metrics["obligation_conditioned_retrieval_enabled"] == 0


def test_evidence_debt_scheduler_reuses_redundant_memory_and_keeps_replanning() -> None:
    local_replan = {
        "diagnosis": "Looking again did not change the state.",
        "replacement_subgoal": _plan_payload()["subgoals"][0],
    }
    result = ENSRHarness(
        _memory(),
        FakeAdapter(
            [
                _plan_payload(),
                {
                    "action": "look around",
                    "expected_fact": None,
                    "expected_progress": "Inspect the current state.",
                },
                local_replan,
                _open_action_payload(),
            ]
        ),
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_evidence_debt_scheduler=True,
    ).run_episode(
        RecoveringEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.retrieval_calls == 1
    assert result.mechanism_metrics["local_replan_count"] == 1
    assert result.mechanism_metrics["evidence_debt_scheduler_enabled"] == 1
    assert result.mechanism_metrics["evidence_scheduler_reuse_count"] == 1
    assert result.evidence_schedule_events[0].reason == "candidate_evidence_redundant"


def test_accumulating_fragment_memory_does_not_replace_prior_fragments() -> None:
    first_fragment = _memory().fragments[0]
    second_fragment = {
        **first_fragment,
        "fragment_id": "fragment-recovery-look",
        "action_template": "look around",
    }

    class SequencedMemory(ENSRMemory):
        def __init__(self) -> None:
            super().__init__([first_fragment, second_fragment])
            self.search_count = 0

        def search(self, query: str, *, task_id: str, limit: int) -> list[str]:
            del query, task_id, limit
            self.search_count += 1
            return [
                "fragment-open-door"
                if self.search_count == 1
                else "fragment-recovery-look"
            ]

    local_replan = {
        "diagnosis": "Looking again did not change the state.",
        "replacement_subgoal": _plan_payload()["subgoals"][0],
    }
    result = ENSRHarness(
        SequencedMemory(),
        FakeAdapter(
            [
                _plan_payload(),
                {
                    "action": "look around",
                    "expected_fact": None,
                    "expected_progress": "Inspect the current state.",
                },
                local_replan,
                _open_action_payload(),
            ]
        ),
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        fragment_memory_policy="accumulate",
    ).run_episode(
        RecoveringEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    obligation_event = next(
        event for event in result.retrieval_events if event.trigger == "obligation"
    )
    assert result.status == "terminal_success"
    assert obligation_event.new_fragment_ids == ["fragment-recovery-look"]
    assert obligation_event.replaced_fragment_ids == []
    assert result.mechanism_metrics["fragment_memory_accumulate_enabled"] == 1


def test_adaptive_slow_path_continues_after_incomplete_plan() -> None:
    continuation_plan = {
        "task_summary": "Enter the kitchen after opening its door.",
        "subgoals": [
            {
                "subgoal_id": "sg-enter",
                "description": "Enter the kitchen.",
                "expected_facts": [
                    {
                        "subject": "agent",
                        "relation": "located_in",
                        "object": "kitchen",
                        "polarity": True,
                    }
                ],
                "completion_test": "The agent is located in the kitchen.",
                "retry_limit": 2,
            }
        ],
    }
    adapter = FakeAdapter(
        [
            _plan_payload(),
            _open_action_payload(),
            continuation_plan,
            {
                "action": "go to kitchen",
                "expected_fact": {
                    "subject": "agent",
                    "relation": "located_in",
                    "object": "kitchen",
                    "polarity": True,
                },
                "expected_progress": "Enter the kitchen.",
            },
        ]
    )

    result = ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_adaptive_slow_path=True,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.final_score == 100
    assert result.model_calls == 4
    assert result.planner_calls == 2
    assert len(result.adaptive_recovery_plans) == 1
    assert result.mechanism_metrics["adaptive_continuation_plan_count"] == 1
    assert result.mechanism_metrics["adaptive_continuation_recovery_count"] == 1


def test_bounded_continuation_drops_satisfied_work_and_truncates_horizon() -> None:
    continuation_plan = {
        "task_summary": "Finish entering the kitchen.",
        "subgoals": [
            _plan_payload()["subgoals"][0],
            {
                "subgoal_id": "sg-enter",
                "description": "Enter the kitchen.",
                "expected_facts": [
                    {
                        "subject": "agent",
                        "relation": "located_in",
                        "object": "kitchen",
                        "polarity": True,
                    }
                ],
                "completion_test": "The agent is located in the kitchen.",
                "retry_limit": 2,
            },
            {
                "subgoal_id": "sg-extra",
                "description": "Inspect the kitchen.",
                "expected_facts": [
                    {
                        "subject": "kitchen",
                        "relation": "has_state",
                        "object": "inspected",
                        "polarity": True,
                    }
                ],
                "completion_test": "The kitchen is inspected.",
                "retry_limit": 2,
            },
        ],
    }
    adapter = FakeAdapter(
        [
            _plan_payload(),
            _open_action_payload(),
            continuation_plan,
            {
                "action": "go to kitchen",
                "expected_fact": {
                    "subject": "agent",
                    "relation": "located_in",
                    "object": "kitchen",
                    "polarity": True,
                },
                "expected_progress": "Enter the kitchen.",
            },
        ]
    )

    result = ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_adaptive_slow_path=True,
        max_continuation_subgoals=1,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.mechanism_metrics[
        "adaptive_continuation_satisfied_subgoal_drop_count"
    ] == 1
    assert result.mechanism_metrics[
        "adaptive_continuation_truncated_subgoal_count"
    ] == 1
    assert result.completed_subgoal_ids[-1] == "adaptive-1-1"


def _controlled_repair_payload() -> dict:
    return {
        "diagnosis": "The door is open but the agent has not entered the kitchen.",
        "residual_subgoal": {
            "subgoal_id": "repair-enter",
            "description": "Enter the kitchen.",
            "expected_facts": [
                {
                    "subject": "agent",
                    "relation": "located_in",
                    "object": "kitchen",
                    "polarity": True,
                }
            ],
            "completion_test": "The agent is located in the kitchen.",
            "retry_limit": 2,
        },
        "next_action": "go to kitchen",
        "expected_progress": "Enter the kitchen and finish the task.",
    }


def test_controlled_residual_repair_recovers_after_shared_plan_exhaustion() -> None:
    adapter = FakeAdapter(
        [_plan_payload(), _open_action_payload(), _controlled_repair_payload()]
    )

    result = ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_controlled_residual_repair=True,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.final_score == 100
    assert result.model_calls == 3
    assert result.planner_calls == 1
    assert result.trajectory[-1].action_source == "controlled_repair"
    assert result.mechanism_metrics["controlled_repair_request_count"] == 1
    assert result.mechanism_metrics["controlled_repair_accepted_count"] == 1
    assert result.mechanism_metrics["controlled_repair_action_count"] == 1
    assert result.mechanism_metrics[
        "controlled_repair_verified_progress_count"
    ] == 1


def test_controlled_residual_repair_rejects_already_satisfied_fact() -> None:
    proposal = _controlled_repair_payload()
    proposal["residual_subgoal"] = _plan_payload()["subgoals"][0]
    adapter = FakeAdapter([_plan_payload(), _open_action_payload(), proposal])

    result = ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_controlled_residual_repair=True,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "plan_exhausted"
    assert result.final_score == 50
    assert result.environment_steps == 1
    assert result.mechanism_metrics["controlled_repair_accepted_count"] == 0
    assert result.mechanism_metrics["controlled_repair_rejected_count"] == 1
    assert result.mechanism_metrics[
        "controlled_repair_satisfied_rejection_count"
    ] == 1


def test_d3_replays_exact_d0_prefix_before_live_repair(tmp_path: Path) -> None:
    source_path = tmp_path / "d0_trace.json"
    d0_delegate = FakeAdapter([_plan_payload(), _open_action_payload()])
    d0_adapter = ModelTraceAdapter(d0_delegate, mode="record")
    d0_adapter.begin_episode(source_path)
    d0_result = ENSRHarness(
        _memory(),
        d0_adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )
    d0_adapter.end_episode()

    d3_delegate = FakeAdapter([_controlled_repair_payload()])
    d3_adapter = ModelTraceAdapter(d3_delegate, mode="replay_prefix")
    d3_adapter.begin_episode(
        tmp_path / "d3_trace.json", source_trace_path=source_path
    )
    d3_result = ENSRHarness(
        _memory(),
        d3_adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_controlled_residual_repair=True,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )
    trace = d3_adapter.end_episode()

    assert d0_result.status == "plan_exhausted"
    assert d3_result.status == "terminal_success"
    assert [request.purpose for request in d3_delegate.requests] == [
        "residual_repair"
    ]
    assert trace.prefix_complete is True
    assert trace.mismatch_count == 0
    assert d3_result.mechanism_metrics["model_trace_replayed_call_count"] == 2
    assert d3_result.mechanism_metrics["model_trace_live_call_count"] == 1


def test_d4_ranks_only_verified_candidate_ids_after_plan_exhaustion() -> None:
    adapter = CandidateRankAdapter([_plan_payload(), _open_action_payload()])
    result = ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_verified_graph_repair=True,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.final_score == 100
    assert [request.purpose for request in adapter.requests] == [
        "plan_generation",
        "environment_action",
        "candidate_ranking",
    ]
    rank_request = adapter.requests[-1]
    assert list(rank_request.response_schema["properties"]) == ["candidate_id"]
    assert rank_request.payload["verified_candidates"][0]["actions"] == [
        "go to kitchen"
    ]
    assert result.trajectory[-1].action_source == "verified_graph_repair"
    assert result.mechanism_metrics["verified_graph_rank_request_count"] == 1
    assert result.mechanism_metrics["verified_graph_action_count"] == 1
    assert result.mechanism_metrics[
        "verified_graph_verified_progress_count"
    ] == 1


def test_d4_and_d3_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ENSRHarness(
            _memory(),
            FakeAdapter([]),
            enable_controlled_residual_repair=True,
            enable_verified_graph_repair=True,
        )


def test_d4_replays_exact_d0_model_prefix_before_candidate_ranking(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "d0_trace.json"
    d0_adapter = ModelTraceAdapter(
        FakeAdapter([_plan_payload(), _open_action_payload()]), mode="record"
    )
    d0_adapter.begin_episode(source_path)
    d0_result = ENSRHarness(
        _memory(),
        d0_adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )
    d0_adapter.end_episode()

    delegate = CandidateRankAdapter([])
    d4_adapter = ModelTraceAdapter(delegate, mode="replay_prefix")
    d4_adapter.begin_episode(
        tmp_path / "d4_trace.json", source_trace_path=source_path
    )
    d4_result = ENSRHarness(
        _memory(),
        d4_adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_verified_graph_repair=True,
    ).run_episode(
        TwoStageEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )
    trace = d4_adapter.end_episode()

    assert d0_result.status == "plan_exhausted"
    assert d4_result.status == "terminal_success"
    assert [request.purpose for request in delegate.requests] == [
        "candidate_ranking"
    ]
    assert trace.prefix_complete is True
    assert trace.divergence_purpose == "candidate_ranking"
    assert trace.mismatch_count == 0


def test_action_prefix_replay_overrides_controller_alias_choice() -> None:
    adapter = FakeAdapter(
        [
            _plan_payload(),
            {
                "action": "look around",
                "expected_fact": None,
                "expected_progress": "Inspect instead of opening.",
            },
        ]
    )
    result = ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
    ).run_episode(
        OpenDoorEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
        action_replay_prefix=[
            {
                "action": "open door to kitchen",
                "score": 100,
                "observation_sha256": hashlib.sha256(
                    b"You open the door."
                ).hexdigest(),
            }
        ],
    )

    assert result.status == "terminal_success"
    assert result.trajectory[0].action == "open door to kitchen"
    assert result.trajectory[0].action_source == "replayed_control"
    assert result.mechanism_metrics["action_replay_source_count"] == 1
    assert result.mechanism_metrics["action_replay_consumed_count"] == 1
    assert result.mechanism_metrics["action_replay_override_count"] == 1
    assert result.mechanism_metrics["action_replay_mismatch_count"] == 0


def test_adaptive_continuation_can_keep_symbolic_action_control() -> None:
    procedure_memory = ENSRMemory(
        [
            {
                **_memory().fragments[0],
                "fragment_id": "fragment-inspect-first",
                "step_index": 0,
                "action_template": "look around",
            },
            _memory().fragments[0],
        ]
    )
    local_replan = {
        "diagnosis": "Looking again did not open the door.",
        "replacement_subgoal": _plan_payload()["subgoals"][0],
    }
    adapter = FakeAdapter(
        [
            _plan_payload(),
            local_replan,
        ]
    )

    result = ENSRHarness(
        procedure_memory,
        adapter,
        enable_procedure_guidance=True,
        enable_symbolic_control=False,
        enable_adaptive_slow_path=True,
        force_llm_after_local_replan=False,
    ).run_episode(
        RecoveringEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.model_calls == 2
    assert result.trajectory[-1].action_source == "procedure"
    assert result.mechanism_metrics["adaptive_slow_action_count"] == 0


def test_adaptive_slow_path_reserves_planner_calls_for_continuations() -> None:
    assert _local_replan_has_budget(
        planner_calls=3,
        max_planner_calls=4,
        adaptive_slow_path_enabled=False,
        adaptive_continuation_plan_count=0,
    )
    assert _local_replan_has_budget(
        planner_calls=1,
        max_planner_calls=4,
        adaptive_slow_path_enabled=True,
        adaptive_continuation_plan_count=0,
    )
    assert not _local_replan_has_budget(
        planner_calls=2,
        max_planner_calls=4,
        adaptive_slow_path_enabled=True,
        adaptive_continuation_plan_count=0,
    )
    assert not _local_replan_has_budget(
        planner_calls=3,
        max_planner_calls=4,
        adaptive_slow_path_enabled=True,
        adaptive_continuation_plan_count=1,
    )


def test_reserved_planner_call_recovers_after_retry_exhaustion() -> None:
    initial_plan = _plan_payload()
    initial_plan["subgoals"][0]["retry_limit"] = 1
    continuation_plan = _plan_payload()
    adapter = FakeAdapter(
        [
            initial_plan,
            {
                "action": "look around",
                "expected_fact": None,
                "expected_progress": "Inspect the blocked state.",
            },
            continuation_plan,
            _open_action_payload(),
        ]
    )

    result = ENSRHarness(
        _memory(),
        adapter,
        enable_procedure_guidance=False,
        enable_symbolic_control=False,
        enable_adaptive_slow_path=True,
    ).run_episode(
        RecoveringEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(
            max_environment_steps=5,
            max_model_calls=10,
            max_planner_calls=2,
        ),
    )

    assert result.status == "terminal_success"
    assert result.final_score == 100
    assert result.mechanism_metrics["local_replan_count"] == 0
    assert result.mechanism_metrics["adaptive_continuation_plan_count"] == 1
    assert result.mechanism_metrics["adaptive_continuation_recovery_count"] == 1


def test_semantic_target_graph_filters_distractors_and_ranks_lifespan() -> None:
    living = _semantic_focus_actions(
        "living",
        [
            "focus on orange",
            "focus on battery",
            "focus on baby baby beaver",
            "focus on egg giant tortoise",
            "pick up baby baby beaver",
            "pick up egg giant tortoise",
        ],
    )
    longest = _semantic_focus_actions(
        "longest_lifespan",
        [
            "focus on baby baby beaver",
            "focus on baby baby dragonfly",
            "focus on egg giant tortoise",
        ],
    )
    nonliving = _semantic_focus_actions(
        "nonliving",
        [
            "focus on sewer",
            "focus on steel table",
            "pick up sewer",
            "pick up steel table",
        ],
    )

    assert living == ["focus on baby baby beaver", "focus on egg giant tortoise"]
    assert longest[0] == "focus on egg giant tortoise"
    assert nonliving == ["focus on steel table"]


def test_semantic_target_graph_parses_delivery_and_exact_actions() -> None:
    task = (
        "Your task is to find a(n) plant. First, focus on the thing. "
        "Then, move it to the green box in the bathroom."
    )

    assert _semantic_task_kind(task) == "plant"
    assert _semantic_destination(task) == ("green box", "bathroom")
    assert (
        _semantic_acquisition_action(
            ["pick up adult apple tree", "pick up orange"],
            "adult apple tree",
        )
        == "pick up adult apple tree"
    )
    assert (
        _semantic_acquisition_action(
            ["pick up apple tree", "pick up apple pollen"],
            "adult apple tree",
        )
        == "pick up apple tree"
    )
    assert (
        _semantic_move_to_box_action(
            [
                "move orange to green box",
                "move adult apple tree to green box",
            ],
            "adult apple tree",
            "green box",
        )
        == "move adult apple tree to green box"
    )


def test_train_memory_builds_a_room_graph_for_routing() -> None:
    fragments = []
    for index, (source, destination) in enumerate(
        [("workshop", "hallway"), ("hallway", "greenhouse")], start=1
    ):
        fragments.append(
            {
                "fragment_id": f"route-{index}",
                "task_id": "4-1",
                "task_name": "find-living-thing",
                "variation": 1,
                "step_index": index,
                "action_template": f"go to {destination}",
                "operator": "go",
                "preconditions": [
                    {
                        "subject": "agent",
                        "relation": "located_in",
                        "object": source,
                        "polarity": True,
                    }
                ],
                "effects": [
                    {
                        "subject": "agent",
                        "relation": "located_in",
                        "object": destination,
                        "polarity": True,
                    }
                ],
                "outcome": {"reward": 0, "score": 0, "completed": False},
            }
        )
    memory = ENSRMemory(fragments)

    assert (
        memory.route_action(
            ["look around", "go to hallway"],
            current_location="workshop",
            target_location="greenhouse",
        )
        == "go to hallway"
    )
    assert (
        memory.route_action(
            ["look around", "open greenhouse door"],
            current_location="hallway",
            target_location="greenhouse",
        )
        == "open greenhouse door"
    )


def test_canonicalizer_repairs_only_unambiguous_fact_forms() -> None:
    payload, counts = _canonicalize_model_payload(
        {
            "subject": "agent",
            "relation": "in_inventory_of",
            "object": "substance called soap",
            "polarity": True,
        }
    )

    assert payload["subject"] == "substance called soap"
    assert payload["object"] == "agent"
    assert counts == {"canonicalized_inventory_orientation": 1}


def test_canonicalizer_repairs_object_locations_containment_and_alternatives() -> None:
    payload, counts = _canonicalize_model_payload(
        {
            "subgoal_id": "sg1",
            "description": "Find or expose rubber.",
            "expected_facts": [
                {
                    "subject": "rubber",
                    "relation": "located_in",
                    "object": "greenhouse",
                    "polarity": True,
                },
                {
                    "subject": "rubber",
                    "relation": "contains",
                    "object": "bee hive",
                    "polarity": True,
                },
            ],
            "completion_test": "rubber is visible OR the hive contains rubber",
            "retry_limit": 2,
        }
    )

    assert payload["expected_facts"][0]["relation"] == "visible_in"
    assert payload["expected_facts"][1]["subject"] == "bee hive"
    assert payload["expected_facts"][1]["object"] == "rubber"
    assert payload["completion_mode"] == "any"
    assert counts == {
        "canonicalized_alternative_completion": 1,
        "canonicalized_object_location_relation": 1,
        "canonicalized_contains_orientation": 1,
    }


def test_target_lock_removes_wrong_irreversible_focus_action() -> None:
    actions = ["look around", "focus on outside", "focus on soap"]
    filtered = _filter_irreversible_focus_actions(
        actions,
        task_description="Your task is to boil soap.",
        visible_state="A substance called soap is here.",
    )

    assert filtered == ["look around", "focus on soap"]


def test_target_lock_always_removes_irreversible_agent_focus() -> None:
    filtered = _filter_irreversible_focus_actions(
        ["look around", "focus on agent", "focus on unknown substance"],
        task_description="Determine whether the unknown substance conducts electricity.",
        visible_state="The agent and an unknown substance are here.",
    )

    assert filtered == ["look around", "focus on unknown substance"]


def test_focus_protocol_extracts_order_and_groups_conditional_answers() -> None:
    task = (
        "First, focus on the thermometer. Next, focus on the mercury. "
        "If the melting point is above -10 C, focus on the orange box. "
        "If it is below -10 C, focus on the yellow box."
    )

    assert _focus_target_stages(task) == [
        ["thermometer"],
        ["mercury"],
        ["orange box", "yellow box"],
    ]


def test_focus_protocol_binds_generic_substance_to_explicit_task_target() -> None:
    task = (
        "Your task is to boil soap. First, focus on the substance. "
        "Then, cause it to change state."
    )

    assert _focus_target_stages(task, explicit_task_target="soap") == [["soap"]]


def test_focus_protocol_exposes_only_current_visible_stage() -> None:
    actions = [
        "look around",
        "focus on door to kitchen",
        "focus on thermometer",
        "focus on mercury",
    ]
    filtered = _filter_irreversible_focus_actions(
        actions,
        task_description="First, focus on the thermometer. Next, focus on mercury.",
        visible_state="A thermometer and mercury are here.",
        allowed_focus_targets=["thermometer"],
    )

    assert filtered == ["look around", "focus on thermometer"]


def test_focus_protocol_accepts_box_surface_alias() -> None:
    filtered = _filter_irreversible_focus_actions(
        ["focus on orange", "focus on yellow", "focus on door"],
        task_description="Focus on the orange box or yellow box.",
        visible_state="An orange and a yellow are here.",
        allowed_focus_targets=["orange box", "yellow box"],
    )

    assert filtered == ["focus on orange", "focus on yellow"]


def test_target_lock_covers_named_chemistry_products() -> None:
    actions = ["open door to kitchen", "focus on door to kitchen", "focus on smores"]
    filtered = _filter_irreversible_focus_actions(
        actions,
        task_description=(
            "Your task is to use chemistry to create the substance 'smores'. "
            "When you are done, focus on the smores."
        ),
        visible_state="The door to kitchen is here; smores is not yet present.",
    )

    assert filtered == ["open door to kitchen", "focus on smores"]


def test_temporal_subgoal_evidence_accumulates_across_transitions() -> None:
    expected = [
        FactPattern(subject="picture", relation="visible_in", object="hallway"),
        FactPattern(subject="picture", relation="in_inventory_of", object="agent"),
    ]
    achieved: set[tuple[str, str, str, bool]] = set()
    visible = parse_observation_facts(CLOSED + "\tA picture\n", step=0)
    carried = parse_observation_facts(
        "In your inventory, you see:\n\ta picture\n", step=1
    )

    assert not _facts_satisfied(expected, visible, "all", achieved)
    assert _facts_satisfied(expected, carried, "all", achieved)


def test_memory_exposes_one_complete_train_procedure() -> None:
    procedure = _memory().procedure_context("1-1")

    assert procedure == {
        "task_id": "1-1",
        "variation": 0,
        "actions": ["open door to kitchen"],
    }


def test_procedure_action_binds_target_and_navigation_surface_form() -> None:
    navigation = _next_procedure_action(
        ["look around", "go to door to kitchen"],
        ["go to kitchen"],
        cursor=0,
        task_target="soap",
    )
    target_action = _next_procedure_action(
        ["move soap in inventory to metal pot", "move axe to inventory"],
        ["move <FOCUS_OBJECT> to metal pot"],
        cursor=0,
        task_target="soap",
    )

    assert navigation == ("go to door to kitchen", 1)
    assert target_action == ("move soap in inventory to metal pot", 1)


def test_procedure_action_skips_unbound_entity_placeholder() -> None:
    selected = _next_procedure_action(
        ["focus on agent", "focus on baby dragonfly", "look around"],
        ["focus on <FOCUS_OBJECT>"],
        cursor=0,
        task_target=None,
    )

    assert selected is None


def test_procedure_action_binds_focus_tool_and_task_target_separately() -> None:
    selected = _next_procedure_action(
        ["use thermometer in inventory on mercury"],
        ["use <FOCUS_OBJECT> on <TASK_TARGET>"],
        cursor=0,
        task_target="mercury",
        focus_object="thermometer",
    )

    assert selected == ("use thermometer in inventory on mercury", 1)


def test_procedure_action_rejects_wrong_bound_action_subject() -> None:
    selected = _next_procedure_action(
        [
            "move thermometer to metal pot",
            "move solid unknown substance k to metal pot",
        ],
        ["move <TASK_TARGET> to metal pot"],
        cursor=0,
        task_target="solid unknown substance k",
        focus_object="thermometer",
    )

    assert selected == ("move solid unknown substance k to metal pot", 1)


def test_procedure_action_requires_same_manipulation_subject() -> None:
    selected = _next_procedure_action(
        [
            "move thermometer to metal pot",
            "move metal pot containing mercury to stove",
        ],
        ["move metal pot containing <TASK_TARGET> to stove"],
        cursor=0,
        task_target="mercury",
        focus_object="thermometer",
    )

    assert selected == ("move metal pot containing mercury to stove", 1)


def test_procedure_action_maps_examine_to_legal_look_at_alias() -> None:
    selected = _next_procedure_action(
        ["look at solid unknown substance k"],
        ["examine <TASK_TARGET>"],
        cursor=0,
        task_target="solid unknown substance k",
    )

    assert selected == ("look at solid unknown substance k", 1)


def test_measurement_conditioned_focus_uses_observed_transition_temperature() -> None:
    task = (
        "If the melting point is above 50.0 degrees celsius, focus on the green box. "
        "If it is below 50.0 degrees celsius, focus on the blue box."
    )

    assert (
        _measurement_conditioned_focus_action(
            task,
            ["focus on green", "focus on blue"],
            105.0,
        )
        == "focus on green"
    )


def test_measurement_poll_alternates_grounded_examine_and_thermometer_actions() -> None:
    actions = [
        "look at solid unknown substance k",
        "use thermometer in inventory on solid unknown substance k",
        "look at stove",
    ]

    assert (
        _measurement_poll_action(
            actions,
            "solid unknown substance k",
            prefer_examine=True,
        )
        == "look at solid unknown substance k"
    )
    assert (
        _measurement_poll_action(
            actions,
            "solid unknown substance k",
            prefer_examine=False,
        )
        == "use thermometer in inventory on solid unknown substance k"
    )


def test_solid_state_above_threshold_is_sufficient_measurement_evidence() -> None:
    task = (
        "If the melting point is above 50.0 degrees celsius, focus on the green box. "
        "If it is below 50.0 degrees celsius, focus on the blue box."
    )

    assert (
        _measurement_decision_temperature(
            task,
            "solid unknown substance k",
            "a metal pot containing solid unknown substance K",
            61.0,
        )
        == 61.0
    )


def test_procedure_target_extracts_measurement_entity() -> None:
    task = (
        "Your task is to measure the melting point of solid unknown substance K, "
        "which is located around the kitchen."
    )

    assert _procedure_task_target(task) == "solid unknown substance K"


def test_acquisition_subgoal_blocks_unrelated_procedure_operations() -> None:
    payload = _plan_payload()["subgoals"][0]
    payload["expected_facts"] = [
        {
            "subject": "rubber",
            "relation": "in_inventory_of",
            "object": "agent",
            "polarity": True,
        }
    ]
    subgoal = PlannerSubgoal.model_validate(payload).as_subgoal()
    filtered = _procedure_action_domain(
        [
            "open cupboard",
            "go to kitchen",
            "pick up thermometer",
            "activate stove",
            "move rubber to inventory",
        ],
        subgoal,
    )

    assert filtered == [
        "open cupboard",
        "go to kitchen",
        "move rubber to inventory",
    ]


def test_symbolic_goal_grounding_prefers_target_inventory_action() -> None:
    subgoal = _plan_payload()["subgoals"][0]
    subgoal["expected_facts"] = [
        {
            "subject": "substance called soap",
            "relation": "in_inventory_of",
            "object": "agent",
            "polarity": True,
        }
    ]
    action = _goal_directed_action(
        ["move axe to inventory", "move soap in kitchen to inventory"],
        PlannerSubgoal.model_validate(subgoal).as_subgoal(),
        [],
        set(),
    )

    assert action == "move soap in kitchen to inventory"


def test_temporal_entity_location_routes_back_to_known_goal_object() -> None:
    entity_locations: dict[str, str] = {}
    _update_entity_locations(
        parse_observation_facts(
            "This room is called the greenhouse. In it, you see:\n"
            "\ta jug (containing nothing)\n",
            step=0,
        ),
        entity_locations,
    )
    subgoal_payload = _plan_payload()["subgoals"][0]
    subgoal_payload["expected_facts"] = [
        {
            "subject": "jug",
            "relation": "contains",
            "object": "rubber",
            "polarity": True,
        }
    ]

    action = _goal_directed_action(
        ["go to door to greenhouse", "go to door to kitchen"],
        PlannerSubgoal.model_validate(subgoal_payload).as_subgoal(),
        [],
        set(),
        entity_locations,
    )

    assert entity_locations["jug"] == "greenhouse"
    assert action == "go to door to greenhouse"


def test_agent_navigation_goal_ignores_stale_entity_location_hint() -> None:
    subgoal_payload = _plan_payload()["subgoals"][0]
    subgoal_payload["expected_facts"] = [
        {
            "subject": "agent",
            "relation": "located_in",
            "object": "foundry",
            "polarity": True,
        }
    ]

    action = _goal_directed_action(
        ["go to outside", "go to foundry"],
        PlannerSubgoal.model_validate(subgoal_payload).as_subgoal(),
        [],
        set(),
        {"agent": "outside"},
    )

    assert action == "go to foundry"


def test_state_change_workflow_places_target_then_activates_heat() -> None:
    carried = parse_observation_facts(
        "In your inventory, you see:\n"
        "\ta metal pot (containing nothing)\n"
        "\ta substance called soap\n",
        step=0,
    )
    place = _state_change_workflow_action(
        ["move soap in inventory to metal pot", "move orange to metal pot"],
        task_description="Your task is to boil soap.",
        task_target="soap",
        observed_facts=carried,
    )
    heating = parse_observation_facts(
        "This room is called the kitchen. In it, you see:\n"
        "\ta stove, which is turned off. On the stove is: "
        "a metal pot (containing solid soap).\n",
        step=1,
    )
    activate = _state_change_workflow_action(
        ["pick up metal pot", "move metal pot to stove", "activate stove", "wait"],
        task_description="Your task is to boil soap.",
        task_target="soap",
        observed_facts=heating,
    )

    assert place == "move soap in inventory to metal pot"
    assert activate == "activate stove"

    jug = parse_observation_facts(
        "This room is called the greenhouse. In it, you see:\n"
        "\ta jug (containing rubber)\n",
        step=2,
    )
    assert (
        _state_change_workflow_action(
            ["move jug to stove"],
            task_description="Your task is to boil rubber.",
            task_target="rubber",
            observed_facts=jug,
        )
        == "move jug to stove"
    )


def test_state_change_workflow_carries_container_and_routes_to_known_device() -> None:
    jug = parse_observation_facts(
        "This room is called the greenhouse. In it, you see:\n"
        "\ta jug (containing rubber)\n",
        step=0,
    )
    pickup = _state_change_workflow_action(
        ["move jug to inventory", "go to door to kitchen"],
        task_description="Your task is to boil rubber.",
        task_target="rubber",
        observed_facts=jug,
        entity_locations={"jug": "greenhouse", "stove": "kitchen"},
    )
    carried = parse_observation_facts(
        "In your inventory, you see:\n\ta jug (containing rubber)\n",
        step=1,
    )
    navigate = _state_change_workflow_action(
        ["go to door to kitchen", "go to door to workshop"],
        task_description="Your task is to boil rubber.",
        task_target="rubber",
        observed_facts=carried,
        entity_locations={"jug": "agent", "stove": "kitchen"},
    )

    assert pickup == "move jug to inventory"
    assert navigate == "go to door to kitchen"


def test_state_change_workflow_requires_container_as_pickup_action_subject() -> None:
    carried_target = parse_observation_facts(
        "In your inventory, you see:\n\ta substance called soap\n",
        step=0,
    )

    action = _state_change_workflow_action(
        [
            "move orange in inventory to metal pot",
            "move metal pot to inventory",
            "pick up metal pot",
        ],
        task_description="Your task is to change the state of matter of soap.",
        task_target="soap",
        observed_facts=carried_target,
    )

    assert action == "pick up metal pot"


def test_target_acquisition_opens_container_then_selects_target_subject() -> None:
    closed = parse_observation_facts(
        "This room is called the kitchen. In it, you see:\n"
        "\ta cupboard. The cupboard door is closed. In the cupboard is: "
        "a substance called soap.\n",
        step=0,
    )
    open_action = _task_target_acquisition_action(
        ["open cupboard", "move orange to inventory"],
        task_target="soap",
        observed_facts=closed,
    )
    visible = parse_observation_facts(
        "This room is called the kitchen. In it, you see:\n"
        "\ta substance called soap\n",
        step=1,
    )
    pickup_action = _task_target_acquisition_action(
        ["move orange to inventory", "move soap in kitchen to inventory"],
        task_target="soap",
        observed_facts=visible,
    )

    assert open_action == "open cupboard"
    assert pickup_action == "move soap in kitchen to inventory"


def test_target_acquisition_preserves_target_inside_portable_vessel() -> None:
    contained = parse_observation_facts(
        "This room is called the kitchen. In it, you see:\n"
        "\ta stove, which is turned off. On the stove is: "
        "a metal pot (containing solid soap).\n",
        step=0,
    )

    action = _task_target_acquisition_action(
        ["pick up soap in metal pot", "activate stove"],
        task_target="soap",
        observed_facts=contained,
    )

    assert action is None


def test_state_change_workflow_opens_tool_container_before_pickup() -> None:
    facts = parse_observation_facts(
        "This room is called the kitchen. In it, you see:\n"
        "\ta cupboard. The cupboard door is closed. In the cupboard is: "
        "a metal pot (containing nothing).\n"
        "In your inventory, you see:\n\ta substance called soap\n",
        step=0,
    )

    action = _state_change_workflow_action(
        ["open cupboard", "pick up metal pot"],
        task_description="Your task is to change the state of matter of soap.",
        task_target="soap",
        observed_facts=facts,
    )

    assert action == "open cupboard"


def test_state_change_workflow_inspects_closed_container_for_unknown_tool() -> None:
    facts = parse_observation_facts(
        "This room is called the kitchen. In it, you see:\n"
        "\ta cupboard. The cupboard door is closed.\n"
        "In your inventory, you see:\n\ta substance called soap\n",
        step=0,
    )

    action = _state_change_workflow_action(
        ["open cupboard", "pick up metal pot"],
        task_description="Your task is to change the state of matter of soap.",
        task_target="soap",
        observed_facts=facts,
        entity_locations={},
    )

    assert action == "open cupboard"


def test_state_change_workflow_abandons_failed_device() -> None:
    heating = parse_observation_facts(
        "This room is called the kitchen. In it, you see:\n"
        "\ta stove, which is turned off. On the stove is: "
        "a metal pot (containing solid rubber).\n",
        step=0,
    )
    failed = _failed_devices_from_observation(
        "The stove appears broken, and can't be activated or deactivated."
    )

    action = _state_change_workflow_action(
        ["activate stove", "move metal pot to inventory"],
        task_description="Your task is to boil rubber.",
        task_target="rubber",
        observed_facts=heating,
        entity_locations={"blast furnace": "foundry", "stove": "kitchen"},
        unavailable_devices=failed,
    )

    assert failed == {"stove"}
    assert action == "move metal pot to inventory"


def test_verified_entity_location_repairs_hallucinated_plan_location() -> None:
    payload = _plan_payload()["subgoals"][0]
    payload["expected_facts"] = [
        {
            "subject": "fire pit",
            "relation": "visible_in",
            "object": "foundry",
            "polarity": True,
        }
    ]
    subgoal = PlannerSubgoal.model_validate(payload).as_subgoal()

    count = _repair_expected_locations([subgoal], {"fire pit": "outside"})

    assert count == 1
    assert subgoal.expected_facts[0].key == (
        "fire pit",
        "visible_in",
        "outside",
        True,
    )


def test_systematic_exploration_prefers_unvisited_route_then_container() -> None:
    actions = [
        "look around",
        "open cupboard",
        "open door to kitchen",
        "go to door to hallway",
    ]

    assert (
        _systematic_exploration_action(actions, visited_locations={"hallway"})
        == "open door to kitchen"
    )
    assert (
        _systematic_exploration_action(
            ["look around", "open cupboard"], visited_locations={"kitchen"}
        )
        == "open cupboard"
    )
    assert (
        _systematic_exploration_action(
            ["go to door", "go to foundry", "go to outside"],
            visited_locations={"foundry", "outside"},
            current_location="foundry",
        )
        == "go to outside"
    )
    assert (
        _systematic_exploration_action(
            ["go to foundry", "go to kitchen", "open cupboard"],
            visited_locations={"foundry", "kitchen"},
            current_location="outside",
            destination_visit_counts={"foundry": 4, "kitchen": 1},
            explored_open_actions={("outside", "open cupboard")},
        )
        == "go to kitchen"
    )


def test_failed_plan_attempts_use_grounded_deterministic_fallback() -> None:
    result = ENSRHarness(_memory(), FailingAdapter([])).run_episode(
        OpenDoorEnvironment(),
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=17,
        seed=13,
        budget=ENSRBudget(max_environment_steps=5, max_model_calls=10),
    )

    assert result.status == "terminal_success"
    assert result.model_calls == 2
    assert result.planner_calls == 2
    assert result.mechanism_metrics["planner_parse_failure_count"] == 2
    assert result.mechanism_metrics["deterministic_fallback_plan_count"] == 1
    assert result.response_normalizations["deterministic_fallback_plan"] == 1


def test_conductivity_probe_builds_role_consistent_circuit() -> None:
    actions = [
        "connect anode in battery to blue wire terminal 1",
        "connect battery cathode to green wire terminal 1",
        "connect blue wire terminal 2 to cathode in red light bulb",
        "connect red wire terminal 2 to anode in red light bulb",
        "connect plastic fork to green wire terminal 2",
        "connect red wire terminal 1 to plastic fork",
        "wait1",
    ]
    components = _conductivity_probe_components(actions)

    assert components == ("blue", "green", "red", "red light bulb")
    assert _conductivity_probe_action(
        actions,
        "plastic fork",
        components,
        connection_index=0,
        wait_count=0,
    ) == (actions[0], "connection")
    assert _conductivity_probe_action(
        actions,
        "plastic fork",
        components,
        connection_index=4,
        wait_count=0,
    ) == (actions[4], "connection")
    assert _conductivity_probe_action(
        actions,
        "plastic fork",
        components,
        connection_index=6,
        wait_count=0,
    ) == ("wait1", "wait")

    unknown_actions = [
        "connect anode in battery to blue wire terminal 1",
        "connect battery cathode to red wire terminal 1",
        "connect terminal 2 in yellow wire to anode in electric motor",
        "connect blue wire terminal 2 to cathode in electric motor",
    ]
    assert _conductivity_probe_components(unknown_actions) == (
        "blue",
        "red",
        "yellow",
        "electric motor",
    )


def test_conductivity_answer_uses_observed_detector_and_current_boxes() -> None:
    task = (
        "Your task is to determine if plastic fork is electrically conductive. "
        "If it is electrically conductive, place it in the orange box. "
        "If it is electrically nonconductive, place it in the yellow box."
    )
    actions = [
        "move plastic fork to orange box",
        "move plastic fork to yellow box",
    ]

    assert _conditional_conductivity_boxes(task) == ("orange box", "yellow box")
    assert not _conductivity_detector_result(
        "a red light bulb, which is off", "red light bulb", probe_complete=True
    )
    assert _conductivity_detector_result(
        "a red light bulb, which is on, a green light bulb, which is off",
        "red light bulb",
        probe_complete=True,
    )
    assert (
        _conductivity_conditioned_move_action(
            actions, "plastic fork", task, conductive=False
        )
        == "move plastic fork to yellow box"
    )


def test_unknown_substance_letter_uses_environment_action_alias() -> None:
    task = (
        "Your task is to determine if unknown substance R is electrically conductive. "
        "If it is electrically conductive, place it in the blue box. "
        "If it is electrically nonconductive, place it in the orange box."
    )

    assert _filter_irreversible_focus_actions(
        ["focus on agent", "focus on unknown substance"],
        task_description=task,
        visible_state="unknown substance R",
        allowed_focus_targets=["unknown substance R"],
    ) == ["focus on unknown substance"]
    assert (
        _conductivity_conditioned_move_action(
            [
                "move unknown substance to blue box",
                "move unknown substance to orange box",
            ],
            "unknown substance R",
            task,
            conductive=True,
        )
        == "move unknown substance to blue box"
    )
