from copy import deepcopy

import pytest

from lanternquest.benchmark import LanternQuestBenchmarkCase
from lanternquest.e21 import aggregate_scores, score_decision, score_failed_run
from lanternquest.planning import PlanCandidate, PlanningDecision
from tests.test_benchmark import make_benchmark_case


def make_decision(
    *,
    action_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    proposed_changes: list[str] | None = None,
) -> PlanningDecision:
    return PlanningDecision(
        method="iper_rag",
        decision_type="propose",
        case_id="lq_case_1",
        state_version="state_v1",
        intent_version="intent_v1",
        plans=[
            PlanCandidate(
                action_ids=action_ids or ["action_1"],
                retained_intent_ids=["preserve_style"],
                proposed_changes=proposed_changes or [],
                required_consent=[],
                evidence_ids=evidence_ids or ["evidence_1"],
                unresolved_conditions=[],
                intent_deviation_cost=0,
                rework_cost=0,
                retrieval_calls=2,
                action_cost=1.0,
            )
        ],
        unresolved_conditions=[],
        retrieval_calls=2,
    )


def test_score_accepts_grounded_gold_plan() -> None:
    case = LanternQuestBenchmarkCase.model_validate(make_benchmark_case())

    score = score_decision(case, "ensr", make_decision(), model_calls=1)

    assert score.grounded_task_success is True
    assert score.goal_reached is True
    assert score.gold_sequence_exact is True
    assert score.evidence_coverage == 1.0
    assert score.unsupported_action_count == 0


def test_score_rejects_unsupported_action_even_when_goal_is_reached() -> None:
    case = LanternQuestBenchmarkCase.model_validate(make_benchmark_case())
    decision = make_decision(evidence_ids=["unknown_evidence"])

    score = score_decision(case, "b0_full_context", decision)

    assert score.goal_reached is True
    assert score.grounded_task_success is False
    assert score.evidence_coverage == 0.0
    assert score.unsupported_action_count == 1
    assert score.hallucinated_evidence_count == 1


def test_score_flags_locked_intent_change_and_rework() -> None:
    payload = deepcopy(make_benchmark_case())
    action = payload["action_space"][0]
    action["changes_intent_ids"] = ["preserve_style"]
    action["invalidates_work_ids"] = ["painted_background"]
    case = LanternQuestBenchmarkCase.model_validate(payload)
    decision = make_decision(proposed_changes=["preserve_style"])

    score = score_decision(case, "iper_rag", decision)

    assert score.locked_intent_preserved is False
    assert score.avoidable_rework_count == 1
    assert score.grounded_task_success is False


def test_score_rejects_cross_case_decision() -> None:
    case = LanternQuestBenchmarkCase.model_validate(make_benchmark_case())
    decision = make_decision().model_copy(update={"case_id": "lq_other"})

    with pytest.raises(ValueError, match="case identifier"):
        score_decision(case, "ensr", decision)


def test_aggregate_scores_combines_method_totals() -> None:
    case = LanternQuestBenchmarkCase.model_validate(make_benchmark_case())
    first = score_decision(case, "ensr", make_decision(), model_calls=1)
    second = first.model_copy(
        update={
            "grounded_task_success": False,
            "unsupported_action_count": 1,
            "wall_seconds": 2.5,
        }
    )

    aggregate = aggregate_scores([first, second])[0]

    assert aggregate.episodes == 2
    assert aggregate.grounded_task_success_rate == 0.5
    assert aggregate.unsupported_action_count == 1
    assert aggregate.model_calls == 2
    assert aggregate.wall_seconds == 2.5


def test_failed_run_stays_in_aggregate_denominator() -> None:
    case = LanternQuestBenchmarkCase.model_validate(make_benchmark_case())
    completed = score_decision(case, "ensr", make_decision())
    failed = score_failed_run(
        case,
        "ensr",
        "provider_error",
        failure_detail="HTTP 503",
        model_calls=1,
    )

    aggregate = aggregate_scores([completed, failed])[0]

    assert aggregate.episodes == 2
    assert aggregate.grounded_task_success_rate == 0.5
    assert aggregate.external_failure_count == 1
    assert aggregate.run_status_counts == {"completed": 1, "provider_error": 1}


def test_correct_clarification_can_be_a_grounded_success() -> None:
    payload = deepcopy(make_benchmark_case())
    payload["initial_state"][0]["value"] = "unknown"
    payload["expected_decision_type"] = "clarify"
    payload["expected_unresolved_conditions"] = ["state:ready"]
    payload["gold_plan"] = []
    case = LanternQuestBenchmarkCase.model_validate(payload)
    decision = PlanningDecision(
        method="iper_rag",
        decision_type="clarify",
        case_id=case.case_id,
        state_version=case.state_version,
        intent_version=case.intent_version,
        plans=[],
        unresolved_conditions=["state:ready"],
        retrieval_calls=1,
    )

    score = score_decision(case, "ensr", decision)

    assert score.goal_reached is False
    assert score.decision_type_exact is True
    assert score.unresolved_conditions_exact is True
    assert score.grounded_task_success is True
