from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lanternquest.benchmark import LanternQuestBenchmarkCase
from lanternquest.planning import PlanningDecision

E21Method = Literal[
    "b0_full_context",
    "b1_rag",
    "b2_kg_rag",
    "b4_sequential",
    "iper_rag",
    "ensr",
]
E21RunStatus = Literal[
    "completed",
    "provider_error",
    "parser_error",
    "timeout",
    "budget_exhausted",
    "internal_error",
]


class E21CaseScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    split: str
    method: E21Method
    run_status: E21RunStatus
    failure_detail: str | None = None
    decision_type: str
    decision_type_exact: bool
    unresolved_conditions_exact: bool
    grounded_task_success: bool
    goal_reached: bool
    failure_condition_triggered: bool
    gold_sequence_exact: bool
    evidence_coverage: float = Field(ge=0, le=1)
    locked_intent_preserved: bool
    unsupported_action_count: int = Field(ge=0)
    invalid_action_count: int = Field(ge=0)
    hallucinated_action_count: int = Field(ge=0)
    hallucinated_evidence_count: int = Field(ge=0)
    avoidable_rework_count: int = Field(ge=0)
    action_count: int = Field(ge=0)
    retrieval_calls: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    wall_seconds: float = Field(ge=0)
    returned_model: str | None = None
    system_fingerprint: str | None = None


class E21Aggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: E21Method
    episodes: int
    grounded_task_success_rate: float
    decision_type_accuracy: float
    mean_evidence_coverage: float
    locked_intent_preservation_rate: float
    unsupported_action_count: int
    invalid_action_count: int
    hallucinated_action_count: int
    hallucinated_evidence_count: int
    avoidable_rework_count: int
    model_calls: int
    retrieval_calls: int
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    external_failure_count: int
    run_status_counts: dict[str, int]
    decision_type_counts: dict[str, int]


def _conditions_reached(conditions: list, state: dict[str, str]) -> bool:
    return all(state.get(item.predicate, "unknown") == item.value for item in conditions)


def score_decision(
    case: LanternQuestBenchmarkCase,
    method: E21Method,
    decision: PlanningDecision,
    *,
    model_calls: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    wall_seconds: float = 0,
) -> E21CaseScore:
    if decision.case_id != case.case_id:
        raise ValueError("decision case identifier does not match benchmark case")
    if decision.state_version != case.state_version:
        raise ValueError("decision state version does not match benchmark case")
    if decision.intent_version != case.intent_version:
        raise ValueError("decision intent version does not match benchmark case")

    actions = {item.action_id: item for item in case.action_space}
    evidence_ids = {item.evidence_id for item in case.evidence_catalog}
    obligations = {
        item.obligation_id: set(item.satisfied_by_evidence_ids)
        for item in case.evidence_obligations
    }
    locked_intents = {
        item.intent_id for item in case.locked_intents if item.locked
    }
    state = {item.predicate: item.value for item in case.initial_state}
    completed_work = set(case.completed_work_ids)

    if decision.plans:
        plan = decision.plans[0]
        selected_actions = list(plan.action_ids)
        selected_evidence = set(plan.evidence_ids)
        proposed_changes = set(plan.proposed_changes)
    else:
        selected_actions = []
        selected_evidence = set()
        proposed_changes = set()

    hallucinated_actions = sum(action_id not in actions for action_id in selected_actions)
    hallucinated_evidence = len(selected_evidence - evidence_ids)
    invalid_actions = 0
    unsupported_actions = 0
    avoidable_rework = 0
    required_obligations: set[str] = set()
    intent_violations = proposed_changes & locked_intents

    for action_id in selected_actions:
        action = actions.get(action_id)
        if action is None:
            continue
        if not _conditions_reached(action.preconditions, state):
            invalid_actions += 1
            continue
        action_obligations = set(action.evidence_obligation_ids)
        required_obligations.update(action_obligations)
        covered = all(
            bool(obligations[obligation_id] & selected_evidence)
            for obligation_id in action_obligations
        )
        if not covered:
            unsupported_actions += 1
        changed_locked = set(action.changes_intent_ids) & locked_intents
        intent_violations.update(changed_locked)
        avoidable_rework += len(set(action.invalidates_work_ids) & completed_work)
        for effect in action.effects:
            state[effect.predicate] = effect.value

    covered_obligations = sum(
        bool(obligations[obligation_id] & selected_evidence)
        for obligation_id in required_obligations
    )
    evidence_coverage = (
        covered_obligations / len(required_obligations)
        if required_obligations
        else 1.0
    )
    goal_reached = _conditions_reached(case.terminal_conditions.success, state)
    failure_triggered = bool(case.terminal_conditions.failure) and _conditions_reached(
        case.terminal_conditions.failure, state
    )
    intent_preserved = not intent_violations
    decision_type_exact = decision.decision_type == case.expected_decision_type
    unresolved_exact = set(decision.unresolved_conditions) == set(
        case.expected_unresolved_conditions
    )
    clean_output = (
        intent_preserved
        and invalid_actions == 0
        and unsupported_actions == 0
        and hallucinated_actions == 0
        and hallucinated_evidence == 0
    )
    if case.expected_decision_type == "propose":
        grounded_success = (
            decision_type_exact
            and goal_reached
            and not failure_triggered
            and clean_output
        )
    else:
        grounded_success = (
            decision_type_exact
            and unresolved_exact
            and not selected_actions
            and clean_output
        )
    return E21CaseScore(
        case_id=case.case_id,
        split=case.split,
        method=method,
        run_status="completed",
        decision_type=decision.decision_type,
        decision_type_exact=decision_type_exact,
        unresolved_conditions_exact=unresolved_exact,
        grounded_task_success=grounded_success,
        goal_reached=goal_reached,
        failure_condition_triggered=failure_triggered,
        gold_sequence_exact=selected_actions == case.gold_plan,
        evidence_coverage=evidence_coverage,
        locked_intent_preserved=intent_preserved,
        unsupported_action_count=unsupported_actions,
        invalid_action_count=invalid_actions,
        hallucinated_action_count=hallucinated_actions,
        hallucinated_evidence_count=hallucinated_evidence,
        avoidable_rework_count=avoidable_rework,
        action_count=len(selected_actions),
        retrieval_calls=decision.retrieval_calls,
        model_calls=model_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_seconds=wall_seconds,
    )


def score_failed_run(
    case: LanternQuestBenchmarkCase,
    method: E21Method,
    status: Literal[
        "provider_error",
        "parser_error",
        "timeout",
        "budget_exhausted",
        "internal_error",
    ],
    *,
    failure_detail: str,
    model_calls: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    wall_seconds: float = 0,
) -> E21CaseScore:
    return E21CaseScore(
        case_id=case.case_id,
        split=case.split,
        method=method,
        run_status=status,
        failure_detail=failure_detail,
        decision_type="run_failure",
        decision_type_exact=False,
        unresolved_conditions_exact=False,
        grounded_task_success=False,
        goal_reached=False,
        failure_condition_triggered=False,
        gold_sequence_exact=False,
        evidence_coverage=0.0,
        locked_intent_preserved=False,
        unsupported_action_count=0,
        invalid_action_count=0,
        hallucinated_action_count=0,
        hallucinated_evidence_count=0,
        avoidable_rework_count=0,
        action_count=0,
        retrieval_calls=0,
        model_calls=model_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_seconds=wall_seconds,
    )


def aggregate_scores(scores: list[E21CaseScore]) -> list[E21Aggregate]:
    output = []
    for method in sorted({item.method for item in scores}):
        subset = [item for item in scores if item.method == method]
        count = len(subset)
        output.append(
            E21Aggregate(
                method=method,
                episodes=count,
                grounded_task_success_rate=(
                    sum(item.grounded_task_success for item in subset) / count
                ),
                decision_type_accuracy=(
                    sum(item.decision_type_exact for item in subset) / count
                ),
                mean_evidence_coverage=(
                    sum(item.evidence_coverage for item in subset) / count
                ),
                locked_intent_preservation_rate=(
                    sum(item.locked_intent_preserved for item in subset) / count
                ),
                unsupported_action_count=sum(
                    item.unsupported_action_count for item in subset
                ),
                invalid_action_count=sum(item.invalid_action_count for item in subset),
                hallucinated_action_count=sum(
                    item.hallucinated_action_count for item in subset
                ),
                hallucinated_evidence_count=sum(
                    item.hallucinated_evidence_count for item in subset
                ),
                avoidable_rework_count=sum(
                    item.avoidable_rework_count for item in subset
                ),
                model_calls=sum(item.model_calls for item in subset),
                retrieval_calls=sum(item.retrieval_calls for item in subset),
                input_tokens=sum(item.input_tokens for item in subset),
                output_tokens=sum(item.output_tokens for item in subset),
                wall_seconds=sum(item.wall_seconds for item in subset),
                external_failure_count=sum(
                    item.run_status
                    in {"provider_error", "parser_error", "timeout"}
                    for item in subset
                ),
                run_status_counts=dict(
                    sorted(Counter(item.run_status for item in subset).items())
                ),
                decision_type_counts=dict(
                    sorted(Counter(item.decision_type for item in subset).items())
                ),
            )
        )
    return output
