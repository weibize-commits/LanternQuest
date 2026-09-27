import heapq
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import count
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

TruthValue = Literal["true", "false", "unknown"]
DecisionType = Literal["propose", "clarify", "conflict", "evidence_insufficient"]
PlanningMethod = Literal[
    "b0_full_context",
    "b1_rag",
    "b2_kg_rag",
    "b4_sequential",
    "iper_rag",
    "ensr",
]


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicate: str
    value: Literal["true", "false"]


class StateFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicate: str
    value: TruthValue
    origin: str


class IntentConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str
    statement: str
    priority: int = Field(ge=1, le=3)
    locked: bool


class EvidenceObligation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    query: str


class ActionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    preconditions: list[Condition] = Field(default_factory=list)
    effects: list[Condition] = Field(default_factory=list)
    evidence_obligations: list[EvidenceObligation] = Field(default_factory=list)
    changes_intent_ids: list[str] = Field(default_factory=list)
    invalidates_work_ids: list[str] = Field(default_factory=list)
    action_cost: float = Field(default=1.0, ge=0)


class PlanningCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    goal_statement: str
    user_question: str
    state_version: str
    intent_version: str
    state: list[StateFact]
    goal: list[Condition]
    intents: list[IntentConstraint] = Field(default_factory=list)
    completed_work_ids: list[str] = Field(default_factory=list)
    actions: list[ActionDefinition]

    @model_validator(mode="after")
    def validate_unique_identifiers(self) -> "PlanningCase":
        for label, identifiers in (
            ("state predicates", [item.predicate for item in self.state]),
            ("intent identifiers", [item.intent_id for item in self.intents]),
            ("action identifiers", [item.action_id for item in self.actions]),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"Planning case has duplicate {label}")
        known_intents = {item.intent_id for item in self.intents}
        unknown_intents = {
            intent_id
            for action in self.actions
            for intent_id in action.changes_intent_ids
            if intent_id not in known_intents
        }
        if unknown_intents:
            raise ValueError(
                "Actions reference unknown intents: " + ", ".join(sorted(unknown_intents))
            )
        return self


class SearchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_depth: int = Field(default=6, ge=1)
    beam_width: int = Field(default=16, ge=1)
    max_retrieval_calls: int = Field(default=6, ge=1)
    initial_evidence_limit: int = Field(default=5, ge=1)
    evidence_per_obligation: int = Field(default=3, ge=1)
    max_plans: int = Field(default=3, ge=1, le=3)


@dataclass(frozen=True)
class EvidenceHit:
    evidence_id: str
    supported_claim_ids: tuple[str, ...]
    source_id: str = ""
    content: str = ""


class EvidenceRetriever(Protocol):
    def retrieve(self, query: str, limit: int) -> list[EvidenceHit]: ...


class PlanCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_ids: list[str]
    retained_intent_ids: list[str]
    proposed_changes: list[str]
    required_consent: list[str]
    evidence_ids: list[str]
    unresolved_conditions: list[str]
    intent_deviation_cost: int = Field(ge=0)
    rework_cost: int = Field(ge=0)
    retrieval_calls: int = Field(ge=0)
    action_cost: float = Field(ge=0)


class PlanningDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: PlanningMethod
    decision_type: DecisionType
    case_id: str
    state_version: str
    intent_version: str
    plans: list[PlanCandidate]
    unresolved_conditions: list[str]
    retrieval_calls: int = Field(ge=0)


@dataclass(frozen=True)
class _Label:
    state: tuple[tuple[str, TruthValue], ...]
    action_ids: tuple[str, ...]
    obligations: tuple[tuple[str, str], ...]
    evidence: tuple[EvidenceHit, ...]
    changed_intents: frozenset[str]
    invalidated_work: frozenset[str]
    retrieval_calls: int
    action_cost: float


def _state_dict(label: _Label) -> dict[str, TruthValue]:
    return dict(label.state)


def _condition_status(
    conditions: Iterable[Condition], state: dict[str, TruthValue]
) -> tuple[bool, list[str]]:
    unknown: list[str] = []
    for condition in conditions:
        current = state.get(condition.predicate, "unknown")
        if current == "unknown":
            unknown.append(condition.predicate)
        elif current != condition.value:
            return False, unknown
    return not unknown, unknown


def _covered_claims(evidence: Iterable[EvidenceHit]) -> set[str]:
    return {claim_id for hit in evidence for claim_id in hit.supported_claim_ids}


def _deduplicate_evidence(evidence: Iterable[EvidenceHit]) -> tuple[EvidenceHit, ...]:
    by_id: dict[str, EvidenceHit] = {}
    for hit in evidence:
        by_id[hit.evidence_id] = hit
    return tuple(by_id[evidence_id] for evidence_id in sorted(by_id))


def _label_cost(label: _Label, intents: dict[str, IntentConstraint]) -> tuple[float, ...]:
    intent_cost = sum(intents[intent_id].priority for intent_id in label.changed_intents)
    return (
        float(intent_cost),
        float(len(label.invalidated_work)),
        float(label.retrieval_calls),
        label.action_cost,
        float(len(label.action_ids)),
    )


def _label_signature(label: _Label) -> tuple[object, ...]:
    return (
        label.state,
        label.obligations,
        tuple(sorted(_covered_claims(label.evidence))),
        label.changed_intents,
        label.invalidated_work,
    )


class ReplanningEngine:
    """Shared search core for the sequential baseline and obligation-driven method."""

    def __init__(self, retriever: EvidenceRetriever) -> None:
        self.retriever = retriever

    def run(
        self,
        case: PlanningCase,
        budget: SearchBudget,
        method: PlanningMethod,
    ) -> PlanningDecision:
        initial_query = f"{case.goal_statement} {case.user_question}".strip()
        initial_evidence = self.retriever.retrieve(
            initial_query, limit=budget.initial_evidence_limit
        )
        start = _Label(
            state=tuple(sorted((item.predicate, item.value) for item in case.state)),
            action_ids=(),
            obligations=(),
            evidence=_deduplicate_evidence(initial_evidence),
            changed_intents=frozenset(),
            invalidated_work=frozenset(),
            retrieval_calls=1,
            action_cost=0.0,
        )
        return self._search(case, budget, method, start)

    def _search(
        self,
        case: PlanningCase,
        budget: SearchBudget,
        method: PlanningMethod,
        start: _Label,
    ) -> PlanningDecision:
        intents = {item.intent_id: item for item in case.intents}
        completed_work = set(case.completed_work_ids)
        queue: list[tuple[tuple[float, ...], int, _Label]] = []
        serial = count()
        start_cost = _label_cost(start, intents)
        heapq.heappush(queue, (start_cost, next(serial), start))
        best_cost_by_signature: dict[tuple[object, ...], tuple[float, ...]] = {
            _label_signature(start): start_cost
        }
        valid_labels: list[_Label] = []
        unknown_conditions: set[str] = set()
        missing_claims: set[str] = set()
        locked_conflicts: set[str] = set()
        max_retrieval_calls = start.retrieval_calls

        while queue and len(valid_labels) < budget.max_plans:
            label_cost, _, label = heapq.heappop(queue)
            if best_cost_by_signature.get(_label_signature(label)) != label_cost:
                continue
            max_retrieval_calls = max(max_retrieval_calls, label.retrieval_calls)
            covered = _covered_claims(label.evidence)
            unresolved_claims = {
                claim_id for claim_id, _ in label.obligations if claim_id not in covered
            }
            state = _state_dict(label)
            goal_reached, goal_unknown = _condition_status(case.goal, state)
            unknown_conditions.update(goal_unknown)
            if goal_reached:
                if not unresolved_claims:
                    valid_labels.append(label)
                    continue
                missing_claims.update(unresolved_claims)

            if len(label.action_ids) >= budget.max_depth:
                continue

            for action in case.actions:
                if action.action_id in label.action_ids:
                    continue
                applicable, action_unknown = _condition_status(action.preconditions, state)
                unknown_conditions.update(action_unknown)
                if not applicable:
                    continue

                locked_changes = {
                    intent_id
                    for intent_id in action.changes_intent_ids
                    if intents[intent_id].locked
                }
                if locked_changes:
                    locked_conflicts.update(locked_changes)
                    continue

                next_state = dict(state)
                for effect in action.effects:
                    next_state[effect.predicate] = effect.value
                next_obligations = dict(label.obligations)
                next_obligations.update(
                    {
                        obligation.claim_id: obligation.query
                        for obligation in action.evidence_obligations
                    }
                )
                next_evidence = list(label.evidence)
                retrieval_calls = label.retrieval_calls
                if method == "iper_rag":
                    covered_after_action = _covered_claims(next_evidence)
                    for claim_id, query in sorted(next_obligations.items()):
                        if claim_id in covered_after_action:
                            continue
                        if retrieval_calls >= budget.max_retrieval_calls:
                            break
                        hits = self.retriever.retrieve(
                            query, limit=budget.evidence_per_obligation
                        )
                        retrieval_calls += 1
                        next_evidence.extend(hits)
                        covered_after_action = _covered_claims(next_evidence)

                next_label = _Label(
                    state=tuple(sorted(next_state.items())),
                    action_ids=label.action_ids + (action.action_id,),
                    obligations=tuple(sorted(next_obligations.items())),
                    evidence=_deduplicate_evidence(next_evidence),
                    changed_intents=label.changed_intents
                    | frozenset(action.changes_intent_ids),
                    invalidated_work=label.invalidated_work
                    | frozenset(set(action.invalidates_work_ids) & completed_work),
                    retrieval_calls=retrieval_calls,
                    action_cost=label.action_cost + action.action_cost,
                )
                signature = _label_signature(next_label)
                next_cost = _label_cost(next_label, intents)
                previous_cost = best_cost_by_signature.get(signature)
                if previous_cost is not None and previous_cost <= next_cost:
                    continue
                best_cost_by_signature[signature] = next_cost
                heapq.heappush(
                    queue,
                    (next_cost, next(serial), next_label),
                )

            if len(queue) > budget.beam_width:
                queue = heapq.nsmallest(budget.beam_width, queue)
                heapq.heapify(queue)

        plans = [self._to_plan(label, intents) for label in valid_labels]
        plans.sort(
            key=lambda plan: (
                plan.intent_deviation_cost,
                plan.rework_cost,
                plan.retrieval_calls,
                plan.action_cost,
                plan.action_ids,
            )
        )
        if plans:
            decision_type: DecisionType = "propose"
            unresolved: list[str] = []
        elif missing_claims:
            decision_type = "evidence_insufficient"
            unresolved = [f"claim:{claim_id}" for claim_id in sorted(missing_claims)]
        elif unknown_conditions:
            decision_type = "clarify"
            unresolved = [
                f"state:{predicate}" for predicate in sorted(unknown_conditions)
            ]
        else:
            decision_type = "conflict"
            unresolved = [
                f"locked_intent:{intent_id}" for intent_id in sorted(locked_conflicts)
            ]

        return PlanningDecision(
            method=method,
            decision_type=decision_type,
            case_id=case.case_id,
            state_version=case.state_version,
            intent_version=case.intent_version,
            plans=plans[: budget.max_plans],
            unresolved_conditions=unresolved,
            retrieval_calls=max_retrieval_calls,
        )

    @staticmethod
    def _to_plan(
        label: _Label, intents: dict[str, IntentConstraint]
    ) -> PlanCandidate:
        changed = set(label.changed_intents)
        obligations = {claim_id for claim_id, _ in label.obligations}
        relevant_evidence = [
            hit.evidence_id
            for hit in label.evidence
            if obligations & set(hit.supported_claim_ids)
        ]
        return PlanCandidate(
            action_ids=list(label.action_ids),
            retained_intent_ids=sorted(set(intents) - changed),
            proposed_changes=sorted(changed),
            required_consent=[],
            evidence_ids=sorted(relevant_evidence),
            unresolved_conditions=[],
            intent_deviation_cost=sum(intents[item].priority for item in changed),
            rework_cost=len(label.invalidated_work),
            retrieval_calls=label.retrieval_calls,
            action_cost=label.action_cost,
        )
