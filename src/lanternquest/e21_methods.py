from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lanternquest.benchmark import LanternQuestBenchmarkCase
from lanternquest.e21 import E21Method
from lanternquest.experiment import (
    CatalogRetriever,
    ExperimentHarness,
    PreparedMethodRun,
    corpus_fingerprint,
    input_fingerprint,
)
from lanternquest.llm import LLMRequest
from lanternquest.planning import (
    EvidenceHit,
    PlanCandidate,
    PlanningCase,
    PlanningDecision,
    ReplanningEngine,
    SearchBudget,
)


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    goal_reached: bool
    evidence_complete: bool
    locked_intents_preserved: bool
    invalid_action_ids: list[str] = Field(default_factory=list)
    unmet_preconditions: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    changed_locked_intent_ids: list[str] = Field(default_factory=list)


class E21PreparedRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: E21Method
    status: Literal["awaiting_llm", "deterministic_plan_ready"]
    case_id: str
    input_fingerprint: str
    corpus_fingerprint: str
    available_evidence_ids: list[str]
    retrieved_evidence_ids: list[str]
    decision: PlanningDecision | None = None
    llm_request: LLMRequest | None = None
    verifier_reports: list[VerificationReport] = Field(default_factory=list)
    mechanism_flags: dict[str, bool] = Field(default_factory=dict)


class GraphCatalogRetriever:
    """Fixed one-hop retrieval over goal-action-obligation-evidence links."""

    def __init__(self, case: LanternQuestBenchmarkCase) -> None:
        relevant_obligations = _goal_relevant_obligations(case)
        evidence_by_id = {item.evidence_id: item for item in case.evidence_catalog}
        selected_ids = {
            evidence_id
            for obligation in case.evidence_obligations
            if obligation.obligation_id in relevant_obligations
            for evidence_id in obligation.satisfied_by_evidence_ids
        }
        self.hits = [
            EvidenceHit(
                evidence_id=item.evidence_id,
                supported_claim_ids=tuple(item.supported_claim_ids),
                source_id=item.source_id,
                content=item.text,
            )
            for evidence_id, item in sorted(evidence_by_id.items())
            if evidence_id in selected_ids
        ]
        self.calls: list[str] = []

    def retrieve(self, query: str, limit: int) -> list[EvidenceHit]:
        self.calls.append(query)
        return self.hits[:limit]


def _goal_relevant_obligations(case: LanternQuestBenchmarkCase) -> set[str]:
    needed = {
        (condition.predicate, condition.value)
        for condition in case.terminal_conditions.success
    }
    selected_actions: set[str] = set()
    changed = True
    while changed:
        changed = False
        for action in case.action_space:
            if action.action_id in selected_actions:
                continue
            effects = {(item.predicate, item.value) for item in action.effects}
            if not effects & needed:
                continue
            selected_actions.add(action.action_id)
            needed.update(
                (item.predicate, item.value) for item in action.preconditions
            )
            changed = True
    return {
        obligation_id
        for action in case.action_space
        if action.action_id in selected_actions
        for obligation_id in action.evidence_obligation_ids
    }


def verify_plan(
    case: PlanningCase,
    plan: PlanCandidate,
    evidence_claims: dict[str, set[str]],
) -> VerificationReport:
    actions = {item.action_id: item for item in case.actions}
    state = {item.predicate: item.value for item in case.state}
    locked_intents = {item.intent_id for item in case.intents if item.locked}
    invalid_actions = []
    unmet_preconditions = []
    required_claims: set[str] = set()
    changed_locked: set[str] = set(plan.proposed_changes) & locked_intents
    for action_id in plan.action_ids:
        action = actions.get(action_id)
        if action is None:
            invalid_actions.append(action_id)
            continue
        unmet = [
            item.predicate
            for item in action.preconditions
            if state.get(item.predicate, "unknown") != item.value
        ]
        if unmet:
            unmet_preconditions.extend(
                f"{action_id}:{predicate}" for predicate in unmet
            )
            continue
        required_claims.update(
            item.claim_id for item in action.evidence_obligations
        )
        changed_locked.update(set(action.changes_intent_ids) & locked_intents)
        for effect in action.effects:
            state[effect.predicate] = effect.value
    covered_claims = {
        claim_id
        for evidence_id in plan.evidence_ids
        for claim_id in evidence_claims.get(evidence_id, set())
    }
    unsupported = sorted(required_claims - covered_claims)
    goal_reached = all(
        state.get(item.predicate, "unknown") == item.value for item in case.goal
    )
    valid = bool(
        goal_reached
        and not invalid_actions
        and not unmet_preconditions
        and not unsupported
        and not changed_locked
    )
    return VerificationReport(
        valid=valid,
        goal_reached=goal_reached,
        evidence_complete=not unsupported,
        locked_intents_preserved=not changed_locked,
        invalid_action_ids=invalid_actions,
        unmet_preconditions=unmet_preconditions,
        unsupported_claim_ids=unsupported,
        changed_locked_intent_ids=sorted(changed_locked),
    )


def _from_existing(method: E21Method, run: PreparedMethodRun) -> E21PreparedRun:
    return E21PreparedRun(
        method=method,
        status=run.status,
        case_id=run.case_id,
        input_fingerprint=run.input_fingerprint,
        corpus_fingerprint=run.corpus_fingerprint,
        available_evidence_ids=run.available_evidence_ids,
        retrieved_evidence_ids=run.retrieved_evidence_ids,
        decision=run.decision,
        llm_request=run.llm_request,
    )


def prepare_e21_method(
    case: LanternQuestBenchmarkCase,
    method: E21Method,
    budget: SearchBudget,
) -> E21PreparedRun:
    fixture = case.to_fixture()
    if method in {"b0_full_context", "b1_rag", "b4_sequential", "iper_rag"}:
        existing = ExperimentHarness(fixture).prepare(method, budget)
        return _from_existing(method, existing)

    if method == "b2_kg_rag":
        retriever = GraphCatalogRetriever(case)
        decision = ReplanningEngine(retriever).run(
            fixture.case, budget, method="b4_sequential"
        )
        decision = decision.model_copy(update={"method": method})
        return E21PreparedRun(
            method=method,
            status="deterministic_plan_ready",
            case_id=case.case_id,
            input_fingerprint=input_fingerprint(fixture.case),
            corpus_fingerprint=corpus_fingerprint(fixture.evidence_catalog),
            available_evidence_ids=sorted(
                item.evidence_id for item in fixture.evidence_catalog
            ),
            retrieved_evidence_ids=sorted(
                {hit.evidence_id for hit in retriever.hits}
            ),
            decision=decision,
            mechanism_flags={
                "fixed_graph_retrieval": True,
                "obligation_conditioned_retrieval": False,
                "symbolic_transition_verifier": False,
            },
        )

    retriever = CatalogRetriever(fixture.evidence_catalog)
    decision = ReplanningEngine(retriever).run(
        fixture.case, budget, method="iper_rag"
    )
    decision = decision.model_copy(update={"method": method})
    evidence_claims = {
        item.evidence_id: set(item.supported_claim_ids)
        for item in fixture.evidence_catalog
    }
    reports = [
        verify_plan(fixture.case, plan, evidence_claims) for plan in decision.plans
    ]
    verified_plans = [
        plan for plan, report in zip(decision.plans, reports, strict=True) if report.valid
    ]
    if len(verified_plans) != len(decision.plans):
        decision = decision.model_copy(update={"plans": verified_plans})
    return E21PreparedRun(
        method=method,
        status="deterministic_plan_ready",
        case_id=case.case_id,
        input_fingerprint=input_fingerprint(fixture.case),
        corpus_fingerprint=corpus_fingerprint(fixture.evidence_catalog),
        available_evidence_ids=sorted(
            item.evidence_id for item in fixture.evidence_catalog
        ),
        retrieved_evidence_ids=sorted(
            {evidence_id for plan in verified_plans for evidence_id in plan.evidence_ids}
        ),
        decision=decision,
        verifier_reports=reports,
        mechanism_flags={
            "fixed_graph_retrieval": False,
            "obligation_conditioned_retrieval": True,
            "symbolic_transition_verifier": True,
            "locked_intent_gate": True,
        },
    )
