from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lanternquest.experiment import EvidenceDocument, ExperimentFixture
from lanternquest.planning import (
    ActionDefinition,
    Condition,
    EvidenceObligation,
    IntentConstraint,
    PlanningCase,
    StateFact,
)

SourceId = Annotated[str, Field(pattern=r"^src_[0-9a-f]{16}$")]
ContentHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class BenchmarkFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str
    predicate: str
    value: Literal["true", "false", "unknown"]
    origin: str
    subject: str | None = None
    object: str | None = None
    source_ids: list[SourceId] = Field(default_factory=list)


class BenchmarkIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str
    description: str
    priority: int = Field(ge=1, le=3)
    locked: bool


class BenchmarkDecisionNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    prompt: str
    options: list[str] = Field(min_length=2)
    evidence_ids: list[str] = Field(min_length=1)


class BenchmarkAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    label: str
    preconditions: list[Condition] = Field(default_factory=list)
    effects: list[Condition] = Field(default_factory=list)
    evidence_obligation_ids: list[str] = Field(default_factory=list)
    changes_intent_ids: list[str] = Field(default_factory=list)
    invalidates_work_ids: list[str] = Field(default_factory=list)
    action_cost: float = Field(ge=0)
    risk_level: Literal["low", "medium", "high"] | None = None


class BenchmarkObligation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation_id: str
    query: str
    satisfied_by_evidence_ids: list[str] = Field(default_factory=list)


class BenchmarkEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: SourceId
    text: str = Field(min_length=1)
    supported_claim_ids: list[str] = Field(min_length=1)
    review_status: Literal["source_checked", "domain_approved"]


class TerminalConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: list[Condition] = Field(min_length=1)
    failure: list[Condition] = Field(default_factory=list)


class BenchmarkRights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_use_allowed: Literal[True]
    external_processing_allowed: bool
    public_release_allowed: bool | None = None
    model_training_allowed: bool | None = None
    review_status: Literal["approved", "frozen"]


class BenchmarkReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_status: Literal["domain_approved", "frozen"]
    annotation_status: Literal["adjudicated", "frozen"]
    reviewed_at: datetime
    reviewer_ids: list[str] = Field(default_factory=list)


class LanternQuestBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    case_id: str = Field(pattern=r"^lq_[a-z0-9_]+$")
    scenario_family: str
    split: Literal["train", "development", "test"]
    task_id: str
    title: str
    goal: str
    user_question: str
    state_version: str
    intent_version: str
    initial_state: list[BenchmarkFact] = Field(min_length=1)
    locked_intents: list[BenchmarkIntent] = Field(default_factory=list)
    completed_work_ids: list[str] = Field(default_factory=list)
    decision_nodes: list[BenchmarkDecisionNode] = Field(default_factory=list)
    action_space: list[BenchmarkAction] = Field(min_length=1)
    evidence_obligations: list[BenchmarkObligation] = Field(min_length=1)
    evidence_catalog: list[BenchmarkEvidence] = Field(min_length=1)
    expected_decision_type: Literal[
        "propose", "clarify", "conflict", "evidence_insufficient"
    ]
    expected_unresolved_conditions: list[str] = Field(default_factory=list)
    gold_plan: list[str] = Field(default_factory=list)
    terminal_conditions: TerminalConditions
    source_ids: list[SourceId] = Field(min_length=1)
    canonical_content_sha256: list[ContentHash] = Field(min_length=1)
    rights: BenchmarkRights
    review: BenchmarkReview

    @model_validator(mode="after")
    def validate_references(self) -> LanternQuestBenchmarkCase:
        collections = {
            "fact": [item.fact_id for item in self.initial_state],
            "intent": [item.intent_id for item in self.locked_intents],
            "decision": [item.decision_id for item in self.decision_nodes],
            "action": [item.action_id for item in self.action_space],
            "obligation": [
                item.obligation_id for item in self.evidence_obligations
            ],
            "evidence": [item.evidence_id for item in self.evidence_catalog],
            "source": self.source_ids,
            "content hash": self.canonical_content_sha256,
        }
        for label, identifiers in collections.items():
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate {label} identifiers")

        sources = set(self.source_ids)
        evidence_ids = set(collections["evidence"])
        obligation_ids = set(collections["obligation"])
        action_ids = set(collections["action"])
        intent_ids = set(collections["intent"])
        if any(
            source_id not in sources
            for fact in self.initial_state
            for source_id in fact.source_ids
        ):
            raise ValueError("initial state references an unknown source")
        if any(item.source_id not in sources for item in self.evidence_catalog):
            raise ValueError("evidence catalog references an unknown source")
        if any(
            evidence_id not in evidence_ids
            for item in self.evidence_obligations
            for evidence_id in item.satisfied_by_evidence_ids
        ):
            raise ValueError("evidence obligation references unknown evidence")
        if any(
            obligation_id not in obligation_ids
            for action in self.action_space
            for obligation_id in action.evidence_obligation_ids
        ):
            raise ValueError("action references an unknown evidence obligation")
        if any(
            intent_id not in intent_ids
            for action in self.action_space
            for intent_id in action.changes_intent_ids
        ):
            raise ValueError("action references an unknown intent")
        if any(action_id not in action_ids for action_id in self.gold_plan):
            raise ValueError("gold plan references an unknown action")
        if self.expected_decision_type == "propose" and not self.gold_plan:
            raise ValueError("a propose case requires a non-empty gold plan")
        if self.expected_decision_type != "propose" and self.gold_plan:
            raise ValueError("a non-propose case cannot contain a gold plan")
        if any(
            evidence_id not in evidence_ids
            for item in self.decision_nodes
            for evidence_id in item.evidence_ids
        ):
            raise ValueError("decision node references unknown evidence")

        claims_by_evidence = {
            item.evidence_id: set(item.supported_claim_ids)
            for item in self.evidence_catalog
        }
        for obligation in self.evidence_obligations:
            if not obligation.satisfied_by_evidence_ids:
                continue
            if not any(
                obligation.obligation_id in claims_by_evidence[evidence_id]
                for evidence_id in obligation.satisfied_by_evidence_ids
            ):
                raise ValueError(
                    f"obligation {obligation.obligation_id} lacks supporting evidence"
                )
        return self

    def to_fixture(self) -> ExperimentFixture:
        obligations = {item.obligation_id: item for item in self.evidence_obligations}
        planning_case = PlanningCase(
            case_id=self.case_id,
            goal_statement=self.goal,
            user_question=self.user_question,
            state_version=self.state_version,
            intent_version=self.intent_version,
            state=[
                StateFact(
                    predicate=item.predicate,
                    value=item.value,
                    origin=item.origin,
                )
                for item in self.initial_state
            ],
            goal=self.terminal_conditions.success,
            intents=[
                IntentConstraint(
                    intent_id=item.intent_id,
                    statement=item.description,
                    priority=item.priority,
                    locked=item.locked,
                )
                for item in self.locked_intents
            ],
            completed_work_ids=self.completed_work_ids,
            actions=[
                ActionDefinition(
                    action_id=item.action_id,
                    preconditions=item.preconditions,
                    effects=item.effects,
                    evidence_obligations=[
                        EvidenceObligation(
                            claim_id=obligations[obligation_id].obligation_id,
                            query=obligations[obligation_id].query,
                        )
                        for obligation_id in item.evidence_obligation_ids
                    ],
                    changes_intent_ids=item.changes_intent_ids,
                    invalidates_work_ids=item.invalidates_work_ids,
                    action_cost=item.action_cost,
                )
                for item in self.action_space
            ],
        )
        return ExperimentFixture(
            fixture_id=f"e21:{self.case_id}",
            purpose="frozen_evaluation",
            case=planning_case,
            evidence_catalog=[
                EvidenceDocument(
                    evidence_id=item.evidence_id,
                    source_id=item.source_id,
                    text=item.text,
                    supported_claim_ids=item.supported_claim_ids,
                    review_status=item.review_status,
                )
                for item in self.evidence_catalog
            ],
        )


def load_benchmark_jsonl(path: Path) -> list[LanternQuestBenchmarkCase]:
    cases = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(LanternQuestBenchmarkCase.model_validate_json(line))
            except Exception as error:
                raise ValueError(
                    f"invalid benchmark case at line {line_number}: {error}"
                ) from error
    case_ids = [item.case_id for item in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark has duplicate case identifiers")
    return cases


def benchmark_fingerprint(cases: list[LanternQuestBenchmarkCase]) -> str:
    payload = [
        item.model_dump(mode="json")
        for item in sorted(cases, key=lambda case: case.case_id)
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
