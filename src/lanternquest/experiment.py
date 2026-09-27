import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lanternquest.llm import LLMRequest
from lanternquest.planning import (
    EvidenceHit,
    PlanningCase,
    PlanningDecision,
    ReplanningEngine,
    SearchBudget,
)

MethodId = Literal["b0_full_context", "b1_rag", "b4_sequential", "iper_rag"]
RunStatus = Literal["awaiting_llm", "deterministic_plan_ready"]


class EvidenceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: str
    text: str
    supported_claim_ids: list[str] = Field(default_factory=list)
    review_status: Literal["source_checked", "domain_pending", "domain_approved"]


class ExperimentFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    fixture_id: str
    purpose: Literal["plumbing_smoke_test", "development", "frozen_evaluation"]
    case: PlanningCase
    evidence_catalog: list[EvidenceDocument]

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> "ExperimentFixture":
        evidence_ids = [item.evidence_id for item in self.evidence_catalog]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Experiment fixture has duplicate evidence identifiers")
        return self


class PreparedMethodRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: MethodId
    status: RunStatus
    case_id: str
    input_fingerprint: str
    corpus_fingerprint: str
    available_evidence_ids: list[str]
    retrieved_evidence_ids: list[str]
    decision: PlanningDecision | None = None
    llm_request: LLMRequest


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_fingerprint(case: PlanningCase) -> str:
    return _canonical_hash(case.model_dump(mode="json"))


def corpus_fingerprint(catalog: list[EvidenceDocument]) -> str:
    ordered = sorted(
        (item.model_dump(mode="json") for item in catalog),
        key=lambda item: str(item["evidence_id"]),
    )
    return _canonical_hash(ordered)


def _tokens(text: str) -> set[str]:
    lowered = text.casefold()
    word_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    chinese_tokens: set[str] = set()
    for run in chinese_runs:
        if len(run) == 1:
            chinese_tokens.add(run)
        else:
            chinese_tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return word_tokens | chinese_tokens


class CatalogRetriever:
    """Deterministic lexical retriever for experiment plumbing and small fixtures."""

    def __init__(self, catalog: list[EvidenceDocument]) -> None:
        self.catalog = list(catalog)
        self.calls: list[str] = []

    def retrieve(self, query: str, limit: int) -> list[EvidenceHit]:
        self.calls.append(query)
        query_tokens = _tokens(query)
        scored: list[tuple[int, str, EvidenceDocument]] = []
        for document in self.catalog:
            overlap = len(query_tokens & _tokens(document.text))
            if overlap:
                scored.append((-overlap, document.evidence_id, document))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [
            EvidenceHit(
                evidence_id=document.evidence_id,
                supported_claim_ids=tuple(document.supported_claim_ids),
                source_id=document.source_id,
                content=document.text,
            )
            for _, _, document in scored[:limit]
        ]


class ExperimentHarness:
    """Prepare equal-input method runs without silently choosing an LLM provider."""

    def __init__(self, fixture: ExperimentFixture) -> None:
        self.fixture = fixture
        self.catalog_by_id = {
            item.evidence_id: item for item in fixture.evidence_catalog
        }

    def prepare(self, method: MethodId, budget: SearchBudget) -> PreparedMethodRun:
        case = self.fixture.case
        shared_fields = {
            "method": method,
            "case_id": case.case_id,
            "input_fingerprint": input_fingerprint(case),
            "corpus_fingerprint": corpus_fingerprint(self.fixture.evidence_catalog),
            "available_evidence_ids": sorted(self.catalog_by_id),
        }

        if method == "b0_full_context":
            evidence = list(self.fixture.evidence_catalog)
            return PreparedMethodRun(
                **shared_fields,
                status="awaiting_llm",
                retrieved_evidence_ids=[item.evidence_id for item in evidence],
                llm_request=self._generation_request(method, evidence),
            )

        retriever = CatalogRetriever(self.fixture.evidence_catalog)
        if method == "b1_rag":
            query = f"{case.goal_statement} {case.user_question}".strip()
            hits = retriever.retrieve(query, budget.initial_evidence_limit)
            evidence = [self.catalog_by_id[hit.evidence_id] for hit in hits]
            return PreparedMethodRun(
                **shared_fields,
                status="awaiting_llm",
                retrieved_evidence_ids=[item.evidence_id for item in evidence],
                llm_request=self._generation_request(method, evidence),
            )

        decision = ReplanningEngine(retriever).run(case, budget, method=method)
        retrieved_ids = sorted(
            {
                evidence_id
                for plan in decision.plans
                for evidence_id in plan.evidence_ids
            }
        )
        evidence = [self.catalog_by_id[evidence_id] for evidence_id in retrieved_ids]
        return PreparedMethodRun(
            **shared_fields,
            status="deterministic_plan_ready",
            retrieved_evidence_ids=retrieved_ids,
            decision=decision,
            llm_request=self._explanation_request(method, decision, evidence),
        )

    def _generation_request(
        self, method: MethodId, evidence: list[EvidenceDocument]
    ) -> LLMRequest:
        response_schema = PlanningDecision.model_json_schema()
        response_schema["properties"]["method"] = {
            "const": method,
            "type": "string",
        }
        for field, value in (
            ("case_id", self.fixture.case.case_id),
            ("state_version", self.fixture.case.state_version),
            ("intent_version", self.fixture.case.intent_version),
        ):
            response_schema["properties"][field] = {
                "const": value,
                "type": "string",
            }
        response_schema["properties"]["plans"]["maxItems"] = 1
        candidate = response_schema["$defs"]["PlanCandidate"]["properties"]
        known_actions = sorted(item.action_id for item in self.fixture.case.actions)
        known_intents = sorted(item.intent_id for item in self.fixture.case.intents)
        known_evidence = sorted(item.evidence_id for item in evidence)
        unresolved_conditions = sorted(
            {f"state:{item.predicate}" for item in self.fixture.case.state}
            | {
                f"claim:{obligation.claim_id}"
                for action in self.fixture.case.actions
                for obligation in action.evidence_obligations
            }
            | {
                f"locked_intent:{item.intent_id}"
                for item in self.fixture.case.intents
                if item.locked
            }
        )

        def constrain_identifiers(field: str, values: list[str]) -> None:
            candidate[field]["uniqueItems"] = True
            candidate[field]["maxItems"] = len(values)
            if values:
                candidate[field]["items"]["enum"] = values

        constrain_identifiers("action_ids", known_actions)
        constrain_identifiers("retained_intent_ids", known_intents)
        constrain_identifiers("proposed_changes", known_intents)
        constrain_identifiers("required_consent", known_intents)
        constrain_identifiers("evidence_ids", known_evidence)
        constrain_identifiers("unresolved_conditions", unresolved_conditions)
        response_schema["properties"]["unresolved_conditions"][
            "uniqueItems"
        ] = True
        response_schema["properties"]["unresolved_conditions"][
            "maxItems"
        ] = len(unresolved_conditions)
        if unresolved_conditions:
            response_schema["properties"]["unresolved_conditions"]["items"][
                "enum"
            ] = unresolved_conditions
        return LLMRequest(
            purpose="plan_generation",
            system_instruction=(
                "Return only the requested structured decision. Use only identifiers "
                "present in the planning case and supplied evidence. Order actions so "
                "each precondition is true before the action. retained_intent_ids, "
                "proposed_changes, and required_consent contain intent_id values only; "
                "state predicates never belong in those fields. For a non-propose "
                "decision, return no plans and encode unresolved items only as "
                "state:<predicate>, claim:<claim_id>, or locked_intent:<intent_id>. "
                "For a propose decision, return exactly one plan and leave both "
                "unresolved_conditions arrays empty. proposed_changes contains only "
                "intent IDs actually changed by a selected action; required_consent "
                "contains only locked proposed changes; retained_intent_ids contains "
                "unchanged intents. Include enough supplied evidence to cover every "
                "selected action obligation. Unknown is not false."
            ),
            payload={
                "method": method,
                "planning_case": self.fixture.case.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            response_schema=response_schema,
        )

    def _explanation_request(
        self,
        method: MethodId,
        decision: PlanningDecision,
        evidence: list[EvidenceDocument],
    ) -> LLMRequest:
        return LLMRequest(
            purpose="grounded_explanation",
            system_instruction=(
                "Explain the validated decision without changing any identifier, adding a "
                "traditional claim, or hiding an unresolved condition."
            ),
            payload={
                "method": method,
                "decision": decision.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["user_visible_explanation"],
                "properties": {"user_visible_explanation": {"type": "string"}},
            },
        )


def validate_generated_decision(
    content: dict[str, object],
    case: PlanningCase,
    allowed_evidence_ids: set[str],
    expected_method: str | None = None,
) -> PlanningDecision:
    """Reject identifier fabrication in B0/B1 model output before presentation."""
    decision = PlanningDecision.model_validate(content)
    if decision.case_id != case.case_id:
        raise ValueError("Generated decision changed the case identifier")
    if decision.state_version != case.state_version:
        raise ValueError("Generated decision changed the state version")
    if decision.intent_version != case.intent_version:
        raise ValueError("Generated decision changed the intent version")
    if expected_method is not None and decision.method != expected_method:
        raise ValueError("Generated decision changed the method identifier")

    known_actions = {item.action_id for item in case.actions}
    known_intents = {item.intent_id: item for item in case.intents}
    for plan in decision.plans:
        unknown_actions = set(plan.action_ids) - known_actions
        unknown_evidence = set(plan.evidence_ids) - allowed_evidence_ids
        unknown_retained = set(plan.retained_intent_ids) - known_intents.keys()
        unknown_changes = set(plan.proposed_changes) - known_intents.keys()
        unknown_consent = set(plan.required_consent) - known_intents.keys()
        if unknown_retained or unknown_changes or unknown_consent:
            raise ValueError("Generated decision invented intent identifiers")
        locked_changes = {
            intent_id
            for intent_id in plan.proposed_changes
            if known_intents[intent_id].locked
        }
        if unknown_actions:
            raise ValueError("Generated decision invented action identifiers")
        if unknown_evidence:
            raise ValueError("Generated decision invented or exceeded evidence identifiers")
        if locked_changes and not locked_changes <= set(plan.required_consent):
            raise ValueError("Generated decision changes locked intent without consent")
    return decision


def write_smoke_report(
    fixture: ExperimentFixture,
    runs: list[PreparedMethodRun],
    budget: SearchBudget,
    output_path: Path,
) -> None:
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_id": fixture.fixture_id,
        "purpose": fixture.purpose,
        "reporting_boundary": (
            "Synthetic plumbing result only; not algorithm performance or domain evidence."
        ),
        "budget": budget.model_dump(mode="json"),
        "runs": [run.model_dump(mode="json") for run in runs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
