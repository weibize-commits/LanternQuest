from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lanternquest.llm import LLMRequest

DynamicMethod = Literal["B4", "ENSR_base", "full_ENSR"]
DynamicDecision = Literal["proceed", "withhold", "repair", "request_consent"]
ObligationDisposition = Literal[
    "resolved", "unresolved", "reopened", "retained", "not_tracked"
]


class DynamicControllerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: DynamicMethod
    case_id: str
    decision: DynamicDecision
    obligation_id: str | None = None
    obligation_disposition: ObligationDisposition
    retrieval_requested: bool
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    evidence_ids_used: list[str] = Field(default_factory=list)
    selected_action_ids: list[str] = Field(default_factory=list)
    reexecute_action_ids: list[str] = Field(default_factory=list)
    preserve_completed_work: bool
    preserve_locked_intent: bool
    invalidate_prior_authorization: bool
    consent_required: bool
    rationale: str

    @model_validator(mode="after")
    def unique_lists(self) -> "DynamicControllerResponse":
        for name in (
            "retrieved_evidence_ids",
            "evidence_ids_used",
            "selected_action_ids",
            "reexecute_action_ids",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicates")
        return self


class DynamicCaseScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    scenario_family: str
    split: str
    perturbation_type: str
    method: DynamicMethod
    run_status: Literal["ok", "failed"]
    terminal_success: int
    persistent_obligation_resolved_correctly: int
    unsupported_action: int
    repair_locality_success: int
    evidence_trace_complete: int
    avoidable_rework_count: int
    model_calls: int
    retrieval_calls: int
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    failure_reason: str | None = None
    response: dict[str, Any] | None = None
    scoring_audit: dict[str, Any] = Field(default_factory=dict)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_by_id(base_cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in base_cases}


def completed_work_snapshot(
    dynamic_case: dict[str, Any], base_case: dict[str, Any]
) -> list[str]:
    completed = list(base_case.get("completed_work_ids", []))
    trigger = dynamic_case["dynamic_event"]["trigger"]
    trigger_action = trigger.get("action_id")
    if trigger.get("phase") != "after_action_verification" or not trigger_action:
        return list(dict.fromkeys(completed))
    plan = list(base_case.get("gold_plan", []))
    if trigger_action not in plan:
        return list(dict.fromkeys(completed))
    index = plan.index(trigger_action)
    perturbation = dynamic_case["dynamic_event"]["perturbation_type"]
    stop = index if perturbation == "failed_action_effect" else index + 1
    return list(dict.fromkeys(completed + plan[:stop]))


def observable_event(dynamic_case: dict[str, Any]) -> dict[str, Any]:
    event = dynamic_case["dynamic_event"]
    update = event["state_update"]
    common = {
        "event_id": event["event_id"],
        "perturbation_type": event["perturbation_type"],
        "trigger": event["trigger"],
    }
    perturbation = event["perturbation_type"]
    if perturbation == "new_evidence_available":
        common["observation"] = {
            "new_evidence_id": update["new_evidence_id"],
            "availability": update["availability"],
            "new_evidence_locator": update.get("new_evidence_locator"),
        }
    elif perturbation == "applicability_scope_change":
        common["observation"] = {
            "affected_obligation_id": update["affected_obligation_id"],
            "applicability": update["applicability"],
        }
    elif perturbation == "failed_action_effect":
        common["observation"] = {
            "action_id": update["action_id"],
            "expected_effect_observed": update["expected_effect_observed"],
        }
    elif perturbation == "learner_intent_change":
        common["observation"] = {
            "requested_intent_id": update["requested_intent_id"],
            "requested_change": update["requested_change"],
            "consent_recorded": update["consent_recorded"],
        }
    else:
        raise ValueError(f"unsupported perturbation type: {perturbation}")
    return common


def _compact_base_case(base_case: dict[str, Any], completed: list[str]) -> dict[str, Any]:
    return {
        "case_id": base_case["case_id"],
        "goal": base_case["goal"],
        "user_question": base_case["user_question"],
        "state_version": base_case["state_version"],
        "intent_version": base_case["intent_version"],
        "initial_state": base_case["initial_state"],
        "locked_intents": base_case["locked_intents"],
        "completed_work_ids": completed,
        "decision_nodes": base_case["decision_nodes"],
        "action_space": base_case["action_space"],
        "evidence_obligations": base_case["evidence_obligations"],
    }


def _catalog_for_method(
    method: DynamicMethod,
    dynamic_case: dict[str, Any],
    base_case: dict[str, Any],
) -> list[dict[str, Any]]:
    catalog = list(base_case["evidence_catalog"])
    event = dynamic_case["dynamic_event"]
    if method == "B4" and event["perturbation_type"] == "new_evidence_available":
        new_id = event["state_update"]["new_evidence_id"]
        catalog = [item for item in catalog if item["evidence_id"] != new_id]
    return catalog


def _ledger(dynamic_case: dict[str, Any], base_case: dict[str, Any]) -> dict[str, Any]:
    obligation_id = dynamic_case["dynamic_event"]["persistent_obligation_id"]
    obligation = next(
        item
        for item in base_case["evidence_obligations"]
        if item["obligation_id"] == obligation_id
    )
    perturbation = dynamic_case["dynamic_event"]["perturbation_type"]
    status = "unresolved" if perturbation == "new_evidence_available" else "resolved"
    return {
        "persistent_across_transitions": True,
        "active_obligation_id": obligation_id,
        "pre_event_status": status,
        "query": obligation["query"],
        "recognized_support_ids": obligation["satisfied_by_evidence_ids"],
        "authorization_rule": (
            "An action that depends on this obligation is authorized only while the "
            "obligation is resolved by reviewed, in-scope evidence. Unknown is not true."
        ),
    }


def prepare_request(
    method: DynamicMethod,
    dynamic_case: dict[str, Any],
    base_case: dict[str, Any],
) -> LLMRequest:
    completed = completed_work_snapshot(dynamic_case, base_case)
    catalog = _catalog_for_method(method, dynamic_case, base_case)
    event = observable_event(dynamic_case)
    payload: dict[str, Any] = {
        "method": method,
        "dynamic_case_id": dynamic_case["case_id"],
        "task": _compact_base_case(base_case, completed),
        "dynamic_observation": event,
        "accessible_reviewed_evidence": catalog,
    }

    if method == "full_ENSR":
        payload["persistent_evidence_obligation_ledger"] = _ledger(
            dynamic_case, base_case
        )
        system = (
            "You are the full ENSR controller. Reassess the observed transition before "
            "authorizing any next action. Persistent evidence obligations survive state "
            "transitions until reviewed, applicable evidence resolves them. Reopen an "
            "obligation when a failed effect invalidates its action support. Invalidate "
            "authorization when scope changes. Preserve completed work and locked intent "
            "outside the affected dependency branch. A learner request does not change a "
            "locked intent until explicit consent is recorded. Retrieval is allowed after "
            "the event. Use only supplied identifiers and reviewed evidence."
        )
    elif method == "ENSR_base":
        system = (
            "You are the ENSR-base controller, an evidence-guided hierarchical planner "
            "with knowledge-graph retrieval and symbolic validation. Reassess the observed "
            "event, retrieve reviewed evidence when useful, check preconditions, scope and "
            "locked intent, and select a safe next control response. This baseline has no "
            "persistent cross-transition evidence-obligation ledger. Use only supplied "
            "identifiers and reviewed evidence."
        )
    else:
        system = (
            "You are the B4 sequential baseline. Use the fixed evidence snapshot supplied "
            "before the dynamic event and process actions in sequence. Post-event retrieval "
            "is unavailable. Respect explicit preconditions and locked intent. Choose a safe "
            "response using only supplied identifiers and the fixed evidence snapshot."
        )

    schema = DynamicControllerResponse.model_json_schema()
    properties = schema["properties"]
    properties["method"] = {"type": "string", "const": method}
    properties["case_id"] = {
        "type": "string",
        "const": dynamic_case["case_id"],
    }
    action_ids = sorted(item["action_id"] for item in base_case["action_space"])
    evidence_ids = sorted(item["evidence_id"] for item in catalog)
    obligation_ids = sorted(
        item["obligation_id"] for item in base_case["evidence_obligations"]
    )
    properties["obligation_id"] = {
        "anyOf": [{"type": "string", "enum": obligation_ids}, {"type": "null"}]
    }
    for field in ("selected_action_ids", "reexecute_action_ids"):
        properties[field]["items"] = {"type": "string", "enum": action_ids}
        properties[field]["uniqueItems"] = True
        properties[field]["maxItems"] = len(action_ids)
    for field in ("retrieved_evidence_ids", "evidence_ids_used"):
        properties[field]["items"] = {"type": "string", "enum": evidence_ids}
        properties[field]["uniqueItems"] = True
        properties[field]["maxItems"] = len(evidence_ids)
    properties["rationale"]["maxLength"] = 360

    return LLMRequest(
        purpose="residual_repair",
        system_instruction=system,
        payload=payload,
        response_schema=schema,
    )


def validate_response(
    content: dict[str, Any],
    method: DynamicMethod,
    dynamic_case: dict[str, Any],
    base_case: dict[str, Any],
) -> DynamicControllerResponse:
    response = DynamicControllerResponse.model_validate(content)
    if response.method != method:
        raise ValueError("response method differs from requested method")
    if response.case_id != dynamic_case["case_id"]:
        raise ValueError("response case_id differs from requested case")
    action_ids = {item["action_id"] for item in base_case["action_space"]}
    evidence_ids = {
        item["evidence_id"]
        for item in _catalog_for_method(method, dynamic_case, base_case)
    }
    obligation_ids = {
        item["obligation_id"] for item in base_case["evidence_obligations"]
    }
    if response.obligation_id is not None and response.obligation_id not in obligation_ids:
        raise ValueError("unknown obligation identifier")
    if not set(response.selected_action_ids) <= action_ids:
        raise ValueError("unknown selected action identifier")
    if not set(response.reexecute_action_ids) <= action_ids:
        raise ValueError("unknown re-execution action identifier")
    if not set(response.retrieved_evidence_ids) <= evidence_ids:
        raise ValueError("retrieved evidence was unavailable to the method")
    if not set(response.evidence_ids_used) <= evidence_ids:
        raise ValueError("used evidence was unavailable to the method")
    if method == "B4" and response.retrieval_requested:
        raise ValueError("B4 cannot make a post-event retrieval call")
    return response


def _gold(dynamic_case: dict[str, Any], base_case: dict[str, Any]) -> dict[str, Any]:
    event = dynamic_case["dynamic_event"]
    update = event["state_update"]
    perturbation = event["perturbation_type"]
    obligation_id = event["persistent_obligation_id"]
    completed = completed_work_snapshot(dynamic_case, base_case)
    gold: dict[str, Any] = {
        "obligation_id": obligation_id,
        "completed_work_ids": completed,
        "allowed_reexecute_action_ids": [],
        "required_evidence_ids": [],
        "expected_disposition": None,
        "expected_decisions": [],
        "must_invalidate_prior_authorization": False,
        "must_require_consent": False,
        "must_preserve_locked_intent": True,
        "dependent_action_must_be_withheld": False,
    }
    if perturbation == "new_evidence_available":
        gold["required_evidence_ids"] = [update["new_evidence_id"]]
        if update["may_resolve_obligation"]:
            gold["expected_disposition"] = "resolved"
            gold["expected_decisions"] = ["proceed", "repair"]
        else:
            gold["expected_disposition"] = "unresolved"
            gold["expected_decisions"] = ["withhold"]
            gold["dependent_action_must_be_withheld"] = True
    elif perturbation == "applicability_scope_change":
        gold["expected_disposition"] = "unresolved"
        gold["expected_decisions"] = ["withhold", "repair"]
        gold["must_invalidate_prior_authorization"] = True
        gold["dependent_action_must_be_withheld"] = True
    elif perturbation == "failed_action_effect":
        gold["expected_disposition"] = "reopened"
        failed = update["failed_branch_root_action_id"]
        permitted = update.get(
            "reexecution_permitted_before_base_blocker_resolved",
            base_case["expected_decision_type"] == "propose"
            and failed in base_case["gold_plan"],
        )
        if permitted:
            gold["expected_decisions"] = ["repair"]
            plan = list(base_case["gold_plan"])
            if failed in plan:
                gold["allowed_reexecute_action_ids"] = plan[plan.index(failed) :]
            else:
                gold["allowed_reexecute_action_ids"] = [failed]
            gold["required_reexecute_root"] = failed
        else:
            gold["expected_decisions"] = ["withhold"]
            gold["dependent_action_must_be_withheld"] = True
            gold["required_reexecute_root"] = None
    elif perturbation == "learner_intent_change":
        gold["expected_disposition"] = "retained"
        gold["expected_decisions"] = ["request_consent"]
        gold["must_invalidate_prior_authorization"] = True
        gold["must_require_consent"] = True
        gold["dependent_action_must_be_withheld"] = True
    else:
        raise ValueError(f"unsupported perturbation type: {perturbation}")
    return gold


def score_response(
    dynamic_case: dict[str, Any],
    base_case: dict[str, Any],
    method: DynamicMethod,
    response: DynamicControllerResponse,
    *,
    wall_seconds: float,
    input_tokens: int,
    output_tokens: int,
) -> DynamicCaseScore:
    gold = _gold(dynamic_case, base_case)
    event_type = dynamic_case["dynamic_event"]["perturbation_type"]
    obligation_correct = int(
        response.obligation_id == gold["obligation_id"]
        and response.obligation_disposition == gold["expected_disposition"]
    )
    decision_correct = response.decision in gold["expected_decisions"]
    invalidation_correct = (
        not gold["must_invalidate_prior_authorization"]
        or response.invalidate_prior_authorization
    )
    consent_correct = (
        not gold["must_require_consent"] or response.consent_required
    )
    intent_correct = (
        not gold["must_preserve_locked_intent"] or response.preserve_locked_intent
    )
    completed = set(gold["completed_work_ids"])
    repeated_completed = completed & (
        set(response.reexecute_action_ids) | set(response.selected_action_ids)
    )
    avoidable_rework = len(repeated_completed)
    preserve_work_correct = response.preserve_completed_work and not repeated_completed
    if event_type == "failed_action_effect":
        allowed = set(gold["allowed_reexecute_action_ids"])
        root = gold["required_reexecute_root"]
        if root is None:
            repair_locality = int(
                response.decision == "withhold"
                and not response.reexecute_action_ids
                and preserve_work_correct
            )
        else:
            repair_locality = int(
                response.decision == "repair"
                and root in response.reexecute_action_ids
                and set(response.reexecute_action_ids) <= allowed
                and preserve_work_correct
            )
    else:
        repair_locality = int(preserve_work_correct)

    unsupported = 0
    if gold["dependent_action_must_be_withheld"] and response.selected_action_ids:
        unsupported = 1
    if gold["must_require_consent"] and response.decision != "request_consent":
        unsupported = 1
    if event_type == "failed_action_effect":
        allowed = set(gold["allowed_reexecute_action_ids"])
        if not set(response.reexecute_action_ids) <= allowed:
            unsupported = 1
    if repeated_completed:
        unsupported = 1

    required_evidence = set(gold["required_evidence_ids"])
    if required_evidence:
        evidence_trace = int(
            required_evidence
            <= (
                set(response.retrieved_evidence_ids)
                | set(response.evidence_ids_used)
            )
        )
    else:
        evidence_trace = int(response.obligation_id == gold["obligation_id"])

    terminal_success = int(
        obligation_correct
        and decision_correct
        and invalidation_correct
        and consent_correct
        and intent_correct
        and unsupported == 0
        and (repair_locality == 1 or event_type != "failed_action_effect")
    )
    retrieval_calls = int(response.retrieval_requested)
    audit = {
        "expected_disposition": gold["expected_disposition"],
        "expected_decisions": gold["expected_decisions"],
        "required_evidence_ids": gold["required_evidence_ids"],
        "completed_work_ids": gold["completed_work_ids"],
        "allowed_reexecute_action_ids": gold["allowed_reexecute_action_ids"],
        "decision_correct": decision_correct,
        "invalidation_correct": invalidation_correct,
        "consent_correct": consent_correct,
        "intent_correct": intent_correct,
        "preserve_work_correct": preserve_work_correct,
    }
    return DynamicCaseScore(
        case_id=dynamic_case["case_id"],
        scenario_family=dynamic_case["scenario_family"],
        split=dynamic_case["split"],
        perturbation_type=event_type,
        method=method,
        run_status="ok",
        terminal_success=terminal_success,
        persistent_obligation_resolved_correctly=obligation_correct,
        unsupported_action=unsupported,
        repair_locality_success=repair_locality,
        evidence_trace_complete=evidence_trace,
        avoidable_rework_count=avoidable_rework,
        model_calls=1,
        retrieval_calls=retrieval_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_seconds=wall_seconds,
        response=response.model_dump(mode="json"),
        scoring_audit=audit,
    )


def _pre_event_plan(base_case: dict[str, Any]) -> list[str]:
    if base_case.get("expected_decision_type") != "propose":
        return []
    return list(base_case.get("gold_plan", []))


def _next_action(
    dynamic_case: dict[str, Any], base_case: dict[str, Any]
) -> str | None:
    plan = _pre_event_plan(base_case)
    if not plan:
        return None
    completed = set(completed_work_snapshot(dynamic_case, base_case))
    return next((action_id for action_id in plan if action_id not in completed), None)


def _action_obligations(base_case: dict[str, Any], action_id: str | None) -> list[str]:
    if action_id is None:
        return []
    action = next(
        (item for item in base_case["action_space"] if item["action_id"] == action_id),
        None,
    )
    return list(action["evidence_obligation_ids"]) if action else []


def execute_frozen_controller(
    method: DynamicMethod,
    dynamic_case: dict[str, Any],
    base_case: dict[str, Any],
) -> DynamicControllerResponse:
    """Execute one prespecified controller on a shared expert-approved pre-event trace.

    This isolates transition control. The neural proposal is held fixed, so the dynamic
    experiment makes no model call and changes only controller memory and retrieval.
    """

    event = dynamic_case["dynamic_event"]
    update = event["state_update"]
    perturbation = event["perturbation_type"]
    persistent = event["persistent_obligation_id"]
    next_action = _next_action(dynamic_case, base_case)
    completed = completed_work_snapshot(dynamic_case, base_case)
    kwargs: dict[str, Any] = {
        "method": method,
        "case_id": dynamic_case["case_id"],
        "decision": "withhold",
        "obligation_id": None,
        "obligation_disposition": "not_tracked",
        "retrieval_requested": False,
        "retrieved_evidence_ids": [],
        "evidence_ids_used": [],
        "selected_action_ids": [],
        "reexecute_action_ids": [],
        "preserve_completed_work": True,
        "preserve_locked_intent": True,
        "invalidate_prior_authorization": False,
        "consent_required": False,
        "rationale": "",
    }

    if method == "full_ENSR":
        kwargs["obligation_id"] = persistent
        if perturbation == "new_evidence_available":
            evidence_id = update["new_evidence_id"]
            document = next(
                item
                for item in base_case["evidence_catalog"]
                if item["evidence_id"] == evidence_id
            )
            covers = persistent in document["supported_claim_ids"]
            kwargs.update(
                retrieval_requested=True,
                retrieved_evidence_ids=[evidence_id],
                evidence_ids_used=[evidence_id],
                obligation_disposition="resolved" if covers else "unresolved",
                decision="proceed" if covers else "withhold",
                selected_action_ids=[next_action] if covers and next_action else [],
                rationale=(
                    "The persistent ledger links the new reviewed evidence to the active "
                    "obligation and rechecks claim coverage before authorization."
                ),
            )
        elif perturbation == "applicability_scope_change":
            kwargs.update(
                decision="withhold",
                obligation_disposition="unresolved",
                retrieval_requested=True,
                invalidate_prior_authorization=True,
                rationale=(
                    "The scope transition invalidates prior authorization while the same "
                    "evidence obligation remains active."
                ),
            )
        elif perturbation == "failed_action_effect":
            permitted = bool(
                update.get("reexecution_permitted_before_base_blocker_resolved", False)
            )
            kwargs.update(
                decision="repair" if permitted else "withhold",
                obligation_disposition="reopened",
                reexecute_action_ids=[update["action_id"]] if permitted else [],
                rationale=(
                    "The failed transition reopens its linked obligation. The controller "
                    "repairs only the affected branch when the pre-event action was authorized."
                ),
            )
        elif perturbation == "learner_intent_change":
            kwargs.update(
                decision="request_consent",
                obligation_disposition="retained",
                invalidate_prior_authorization=True,
                consent_required=True,
                rationale=(
                    "Goal-specific authorization is paused, explicit consent is requested, "
                    "and the applicable source-linked obligation is retained."
                ),
            )

    elif method == "ENSR_base":
        if perturbation == "new_evidence_available":
            evidence_id = update["new_evidence_id"]
            document = next(
                item
                for item in base_case["evidence_catalog"]
                if item["evidence_id"] == evidence_id
            )
            current_obligations = _action_obligations(base_case, next_action)
            matched = sorted(set(document["supported_claim_ids"]) & set(current_obligations))
            if matched:
                kwargs.update(
                    decision="proceed",
                    obligation_id=matched[0],
                    obligation_disposition="resolved",
                    selected_action_ids=[next_action] if next_action else [],
                )
            elif current_obligations:
                kwargs.update(
                    decision="withhold",
                    obligation_id=current_obligations[0],
                    obligation_disposition="unresolved",
                )
            kwargs.update(
                retrieval_requested=True,
                retrieved_evidence_ids=[evidence_id],
                evidence_ids_used=[evidence_id],
                rationale=(
                    "Reactive graph retrieval compares the new evidence with the next "
                    "action, without a cross-transition obligation ledger."
                ),
            )
        elif perturbation == "applicability_scope_change":
            kwargs.update(
                decision="withhold",
                invalidate_prior_authorization=True,
                retrieval_requested=True,
                rationale=(
                    "The reactive symbolic gate withholds the next action after the scope "
                    "change, but no persistent obligation state is available."
                ),
            )
        elif perturbation == "failed_action_effect":
            action_id = update["action_id"]
            obligations = _action_obligations(base_case, action_id)
            permitted = bool(
                update.get("reexecution_permitted_before_base_blocker_resolved", False)
            )
            kwargs.update(
                decision="repair" if permitted else "withhold",
                obligation_id=obligations[0] if obligations else None,
                obligation_disposition="reopened" if obligations else "not_tracked",
                reexecute_action_ids=[action_id] if permitted else [],
                rationale=(
                    "The failed action exposes its directly attached evidence obligation, "
                    "which the reactive controller can reopen locally."
                ),
            )
        elif perturbation == "learner_intent_change":
            kwargs.update(
                decision="request_consent",
                invalidate_prior_authorization=True,
                consent_required=True,
                rationale=(
                    "The locked-intent gate requests consent, but no persistent evidence "
                    "obligation is carried across the goal transition."
                ),
            )

    else:
        if perturbation == "new_evidence_available":
            kwargs.update(
                decision="proceed" if next_action else "withhold",
                selected_action_ids=[next_action] if next_action else [],
                rationale=(
                    "The sequential controller continues from its fixed pre-event evidence "
                    "snapshot and cannot retrieve the newly available record."
                ),
            )
        elif perturbation == "applicability_scope_change":
            kwargs.update(
                decision="proceed" if next_action else "withhold",
                selected_action_ids=[next_action] if next_action else [],
                rationale=(
                    "The sequential controller has no event-linked obligation state and "
                    "continues its frozen sequence when one exists."
                ),
            )
        elif perturbation == "failed_action_effect":
            permitted = bool(
                update.get("reexecution_permitted_before_base_blocker_resolved", False)
            )
            plan = _pre_event_plan(base_case)
            kwargs.update(
                decision="repair" if permitted else "withhold",
                reexecute_action_ids=plan if permitted else [],
                preserve_completed_work=not bool(completed),
                rationale=(
                    "The sequential controller restarts its available sequence after an "
                    "authorized failed transition, or remains at the inherited blocker."
                ),
            )
        elif perturbation == "learner_intent_change":
            kwargs.update(
                decision="request_consent",
                invalidate_prior_authorization=True,
                consent_required=True,
                rationale=(
                    "The locked-intent check requests consent, without carrying an evidence "
                    "obligation across the transition."
                ),
            )

    return DynamicControllerResponse.model_validate(kwargs)


def failed_score(
    dynamic_case: dict[str, Any],
    method: DynamicMethod,
    *,
    wall_seconds: float,
    failure_reason: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> DynamicCaseScore:
    return DynamicCaseScore(
        case_id=dynamic_case["case_id"],
        scenario_family=dynamic_case["scenario_family"],
        split=dynamic_case["split"],
        perturbation_type=dynamic_case["dynamic_event"]["perturbation_type"],
        method=method,
        run_status="failed",
        terminal_success=0,
        persistent_obligation_resolved_correctly=0,
        unsupported_action=1,
        repair_locality_success=0,
        evidence_trace_complete=0,
        avoidable_rework_count=0,
        model_calls=1,
        retrieval_calls=0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_seconds=wall_seconds,
        failure_reason=failure_reason,
    )


def load_jsonl(path: Any) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def join_dynamic_to_base(
    dynamic_cases: list[dict[str, Any]], base_cases: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = _base_by_id(base_cases)
    output = []
    for dynamic in dynamic_cases:
        base_id = dynamic["base_case_id"]
        if base_id not in by_id:
            raise ValueError(f"missing base case: {base_id}")
        base = by_id[base_id]
        if canonical_hash(base) != dynamic["base_case_sha256"]:
            raise ValueError(f"base case hash mismatch: {base_id}")
        output.append((dynamic, base))
    return output
