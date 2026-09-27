import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lanternquest.evidence_scheduler import EvidenceDebtLedger, EvidenceDebtSignal
from lanternquest.llm import LLMAdapter, LLMAdapterError, LLMRequest
from lanternquest.verified_repair import (
    ActionTransition,
    VerifiedRepairCandidate,
    fact_key,
    infer_current_action_transitions,
    residual_goal_diff,
    search_verified_transition_graph,
    symbolic_state_sha256,
    validate_selected_candidate,
)

try:
    from scripts.scienceworld_ensr_components import (
        EvidenceObligation,
        FactPattern,
        Subgoal,
        TemporalFact,
        TransitionAssessment,
        observation_sha256,
        parse_observation_facts,
        shortlist_actions,
        verify_transition,
    )
except ModuleNotFoundError:  # Direct script execution uses the scripts directory.
    from scienceworld_ensr_components import (  # type: ignore[no-redef]
        EvidenceObligation,
        FactPattern,
        Subgoal,
        TemporalFact,
        TransitionAssessment,
        observation_sha256,
        parse_observation_facts,
        shortlist_actions,
        verify_transition,
    )

SupportedRelation = Literal[
    "located_in",
    "visible_in",
    "has_state",
    "in_inventory_of",
    "contains",
]
ENSRStatus = Literal[
    "terminal_success",
    "terminal_failure",
    "environment_budget_exhausted",
    "model_budget_exhausted",
    "invalid_decision_limit",
    "adapter_error",
    "environment_error",
    "prefix_replay_error",
    "plan_exhausted",
]


class ENSREnvironment(Protocol):
    def load(
        self,
        task_name: str,
        variationIdx: int = 0,
        simplificationStr: str = "",
        generateGoldPath: bool = False,
    ) -> None: ...

    def reset(self) -> tuple[str, dict[str, Any]]: ...

    def step(self, action: str) -> tuple[str, int, bool, dict[str, Any]]: ...


class PlannerFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    relation: SupportedRelation
    object: str
    polarity: Literal[True]

    def as_fact_pattern(self) -> FactPattern:
        return FactPattern.model_validate(self.model_dump())


class PlannerSubgoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoal_id: str
    description: str
    expected_facts: list[PlannerFact] = Field(min_length=1, max_length=4)
    completion_test: str
    completion_mode: Literal["all", "any"] = "all"
    retry_limit: int = Field(ge=1, le=4)

    def as_subgoal(self) -> Subgoal:
        return Subgoal(
            subgoal_id=self.subgoal_id,
            description=self.description,
            expected_facts=[fact.as_fact_pattern() for fact in self.expected_facts],
            completion_test=self.completion_test,
            completion_mode=self.completion_mode,
            retry_limit=self.retry_limit,
        )


class HierarchicalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_summary: str
    subgoals: list[PlannerSubgoal] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def unique_subgoal_ids(self) -> "HierarchicalPlan":
        identifiers = [subgoal.subgoal_id for subgoal in self.subgoals]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("subgoal_id values must be unique")
        return self


class LocalReplan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis: str
    replacement_subgoal: PlannerSubgoal


class ActionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    expected_fact: PlannerFact | None
    expected_progress: str


class ResidualRepairProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis: str
    residual_subgoal: PlannerSubgoal
    next_action: str
    expected_progress: str


class CandidateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str


class ENSRBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_environment_steps: int = Field(default=100, ge=1, le=500)
    max_model_calls: int = Field(default=100, ge=1, le=500)
    max_retrieval_calls: int = Field(default=8, ge=1, le=100)
    max_planner_calls: int = Field(default=4, ge=1, le=20)
    initial_retrieval_k: int = Field(default=5, ge=1, le=50)
    obligation_retrieval_k: int = Field(default=3, ge=1, le=50)
    candidate_action_limit: int = Field(default=20, ge=3, le=100)
    max_invalid_decisions: int = Field(default=5, ge=1, le=20)


def _local_replan_has_budget(
    *,
    planner_calls: int,
    max_planner_calls: int,
    adaptive_slow_path_enabled: bool,
    adaptive_continuation_plan_count: int,
) -> bool:
    """Keep the remaining adaptive continuation calls available for recovery."""
    continuation_reserve = 0
    if adaptive_slow_path_enabled:
        continuation_reserve = max(0, 2 - adaptive_continuation_plan_count)
    return planner_calls < max(0, max_planner_calls - continuation_reserve)


class ENSRStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    subgoal_id: str
    action: str
    action_source: Literal[
        "procedure",
        "symbolic_goal",
        "exploration",
        "llm",
        "controlled_repair",
        "verified_graph_repair",
        "replayed_control",
    ] = "llm"
    expected_fact: FactPattern | None
    expected_progress: str
    reward: int
    score: int
    completed: bool
    observation_sha256: str
    fact_count: int
    verified_expected_fact: bool
    obligation_ids: list[str] = Field(default_factory=list)


class ENSRRetrievalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_index: int
    trigger: Literal["initial", "obligation"]
    query: str
    returned_fragment_ids: list[str]
    new_fragment_ids: list[str]
    replaced_fragment_ids: list[str]
    obligation_id: str | None = None


class ENSREvidenceScheduleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation_id: str
    debt_signature: str
    source: str
    evidence_state: Literal["unknown", "contradicted", "resolved"]
    schedule_action: Literal["retrieve", "reuse_active_evidence", "defer"]
    requested_k: int
    priority: float
    severity: float
    uncertainty: float
    candidate_novelty: float
    budget_ratio: float
    occurrence_count: int
    reason: str
    candidate_fragment_ids: list[str]
    selected_fragment_ids: list[str]


class ENSREpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0"
    experiment_id: str = "E12"
    reporting_boundary: str
    method: Literal["ensr_v2"] = "ensr_v2"
    model_id: str
    model_profile_id: str | None = None
    resolved_model_ids: list[str] = Field(default_factory=list)
    system_fingerprints: list[str] = Field(default_factory=list)
    task_id: str
    task_name: str
    split: Literal["dev", "test"]
    variation: int
    seed: int
    status: ENSRStatus
    completed: bool
    task_success: bool
    final_score: int
    environment_steps: int
    model_calls: int
    planner_calls: int
    retrieval_calls: int
    invalid_decisions: int
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    active_fragment_ids: list[str]
    retrieval_events: list[ENSRRetrievalEvent]
    evidence_schedule_events: list[ENSREvidenceScheduleEvent] = Field(
        default_factory=list
    )
    evidence_debt_ledger: list[dict[str, Any]] = Field(default_factory=list)
    plan: HierarchicalPlan | None
    adaptive_recovery_plans: list[HierarchicalPlan] = Field(default_factory=list)
    completed_subgoal_ids: list[str]
    obligations: list[EvidenceObligation]
    trajectory: list[ENSRStep]
    temporal_fact_history: list[TemporalFact]
    mechanism_metrics: dict[str, int | float]
    response_normalizations: dict[str, int] = Field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _task_target_action_tokens(task_target: str) -> set[str]:
    tokens = _tokens(task_target) - {"liquid", "solid"}
    if re.fullmatch(
        r"(?:liquid\s+|solid\s+)?unknown substance [a-z]",
        task_target.casefold().strip(),
    ):
        tokens.discard(task_target.casefold().strip()[-1])
    return tokens


def _state_text(info: dict[str, Any]) -> str:
    return "\n".join(
        value
        for value in (str(info.get("look", "")), str(info.get("inv", "")))
        if value
    )


def _fact_prompt(fact: TemporalFact | FactPattern) -> dict[str, Any]:
    return {
        "subject": fact.subject,
        "relation": fact.relation,
        "object": fact.object,
        "polarity": fact.polarity,
    }


def _facts_satisfied(
    expected: list[FactPattern],
    observed: list[TemporalFact],
    completion_mode: Literal["all", "any"] = "all",
    achieved_keys: set[tuple[str, str, str, bool]] | None = None,
) -> bool:
    observed_keys = {fact.key for fact in observed}
    expected_keys = {fact.key for fact in expected}
    if achieved_keys is not None:
        achieved_keys.update(expected_keys & observed_keys)
        evidence_keys = observed_keys | achieved_keys
    else:
        evidence_keys = observed_keys
    checks = [fact.key in evidence_keys for fact in expected]
    return bool(checks) and (
        any(checks) if completion_mode == "any" else all(checks)
    )


def _explicit_task_target(task_description: str) -> str | None:
    for pattern in (
        r"your task is to (?:boil|melt|freeze) (.+?)\.",
        r"your task is to change the state of matter of (.+?)\.",
        r"your task is to use chemistry to create the substance ['\"](.+?)['\"]\.",
        r"your task is to use chemistry to create ([^.]+)\. when you are done",
        r"your task is to turn on (.+?)\.",
    ):
        match = re.search(pattern, task_description, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _procedure_task_target(task_description: str) -> str | None:
    explicit = _explicit_task_target(task_description)
    if explicit:
        return explicit
    for pattern in (
        r"your task is to measure the temperature of (.+?), which",
        r"your task is to measure the melting point of (.+?), which",
        r"your task is to determine if (.+?) is electrically conductive",
    ):
        match = re.search(pattern, task_description, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _focus_target_stages(
    task_description: str, *, explicit_task_target: str | None = None
) -> list[list[str]]:
    """Extract ordered focus targets, grouping conditional alternatives."""
    stages: list[list[str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", task_description):
        targets = []
        for match in re.finditer(
                r"\bfocus on (?:the\s+)?(.+?)(?=[.,;]|$)",
                sentence,
                flags=re.IGNORECASE,
        ):
            target = match.group(1).strip().removeprefix("the ").strip()
            if explicit_task_target and target.casefold() in {
                "object",
                "substance",
                "target",
            }:
                target = explicit_task_target
            targets.append(target)
        if not targets:
            continue
        if sentence.lstrip().casefold().startswith("if "):
            if stages and stages[-1] and stages[-1][0].startswith("__conditional__:"):
                stages[-1].extend(targets)
            else:
                stages.append([f"__conditional__:{targets[0]}", *targets[1:]])
        else:
            stages.extend([[target] for target in targets])
    for stage in stages:
        if stage and stage[0].startswith("__conditional__:"):
            stage[0] = stage[0].split(":", maxsplit=1)[1]
    return stages


def _focus_action_matches_target(action: str, target: str) -> bool:
    lowered = action.casefold().strip()
    if not lowered.startswith("focus on "):
        return False
    focus_object = lowered[len("focus on ") :].strip()
    target_key = target.casefold().strip().removeprefix("the ").strip()
    target_forms = {target_key}
    if target_key.endswith(" box"):
        target_forms.add(target_key.removesuffix(" box").strip())
    if re.fullmatch(r"unknown substance [a-z]", target_key):
        target_forms.add("unknown substance")
    return any(
        focus_object == form
        or focus_object.startswith(f"{form} in ")
        or focus_object.startswith(f"{form} (")
        or focus_object == f"substance called {form}"
        for form in target_forms
    )


def _filter_irreversible_focus_actions(
    actions: list[str],
    *,
    task_description: str,
    visible_state: str,
    allowed_focus_targets: list[str] | None = None,
) -> list[str]:
    """Prevent an irreversible focus on an object other than the named target."""
    actions = [
        action for action in actions if action.casefold().strip() != "focus on agent"
    ]
    if allowed_focus_targets is not None:
        visible_key = visible_state.casefold()
        visible_targets = [
            target
            for target in allowed_focus_targets
            if target.casefold().removeprefix("the ").removesuffix(" box")
            in visible_key
        ]
        return [
            action
            for action in actions
            if not action.casefold().startswith("focus on ")
            or any(
                _focus_action_matches_target(action, target)
                for target in visible_targets
            )
        ]
    target = _explicit_task_target(task_description)
    if not target:
        return actions
    target_key = target.casefold().strip()
    target_visible = target_key in visible_state.casefold()
    filtered: list[str] = []
    for action in actions:
        if not action.casefold().startswith("focus on "):
            filtered.append(action)
            continue
        if target_visible and _focus_action_matches_target(action, target):
            filtered.append(action)
    return filtered


def _repair_fact_is_grounded(
    fact: FactPattern,
    *,
    task_description: str,
    visible_state: str,
    legal_actions: list[str],
) -> bool:
    """Require every repair entity to occur in the current grounded context."""
    grounded_text = "\n".join(
        [task_description, visible_state, *legal_actions]
    ).casefold()
    subject = fact.subject.casefold().strip()
    object_value = fact.object.casefold().strip()
    subject_grounded = subject == "agent" or subject in grounded_text
    if fact.relation == "in_inventory_of":
        object_grounded = object_value == "agent"
    elif fact.relation == "has_state":
        object_grounded = object_value in grounded_text or object_value in {
            "open",
            "closed",
            "on",
            "off",
            "activated",
            "deactivated",
            "solid",
            "liquid",
            "gas",
            "complete",
        }
    else:
        object_grounded = object_value == "agent" or object_value in grounded_text
    return subject_grounded and object_grounded


def _open_blocking_container_action(
    eligible_actions: list[str],
    *,
    entity: str,
    observed_facts: list[TemporalFact],
) -> str | None:
    fact_keys = {fact.key for fact in observed_facts}
    blocking_containers = sorted(
        {
            fact.subject
            for fact in observed_facts
            if fact.relation == "contains"
            and fact.key[2] == entity
            and (fact.key[0], "has_state", "closed", True) in fact_keys
        }
    )
    for container in blocking_containers:
        action = next(
            (
                candidate
                for candidate in sorted(eligible_actions, key=str.casefold)
                if candidate.casefold() == f"open {container}"
            ),
            None,
        )
        if action:
            return action
    return None


def _open_unknown_tool_container_action(
    eligible_actions: list[str], observed_facts: list[TemporalFact]
) -> str | None:
    closed_containers = sorted(
        {
            fact.subject
            for fact in observed_facts
            if fact.relation == "has_state" and fact.key[2] == "closed"
        }
    )
    for container in closed_containers:
        action = next(
            (
                candidate
                for candidate in sorted(eligible_actions, key=str.casefold)
                if candidate.casefold() == f"open {container}"
            ),
            None,
        )
        if action:
            return action
    return None


def _is_portable_vessel(entity: str) -> bool:
    return bool(
        _tokens(entity)
        & {"beaker", "bottle", "bowl", "cup", "flask", "jar", "jug", "pot"}
    )


def _task_target_acquisition_action(
    eligible_actions: list[str],
    *,
    task_target: str | None,
    observed_facts: list[TemporalFact],
) -> str | None:
    if not task_target:
        return None
    target_key = FactPattern(
        subject=task_target,
        relation="in_inventory_of",
        object="agent",
    ).key[0]
    fact_keys = {fact.key for fact in observed_facts}
    if (target_key, "in_inventory_of", "agent", True) in fact_keys:
        return None
    if any(
        fact.relation == "contains"
        and fact.key[2] == target_key
        and _is_portable_vessel(fact.key[0])
        for fact in observed_facts
    ):
        return None
    blocking_action = _open_blocking_container_action(
        eligible_actions,
        entity=target_key,
        observed_facts=observed_facts,
    )
    if blocking_action:
        return blocking_action
    candidates = sorted(
        (
            action
            for action in eligible_actions
            if (
                action.casefold().startswith(f"pick up {target_key}")
                or (
                    action.casefold().startswith(f"move {target_key}")
                    and "inventory" in action.casefold()
                )
            )
        ),
        key=lambda action: (
            0 if action.casefold().startswith("pick up ") else 1,
            action.casefold(),
        ),
    )
    return candidates[0] if candidates else None


_ACTION_MATCH_STOPWORDS = {
    "a",
    "an",
    "containing",
    "door",
    "from",
    "in",
    "inventory",
    "of",
    "on",
    "the",
    "to",
}


def _next_procedure_action(
    eligible_actions: list[str],
    procedure_actions: list[str],
    *,
    cursor: int,
    task_target: str | None,
    focus_object: str | None = None,
    lookahead: int = 6,
) -> tuple[str, int] | None:
    """Bind the next viable abstract train action to a current legal action."""
    target = task_target or ""
    focus = focus_object or target
    for index in range(cursor, min(len(procedure_actions), cursor + lookahead)):
        template = procedure_actions[index]
        if task_target is None and re.search(
            r"<task_target>", template, flags=re.IGNORECASE
        ):
            continue
        if not focus and re.search(r"<focus_object>", template, flags=re.IGNORECASE):
            continue
        instantiated = (
            template
            .casefold()
            .replace("<task_target>", target.casefold())
            .replace("<focus_object>", focus.casefold())
        )
        procedure_tokens = _tokens(instantiated) - _ACTION_MATCH_STOPWORDS
        if not procedure_tokens:
            continue
        operator = instantiated.split(maxsplit=1)[0]
        matches: list[tuple[float, str]] = []
        for action in eligible_actions:
            lowered = action.casefold()
            examine_alias = operator == "examine" and lowered.startswith("look at ")
            if lowered.split(maxsplit=1)[0] != operator and not examine_alias:
                continue
            action_tokens = _tokens(lowered) - _ACTION_MATCH_STOPWORDS
            if examine_alias:
                action_tokens.add("examine")
            binding_mismatch = False
            for placeholder, binding in (
                ("<task_target>", task_target),
                ("<focus_object>", focus_object or task_target),
            ):
                if placeholder not in template.casefold() or not binding:
                    continue
                binding_tokens = _tokens(binding) - {"liquid", "solid"}
                if not binding_tokens <= action_tokens:
                    binding_mismatch = True
                    break
                prefix = template.casefold().split(placeholder, maxsplit=1)[0].strip()
                if prefix in {
                    "connect",
                    "drop",
                    "examine",
                    "focus on",
                    "look at",
                    "move",
                    "pick up",
                    "use",
                }:
                    action_argument = (
                        lowered.removeprefix("look at").strip()
                        if prefix == "examine" and examine_alias
                        else lowered.removeprefix(prefix).strip()
                    )
                    binding_key = binding.casefold().strip()
                    binding_without_state = re.sub(
                        r"^(?:liquid|solid)\s+", "", binding_key
                    )
                    if not (
                        action_argument.startswith(binding_key)
                        or action_argument.startswith(binding_without_state)
                    ):
                        binding_mismatch = True
                        break
            if binding_mismatch:
                continue
            template_subject = _manipulation_subject_tokens(instantiated)
            action_subject = _manipulation_subject_tokens(lowered)
            if template_subject and not template_subject <= action_subject:
                continue
            coverage = len(procedure_tokens & action_tokens) / len(procedure_tokens)
            if coverage >= 0.75:
                matches.append((coverage, action))
        if matches:
            matches.sort(key=lambda item: (-item[0], item[1].casefold()))
            return matches[0][1], index + 1
    return None


def _manipulation_subject_tokens(action: str) -> set[str]:
    lowered = action.casefold().strip()
    patterns = (
        r"^(?:move|pour|dunk|connect)\s+(.+?)(?:\s+to\s+|\s+into\s+)",
        r"^use\s+(.+?)\s+on\s+",
        r"^(?:pick up|focus on|look at|look in|drop|examine|activate|deactivate|mix)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return _tokens(match.group(1)) - {
                "containing",
                "in",
                "inventory",
                "liquid",
                "nothing",
                "solid",
            }
    return set()


def _measurement_conditioned_focus_action(
    task_description: str,
    focus_actions: list[str],
    transition_temperature: float | None,
) -> str | None:
    if transition_temperature is None:
        return None
    above = re.search(
        r"above\s+(-?\d+(?:\.\d+)?)\s+degrees celsius,\s*focus on (?:the\s+)?(.+?)(?:\.|$)",
        task_description,
        flags=re.IGNORECASE,
    )
    below = re.search(
        r"below\s+(-?\d+(?:\.\d+)?)\s+degrees celsius,\s*focus on (?:the\s+)?(.+?)(?:\.|$)",
        task_description,
        flags=re.IGNORECASE,
    )
    if not above or not below:
        return None
    threshold = float(above.group(1))
    target = above.group(2) if transition_temperature > threshold else below.group(2)
    return next(
        (
            action
            for action in focus_actions
            if _focus_action_matches_target(action, target)
        ),
        None,
    )


def _measurement_decision_temperature(
    task_description: str,
    task_target: str,
    visible_state: str,
    last_temperature: float | None,
) -> float | None:
    if last_temperature is None:
        return None
    initial_phase_match = re.match(
        r"^(solid|liquid)\s+(.+)$", task_target.casefold()
    )
    threshold_match = re.search(
        r"above\s+(-?\d+(?:\.\d+)?)\s+degrees celsius",
        task_description,
        flags=re.IGNORECASE,
    )
    if not initial_phase_match or not threshold_match:
        return None
    initial_phase, target_base = initial_phase_match.groups()
    opposite_phase = "liquid" if initial_phase == "solid" else "solid"
    state_key = visible_state.casefold()
    if f"{opposite_phase} {target_base}" in state_key:
        return last_temperature
    threshold = float(threshold_match.group(1))
    if initial_phase == "solid" and f"solid {target_base}" in state_key:
        return last_temperature if last_temperature > threshold else None
    if initial_phase == "liquid" and f"liquid {target_base}" in state_key:
        return last_temperature if last_temperature < threshold else None
    return None


def _measurement_poll_action(
    eligible_actions: list[str],
    task_target: str,
    *,
    prefer_examine: bool,
) -> str | None:
    target_tokens = _tokens(task_target) - {"liquid", "solid"}
    examine_actions = sorted(
        (
            action
            for action in eligible_actions
            if action.casefold().startswith(("examine ", "look at "))
            and target_tokens <= _tokens(action)
        ),
        key=str.casefold,
    )
    thermometer_actions = sorted(
        (
            action
            for action in eligible_actions
            if action.casefold().startswith("use thermometer")
            and target_tokens <= _tokens(action)
        ),
        key=str.casefold,
    )
    ordered = (
        (examine_actions, thermometer_actions)
        if prefer_examine
        else (thermometer_actions, examine_actions)
    )
    return next((actions[0] for actions in ordered if actions), None)


def _conditional_conductivity_boxes(
    task_description: str,
) -> tuple[str, str] | None:
    conductive = re.search(
        r"if it is electrically conductive,\s*place it in (?:the\s+)?(.+? box)\.",
        task_description,
        flags=re.IGNORECASE,
    )
    nonconductive = re.search(
        r"if it is electrically nonconductive,\s*place it in (?:the\s+)?(.+? box)\.",
        task_description,
        flags=re.IGNORECASE,
    )
    if not conductive or not nonconductive:
        return None
    return conductive.group(1).strip(), nonconductive.group(1).strip()


def _connect_endpoints(action: str) -> tuple[str, str] | None:
    match = re.match(r"^connect\s+(.+?)\s+to\s+(.+)$", action.casefold().strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _wire_endpoint(endpoint: str) -> tuple[str, int] | None:
    match = re.search(r"\b([a-z]+) wire terminal ([12])\b", endpoint)
    if match:
        return match.group(1), int(match.group(2))
    inverse = re.search(r"\bterminal ([12]) in ([a-z]+) wire\b", endpoint)
    if inverse:
        return inverse.group(2), int(inverse.group(1))
    return None


def _detector_endpoint(endpoint: str) -> tuple[str, str] | None:
    terminal = "anode" if "anode" in endpoint else "cathode" if "cathode" in endpoint else None
    if terminal is None:
        return None
    component = re.sub(r"^(?:anode|cathode)\s+(?:in|on)\s+", "", endpoint)
    component = re.sub(r"\s+(?:anode|cathode)$", "", component)
    if not re.search(r"\b(?:light bulb|electric motor|buzzer)\b", component):
        return None
    return component.strip(), terminal


def _conductivity_probe_components(
    eligible_actions: list[str],
) -> tuple[str, str, str, str] | None:
    wires: set[str] = set()
    detector_terminals: dict[str, set[str]] = {}
    for action in eligible_actions:
        endpoints = _connect_endpoints(action)
        if endpoints is None:
            continue
        for endpoint in endpoints:
            wire = _wire_endpoint(endpoint)
            if wire is not None:
                wires.add(wire[0])
            detector = _detector_endpoint(endpoint)
            if detector is not None:
                detector_terminals.setdefault(detector[0], set()).add(detector[1])
    detectors = sorted(
        (
            component
            for component, terminals in detector_terminals.items()
            if {"anode", "cathode"} <= terminals
        ),
        key=lambda component: (
            0 if "light bulb" in component else 1,
            component,
        ),
    )
    ordered_wires = sorted(wires)
    if len(ordered_wires) < 3 or not detectors:
        return None
    return ordered_wires[0], ordered_wires[1], ordered_wires[2], detectors[0]


def _endpoint_has_battery_terminal(endpoint: str, terminal: str) -> bool:
    return "battery" in endpoint and terminal in endpoint


def _endpoint_has_detector_terminal(
    endpoint: str, detector: str, terminal: str
) -> bool:
    parsed = _detector_endpoint(endpoint)
    return parsed == (detector, terminal)


def _endpoint_has_target(endpoint: str, task_target: str) -> bool:
    target_tokens = _task_target_action_tokens(task_target)
    return bool(target_tokens) and target_tokens <= _tokens(endpoint)


def _find_connection_action(
    eligible_actions: list[str],
    left_matches,
    right_matches,
) -> str | None:
    matches: list[str] = []
    for action in eligible_actions:
        endpoints = _connect_endpoints(action)
        if endpoints is None:
            continue
        left, right = endpoints
        if (left_matches(left) and right_matches(right)) or (
            left_matches(right) and right_matches(left)
        ):
            matches.append(action)
    return min(
        matches,
        key=lambda action: (
            0 if " in inventory" in action.casefold() else 1,
            0 if " in room" in action.casefold() else 1,
            action.casefold(),
        ),
        default=None,
    )


def _conductivity_probe_action(
    eligible_actions: list[str],
    task_target: str,
    components: tuple[str, str, str, str],
    *,
    connection_index: int,
    wait_count: int,
) -> tuple[str, str] | None:
    wire_a, wire_b, wire_c, detector = components

    def wire(name: str, terminal: int):
        return lambda endpoint: _wire_endpoint(endpoint) == (name, terminal)

    steps = (
        (
            lambda endpoint: _endpoint_has_battery_terminal(endpoint, "anode"),
            wire(wire_a, 1),
        ),
        (
            lambda endpoint: _endpoint_has_battery_terminal(endpoint, "cathode"),
            wire(wire_b, 1),
        ),
        (
            wire(wire_a, 2),
            lambda endpoint: _endpoint_has_detector_terminal(
                endpoint, detector, "cathode"
            ),
        ),
        (
            wire(wire_c, 2),
            lambda endpoint: _endpoint_has_detector_terminal(
                endpoint, detector, "anode"
            ),
        ),
        (
            lambda endpoint: _endpoint_has_target(endpoint, task_target),
            wire(wire_b, 2),
        ),
        (
            lambda endpoint: _endpoint_has_target(endpoint, task_target),
            wire(wire_c, 1),
        ),
    )
    if connection_index < len(steps):
        action = _find_connection_action(
            eligible_actions,
            *steps[connection_index],
        )
        return (action, "connection") if action is not None else None
    if wait_count < 2:
        wait_action = next(
            (
                action
                for action in eligible_actions
                if action.casefold() in {"wait1", "wait"}
            ),
            None,
        )
        return (wait_action, "wait") if wait_action is not None else None
    return None


def _conductivity_detector_result(
    visible_state: str,
    detector: str,
    *,
    probe_complete: bool,
) -> bool | None:
    if not probe_complete:
        return None
    match = re.search(
        rf"\b{re.escape(detector)}\b[^.\n]*?\b(on|off)\b",
        visible_state,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).casefold() == "on"


def _conductivity_conditioned_move_action(
    eligible_actions: list[str],
    task_target: str,
    task_description: str,
    conductive: bool,
) -> str | None:
    boxes = _conditional_conductivity_boxes(task_description)
    if boxes is None:
        return None
    target_box = boxes[0] if conductive else boxes[1]
    target_tokens = _task_target_action_tokens(task_target)
    box_tokens = _tokens(target_box)
    matches = [
        action
        for action in eligible_actions
        if action.casefold().startswith("move ")
        and target_tokens <= _tokens(action)
        and box_tokens <= _tokens(action)
    ]
    return min(
        matches,
        key=lambda action: (
            0 if " in inventory" in action.casefold() else 1,
            0 if " in room" in action.casefold() else 1,
            action.casefold(),
        ),
        default=None,
    )


def _is_conditional_conductivity_answer_action(
    action: str,
    task_target: str,
    task_description: str,
) -> bool:
    boxes = _conditional_conductivity_boxes(task_description)
    if boxes is None or not action.casefold().startswith("move "):
        return False
    action_tokens = _tokens(action)
    return (_task_target_action_tokens(task_target) <= action_tokens) and any(
        _tokens(box) <= action_tokens for box in boxes
    )


def _procedure_action_domain(
    eligible_actions: list[str], subgoal: Subgoal
) -> list[str]:
    acquisition_subjects = [
        _tokens(fact.key[0]) - _ACTION_MATCH_STOPWORDS
        for fact in subgoal.expected_facts
        if fact.relation in {"in_inventory_of", "visible_in"}
    ]
    acquisition_subjects = [tokens for tokens in acquisition_subjects if tokens]
    if not acquisition_subjects:
        return eligible_actions
    information_operators = {
        "examine",
        "go",
        "inventory",
        "look",
        "open",
        "read",
    }
    filtered: list[str] = []
    for action in eligible_actions:
        lowered = action.casefold()
        operator = lowered.split(maxsplit=1)[0] if lowered else ""
        action_tokens = _tokens(lowered) - _ACTION_MATCH_STOPWORDS
        if operator in information_operators or any(
            subject <= action_tokens for subject in acquisition_subjects
        ):
            filtered.append(action)
    return filtered


def _goal_directed_action(
    eligible_actions: list[str],
    subgoal: Subgoal,
    observed_facts: list[TemporalFact],
    achieved_keys: set[tuple[str, str, str, bool]],
    entity_locations: dict[str, str] | None = None,
) -> str | None:
    entity_locations = entity_locations or {}
    observed_keys = {fact.key for fact in observed_facts} | achieved_keys
    ranked: list[tuple[int, str]] = []
    for fact in subgoal.expected_facts:
        if fact.key in observed_keys:
            continue
        subject_tokens = _tokens(fact.key[0]) - _ACTION_MATCH_STOPWORDS
        object_tokens = _tokens(fact.key[2]) - _ACTION_MATCH_STOPWORDS
        for action in eligible_actions:
            lowered = action.casefold()
            action_tokens = _tokens(lowered) - _ACTION_MATCH_STOPWORDS
            operator = lowered.split(maxsplit=1)[0] if lowered else ""
            priority: int | None = None
            location_hint = entity_locations.get(fact.key[0])
            navigation_target = _navigation_destination(action)
            if (
                location_hint
                and fact.key[0] != "agent"
                and location_hint != "agent"
                and navigation_target == location_hint
                and operator in {"go", "open"}
            ):
                priority = -1
            elif (
                fact.relation == "in_inventory_of"
                and fact.object.casefold() == "agent"
                and subject_tokens
                and subject_tokens <= action_tokens
                and (
                    operator in {"pick", "take"}
                    or (operator == "move" and "inventory" in _tokens(lowered))
                )
            ):
                priority = 0
            elif (
                fact.relation == "contains"
                and subject_tokens
                and object_tokens
                and subject_tokens <= action_tokens
                and object_tokens <= action_tokens
                and operator in {"move", "pour", "put", "place"}
            ):
                priority = 1
            elif (
                fact.relation == "located_in"
                and fact.subject.casefold() == "agent"
                and object_tokens
                and object_tokens <= action_tokens
                and operator == "go"
            ):
                priority = 2
            elif (
                fact.relation == "has_state"
                and subject_tokens
                and subject_tokens <= action_tokens
                and fact.object.casefold() == "open"
                and operator == "open"
            ):
                priority = 3
            elif (
                fact.relation == "has_state"
                and subject_tokens
                and subject_tokens <= action_tokens
                and fact.object.casefold() in {"on", "activated"}
                and operator in {"activate", "turn"}
            ):
                priority = 4
            if priority is not None:
                ranked.append((priority, action))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1].casefold()))
    return ranked[0][1]


def _update_entity_locations(
    facts: list[TemporalFact], entity_locations: dict[str, str]
) -> None:
    for fact in facts:
        if fact.relation == "visible_in" and fact.key[0] != "agent":
            entity_locations[fact.key[0]] = fact.key[2]
        elif fact.relation == "in_inventory_of":
            entity_locations[fact.key[0]] = "agent"
    for fact in facts:
        if fact.relation == "contains":
            container_location = entity_locations.get(fact.key[0])
            if container_location:
                entity_locations[fact.key[2]] = container_location


def _failed_devices_from_observation(observation: str) -> set[str]:
    lowered = observation.casefold()
    return {
        device
        for device in ("stove", "oven", "blast furnace", "fire pit", "freezer")
        if re.search(
            rf"\b{re.escape(device)}\b[^.\n]*(?:appears|is) broken\b",
            lowered,
        )
    }


def _repair_expected_locations(
    subgoals: list[Subgoal], entity_locations: dict[str, str]
) -> int:
    repair_count = 0
    for subgoal in subgoals:
        repaired: list[FactPattern] = []
        for fact in subgoal.expected_facts:
            verified_location = entity_locations.get(fact.key[0])
            if (
                fact.relation == "visible_in"
                and fact.key[0] != "agent"
                and verified_location
                and verified_location != "agent"
                and verified_location != fact.key[2]
            ):
                repaired.append(
                    FactPattern(
                        subject=fact.subject,
                        relation=fact.relation,
                        object=verified_location,
                        polarity=fact.polarity,
                    )
                )
                repair_count += 1
            else:
                repaired.append(fact)
        subgoal.expected_facts = repaired
    return repair_count


def _state_change_workflow_action(
    eligible_actions: list[str],
    *,
    task_description: str,
    task_target: str | None,
    observed_facts: list[TemporalFact],
    entity_locations: dict[str, str] | None = None,
    unavailable_devices: set[str] | None = None,
) -> str | None:
    entity_locations = entity_locations or {}
    unavailable_devices = unavailable_devices or set()
    if not task_target or not re.search(
        r"\b(?:boil|melt|freeze|change the state of matter)\b",
        task_description,
        flags=re.IGNORECASE,
    ):
        return None
    target_key = FactPattern(
        subject=task_target,
        relation="in_inventory_of",
        object="agent",
    ).key[0]
    fact_keys = {fact.key for fact in observed_facts}
    target_carried = (target_key, "in_inventory_of", "agent", True) in fact_keys
    pot_carried = ("metal pot", "in_inventory_of", "agent", True) in fact_keys
    target_containers = sorted(
        {
            fact.subject
            for fact in observed_facts
            if fact.relation == "contains" and fact.key[2] == target_key
            and _is_portable_vessel(fact.key[0])
        }
    )
    pot_contains_target = "metal pot" in target_containers
    target_tokens = _tokens(target_key)

    if target_carried and not pot_contains_target:
        if not pot_carried:
            open_pot_container = _open_blocking_container_action(
                eligible_actions,
                entity="metal pot",
                observed_facts=observed_facts,
            )
            if open_pot_container:
                return open_pot_container
            if "metal pot" not in entity_locations:
                inspect_closed_container = _open_unknown_tool_container_action(
                    eligible_actions, observed_facts
                )
                if inspect_closed_container:
                    return inspect_closed_container
            pickup_pot = sorted(
                (
                    action
                    for action in eligible_actions
                    if action.casefold().startswith(
                        ("pick up metal pot", "move metal pot")
                    )
                    and (
                        action.casefold().startswith("pick up ")
                        or "inventory" in action.casefold()
                    )
                ),
                key=lambda action: (
                    0 if action.casefold().startswith("pick up ") else 1,
                    action.casefold(),
                ),
            )
            if pickup_pot:
                return pickup_pot[0]
            pot_location = entity_locations.get("metal pot")
            if pot_location and pot_location != "agent":
                navigation = sorted(
                    (
                        action
                        for action in eligible_actions
                        if _navigation_destination(action) == pot_location
                    ),
                    key=str.casefold,
                )
                if navigation:
                    return navigation[0]
        put_in_pot = sorted(
            (
                action
                for action in eligible_actions
                if action.casefold().startswith(("move ", "pour ", "put ", "place "))
                and "metal pot" in action.casefold()
                and target_tokens <= _tokens(action)
                and action.casefold().startswith(
                    (
                        f"move {target_key}",
                        f"pour {target_key}",
                        f"put {target_key}",
                        f"place {target_key}",
                    )
                )
            ),
            key=str.casefold,
        )
        if put_in_pot:
            return put_in_pot[0]

    if not target_containers:
        return None
    active_container = "metal pot" if pot_contains_target else target_containers[0]
    lowered_task = task_description.casefold()
    candidate_devices = (
        ("freezer",)
        if "freeze" in lowered_task
        else ("stove", "blast furnace", "fire pit")
    )
    devices = tuple(
        device for device in candidate_devices if device not in unavailable_devices
    )
    container_carried = (
        active_container,
        "in_inventory_of",
        "agent",
        True,
    ) in fact_keys
    container_on_device = any(
        (device, "contains", active_container, True) in fact_keys
        for device in devices
    )
    if not container_carried and not container_on_device:
        pickup_container = sorted(
            (
                action
                for action in eligible_actions
                if (
                    action.casefold().startswith(f"pick up {active_container}")
                    or (
                        action.casefold().startswith(f"move {active_container}")
                        and "inventory" in action.casefold()
                    )
                )
            ),
            key=lambda action: (
                0 if action.casefold().startswith("pick up ") else 1,
                action.casefold(),
            ),
        )
        if pickup_container:
            return pickup_container[0]
    for device in devices:
        if (device, "contains", active_container, True) in fact_keys:
            activation = next(
                (
                    action
                    for action in eligible_actions
                    if action.casefold() in {f"activate {device}", f"turn on {device}"}
                ),
                None,
            )
            if activation:
                return activation
            waits = [
                action
                for action in eligible_actions
                if action.casefold() in {"wait", "wait1"}
            ]
            if waits:
                return sorted(waits, key=str.casefold)[0]
        move_to_device = sorted(
            (
                action
                for action in eligible_actions
                if action.casefold().startswith(f"move {active_container}")
                and f"to {device}" in action.casefold()
            ),
            key=str.casefold,
        )
        if move_to_device:
            return move_to_device[0]
        device_location = entity_locations.get(device)
        if device_location and device_location != "agent":
            navigation = sorted(
                (
                    action
                    for action in eligible_actions
                    if _navigation_destination(action) == device_location
                ),
                key=lambda action: (
                    0 if action.casefold().startswith("open door to ") else 1,
                    action.casefold(),
                ),
            )
            if navigation:
                return navigation[0]
    return None


def _navigation_destination(action: str) -> str | None:
    lowered = action.strip().casefold()
    if lowered == "go to door":
        return None
    match = re.match(r"open door to (.+)$", lowered)
    if not match:
        match = re.match(r"go to (?:door to )?(.+)$", lowered)
    destination_name = match.group(1).strip() if match else None
    return destination_name if destination_name and destination_name != "door" else None


def _systematic_exploration_action(
    eligible_actions: list[str],
    *,
    visited_locations: set[str],
    current_location: str | None = None,
    destination_visit_counts: dict[str, int] | None = None,
    explored_open_actions: set[tuple[str | None, str]] | None = None,
) -> str | None:
    destination_visit_counts = destination_visit_counts or {}
    explored_open_actions = explored_open_actions or set()
    unopened_routes = sorted(
        (
            action
            for action in eligible_actions
            if action.casefold().startswith("open door to ")
            and _navigation_destination(action) not in visited_locations
            and (current_location, action.casefold()) not in explored_open_actions
        ),
        key=str.casefold,
    )
    if unopened_routes:
        return unopened_routes[0]
    unvisited_moves = sorted(
        (
            action
            for action in eligible_actions
            if action.casefold().startswith("go to ")
            and _navigation_destination(action) is not None
            and _navigation_destination(action) not in visited_locations
        ),
        key=str.casefold,
    )
    if unvisited_moves:
        return unvisited_moves[0]
    unopened_containers = sorted(
        (
            action
            for action in eligible_actions
            if action.casefold().startswith("open ")
            and not action.casefold().startswith("open door to ")
            and (current_location, action.casefold()) not in explored_open_actions
        ),
        key=str.casefold,
    )
    if unopened_containers:
        return unopened_containers[0]
    transit_moves = sorted(
        (
            action
            for action in eligible_actions
            if action.casefold().startswith("go to ")
            and _navigation_destination(action) is not None
            and _navigation_destination(action) != current_location
        ),
        key=lambda action: (
            destination_visit_counts.get(_navigation_destination(action) or "", 0),
            action.casefold(),
        ),
    )
    if transit_moves:
        return transit_moves[0]
    if "look around" in eligible_actions:
        return "look around"
    return None


_SEMANTIC_TARGET_EXCLUSIONS = {
    "agent",
    "air",
    "inventory",
    "orange",
    "sewer",
    "sink",
    "drain",
    "ground",
    "fountain",
    "fire pit",
}
_ANIMAL_MARKERS = {
    "ant",
    "bear",
    "beaver",
    "bird",
    "butterfly",
    "chameleon",
    "chipmunk",
    "dove",
    "dragonfly",
    "elephant",
    "hedgehog",
    "mouse",
    "parrot",
    "toad",
    "tortoise",
    "turtle",
    "wolf",
}
_LIFESPAN_RANK = {
    "giant tortoise": 100,
    "elephant": 90,
    "parrot": 80,
    "brown bear": 70,
    "bear": 68,
    "wolf": 60,
    "beaver": 50,
    "turtle": 45,
    "toad": 30,
    "hedgehog": 20,
    "chameleon": 15,
    "mouse": 10,
    "ant": 5,
    "dragonfly": 3,
}


def _semantic_task_kind(task_description: str) -> str | None:
    lowered = task_description.casefold()
    if "longest life span" in lowered:
        return "longest_lifespan"
    if "find a(n) non-living thing" in lowered:
        return "nonliving"
    if "find a(n) living thing" in lowered:
        return "living"
    if "find a(n) plant" in lowered:
        return "plant"
    return None


def _semantic_destination(task_description: str) -> tuple[str, str] | None:
    match = re.search(
        r"move it to the (.+? box) in the ([^.]+)",
        task_description,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip().casefold(), match.group(2).strip().casefold()


def _is_animal_target(target: str) -> bool:
    lowered = target.casefold()
    tokens = _tokens(lowered)
    return lowered.startswith(("baby ", "egg ")) or bool(tokens & _ANIMAL_MARKERS)


def _semantic_focus_actions(kind: str | None, eligible_actions: list[str]) -> list[str]:
    if kind is None:
        return []
    focus_actions = [
        action for action in eligible_actions if action.casefold().startswith("focus on ")
    ]
    movable_targets = {
        action[len("pick up ") :].casefold()
        for action in eligible_actions
        if action.casefold().startswith("pick up ")
    }
    movable_targets.update(
        match.group(1).casefold()
        for action in eligible_actions
        if (match := re.match(r"move (.+?) to inventory$", action, re.IGNORECASE))
    )
    ranked: list[tuple[int, str]] = []
    for action in focus_actions:
        target = action[len("focus on ") :].casefold()
        if target in _SEMANTIC_TARGET_EXCLUSIONS or target.startswith("door"):
            continue
        is_plant = " plant" in f" {target}" or " tree" in f" {target}"
        is_animal = _is_animal_target(target)
        priority: int | None = None
        if kind == "plant" and is_plant:
            priority = 0
        elif kind == "living" and is_animal:
            priority = 0
        elif kind == "longest_lifespan" and is_animal:
            priority = -max(
                (rank for name, rank in _LIFESPAN_RANK.items() if name in target),
                default=0,
            )
        elif (
            kind == "nonliving"
            and target in movable_targets
            and not is_plant
            and not is_animal
        ):
            preferred = {
                "painting": 0,
                "steel table": 1,
                "bed": 2,
                "counter": 3,
                "bowl": 4,
                "object": 5,
                "picture": 6,
                "book": 7,
                "chair": 8,
                "table": 9,
            }
            priority = preferred.get(target, 20)
        if priority is not None:
            ranked.append((priority, action))
    ranked.sort(key=lambda item: (item[0], item[1].casefold()))
    return [action for _, action in ranked]


def _semantic_acquisition_action(
    eligible_actions: list[str], target: str | None
) -> str | None:
    if not target:
        return None
    exact = {
        f"pick up {target}".casefold(),
        f"move {target} to inventory".casefold(),
    }
    direct = next(
        (action for action in eligible_actions if action.casefold() in exact),
        None,
    )
    if direct is not None:
        return direct
    target_tokens = _tokens(target) - {
        "adult",
        "baby",
        "egg",
        "height",
        "stage",
        "tall",
    }
    ranked: list[tuple[int, str]] = []
    for action in eligible_actions:
        lowered = action.casefold()
        match = re.match(r"pick up (.+)$", lowered)
        if match is None:
            match = re.match(r"move (.+?) to inventory$", lowered)
        if match is None:
            continue
        subject_tokens = _tokens(match.group(1)) - {
            "adult",
            "baby",
            "egg",
            "height",
            "stage",
            "tall",
        }
        overlap = len(target_tokens & subject_tokens)
        if target_tokens and target_tokens <= subject_tokens:
            ranked.append((-overlap, action))
    ranked.sort(key=lambda item: (item[0], item[1].casefold()))
    return ranked[0][1] if ranked else None


def _semantic_move_to_box_action(
    eligible_actions: list[str], target: str | None, box: str | None
) -> str | None:
    if not target or not box:
        return None
    target_tokens = _tokens(target) - _ACTION_MATCH_STOPWORDS
    ranked: list[tuple[int, str]] = []
    for action in eligible_actions:
        lowered = action.casefold()
        if not lowered.startswith(("move ", "place ", "put ", "pour ")):
            continue
        if not lowered.endswith(f"to {box}"):
            continue
        overlap = len(target_tokens & (_tokens(lowered) - _ACTION_MATCH_STOPWORDS))
        if overlap:
            ranked.append((-overlap, action))
    ranked.sort(key=lambda item: (item[0], item[1].casefold()))
    return ranked[0][1] if ranked else None


def _canonicalize_model_payload(content: Any) -> tuple[Any, dict[str, int]]:
    """Normalize only relation forms with one unambiguous canonical orientation."""
    counts: dict[str, int] = {}

    def walk(value: Any) -> Any:
        if isinstance(value, list):
            return [walk(item) for item in value]
        if not isinstance(value, dict):
            return value
        normalized = {key: walk(item) for key, item in value.items()}
        completion_test = str(normalized.get("completion_test", ""))
        if (
            "expected_facts" in normalized
            and "completion_mode" not in normalized
            and re.search(r"\bor\b", completion_test, flags=re.IGNORECASE)
        ):
            normalized["completion_mode"] = "any"
            counts["canonicalized_alternative_completion"] = (
                counts.get("canonicalized_alternative_completion", 0) + 1
            )
        relation = str(normalized.get("relation", "")).strip().casefold()
        subject = str(normalized.get("subject", "")).strip()
        object_ = str(normalized.get("object", "")).strip()
        if relation == "in_inventory_of" and subject.casefold() == "agent" and object_:
            normalized["subject"], normalized["object"] = object_, "agent"
            counts["canonicalized_inventory_orientation"] = (
                counts.get("canonicalized_inventory_orientation", 0) + 1
            )
        if relation == "located_in" and subject.casefold() != "agent" and subject:
            normalized["relation"] = "visible_in"
            counts["canonicalized_object_location_relation"] = (
                counts.get("canonicalized_object_location_relation", 0) + 1
            )
        container_terms = {
            "bowl",
            "box",
            "chair",
            "counter",
            "cup",
            "cupboard",
            "drawer",
            "freezer",
            "fridge",
            "furnace",
            "hive",
            "jar",
            "jug",
            "oven",
            "pit",
            "pot",
            "sink",
            "stove",
            "table",
        }
        subject_tokens = _tokens(subject)
        object_tokens = _tokens(object_)
        if (
            relation == "contains"
            and not (subject_tokens & container_terms)
            and object_tokens & container_terms
        ):
            normalized["subject"], normalized["object"] = object_, subject
            counts["canonicalized_contains_orientation"] = (
                counts.get("canonicalized_contains_orientation", 0) + 1
            )
        if relation == "has_state" and object_.casefold() in {"turned on", "turned off"}:
            normalized["object"] = object_.casefold().removeprefix("turned ")
            counts["canonicalized_state_label"] = (
                counts.get("canonicalized_state_label", 0) + 1
            )
        state_aliases = {
            "combusting": "burning",
            "combusted": "burning",
            "frozen": "solid",
            "melted": "liquid",
        }
        if relation == "has_state" and object_.casefold() in state_aliases:
            normalized["object"] = state_aliases[object_.casefold()]
            counts["canonicalized_state_label"] = (
                counts.get("canonicalized_state_label", 0) + 1
            )
        if (
            "expected_facts" in normalized
            and str(normalized.get("completion_test", "")).casefold() == "any"
            and normalized.get("completion_mode") != "any"
        ):
            normalized["completion_mode"] = "any"
            counts["canonicalized_alternative_completion"] = (
                counts.get("canonicalized_alternative_completion", 0) + 1
            )
        return normalized

    return walk(content), counts


class ENSRMemory:
    def __init__(self, fragments: list[dict[str, Any]]) -> None:
        if not fragments:
            raise ValueError("ENSR memory requires at least one train fragment")
        self.fragments = list(fragments)
        self.by_id = {str(fragment["fragment_id"]): fragment for fragment in fragments}
        if len(self.by_id) != len(self.fragments):
            raise ValueError("ENSR fragment IDs must be unique")
        self.text = {
            fragment_id: json.dumps(fragment, ensure_ascii=False, sort_keys=True)
            for fragment_id, fragment in self.by_id.items()
        }
        self.navigation_graph: dict[str, set[str]] = {}
        for fragment in self.fragments:
            if str(fragment.get("operator", "")).casefold() != "go":
                continue
            sources = [
                str(fact["object"]).casefold()
                for fact in fragment.get("preconditions", [])
                if fact.get("subject") == "agent"
                and fact.get("relation") == "located_in"
                and bool(fact.get("polarity"))
            ]
            destinations = [
                str(fact["object"]).casefold()
                for fact in fragment.get("effects", [])
                if fact.get("subject") == "agent"
                and fact.get("relation") == "located_in"
                and bool(fact.get("polarity"))
            ]
            for source in sources:
                for destination in destinations:
                    if source == destination or "<" in source or "<" in destination:
                        continue
                    self.navigation_graph.setdefault(source, set()).add(destination)
                    self.navigation_graph.setdefault(destination, set()).add(source)

    def route_action(
        self,
        eligible_actions: list[str],
        *,
        current_location: str | None,
        target_location: str | None,
    ) -> str | None:
        if not current_location or not target_location:
            return None
        source = current_location.casefold()
        target = target_location.casefold()
        if source == target:
            return None
        frontier = [source]
        parent: dict[str, str | None] = {source: None}
        for location in frontier:
            if target in parent:
                break
            for neighbor in sorted(self.navigation_graph.get(location, set())):
                if neighbor not in parent:
                    parent[neighbor] = location
                    frontier.append(neighbor)
        if target not in parent:
            return None
        next_hop = target
        while parent[next_hop] != source:
            predecessor = parent[next_hop]
            if predecessor is None:
                return None
            next_hop = predecessor
        exact_preferences = [
            f"go to {next_hop}",
            f"open door to {next_hop}",
            f"open {next_hop} door",
            f"go to door to {next_hop}",
            f"go to {next_hop} door",
        ]
        canonical = {action.casefold(): action for action in eligible_actions}
        for preference in exact_preferences:
            if preference in canonical:
                return canonical[preference]
        open_doors = [
            action
            for action in eligible_actions
            if action.casefold() in {"open door", "go to door"}
        ]
        return sorted(open_doors, key=str.casefold)[0] if open_doors else None

    @classmethod
    def load(cls, path: Path) -> "ENSRMemory":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("reporting_boundary") != "train_split_derived_memory_not_test_result":
            raise ValueError("ENSR memory must be derived only from train trajectories")
        return cls(list(payload["fragments"]))

    def search(
        self,
        query: str,
        *,
        task_id: str,
        limit: int,
    ) -> list[str]:
        query_tokens = _tokens(query)
        topic = task_id.split("-", maxsplit=1)[0]
        scored: list[tuple[float, str]] = []
        for fragment_id, fragment in self.by_id.items():
            fragment_tokens = _tokens(self.text[fragment_id])
            union = len(query_tokens | fragment_tokens) or 1
            lexical = len(query_tokens & fragment_tokens) / union
            fragment_task = str(fragment["task_id"])
            structural = (
                1.0
                if fragment_task == task_id
                else 0.25
                if fragment_task.split("-", maxsplit=1)[0] == topic
                else 0.0
            )
            scored.append((-(lexical + structural), fragment_id))
        scored.sort()
        return [fragment_id for _, fragment_id in scored[:limit]]

    def prompt_view(self, fragment_ids: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "fragment_id": fragment_id,
                "task_id": self.by_id[fragment_id]["task_id"],
                "action_template": self.by_id[fragment_id]["action_template"],
                "preconditions": self.by_id[fragment_id]["preconditions"],
                "effects": self.by_id[fragment_id]["effects"],
                "outcome": self.by_id[fragment_id]["outcome"],
            }
            for fragment_id in fragment_ids
        ]

    def procedure_actions(self, fragment_ids: list[str]) -> list[str]:
        return [str(self.by_id[item]["action_template"]) for item in fragment_ids]

    def procedure_context(
        self, task_id: str, *, max_actions: int = 64
    ) -> dict[str, Any]:
        by_variation: dict[int, list[dict[str, Any]]] = {}
        for fragment in self.fragments:
            if str(fragment["task_id"]) != task_id:
                continue
            by_variation.setdefault(int(fragment["variation"]), []).append(fragment)
        if not by_variation:
            return {"task_id": task_id, "variation": None, "actions": []}
        ranked: list[tuple[int, int, int, list[dict[str, Any]]]] = []
        for variation, fragments in by_variation.items():
            ordered = sorted(fragments, key=lambda item: int(item["step_index"]))
            completed = any(bool(item["outcome"]["completed"]) for item in ordered)
            ranked.append((0 if completed else 1, len(ordered), variation, ordered))
        _, _, variation, selected = min(ranked, key=lambda item: item[:3])
        return {
            "task_id": task_id,
            "variation": variation,
            "actions": [
                str(fragment["action_template"])
                for fragment in selected[:max_actions]
            ],
        }


class ENSRHarness:
    def __init__(
        self,
        memory: ENSRMemory,
        adapter: LLMAdapter,
        *,
        experiment_id: str = "E12",
        enable_procedure_guidance: bool = True,
        enable_symbolic_control: bool = True,
        enable_adaptive_slow_path: bool = False,
        force_llm_after_local_replan: bool = True,
        max_continuation_plans_per_episode: int = 2,
        max_continuation_subgoals: int = 4,
        enable_controlled_residual_repair: bool = False,
        max_controlled_repair_requests: int = 1,
        max_controlled_repair_actions: int = 3,
        enable_verified_graph_repair: bool = False,
        max_verified_graph_rank_requests: int = 1,
        max_verified_graph_actions: int = 3,
        max_verified_graph_candidates: int = 8,
        enable_semantic_kg_control: bool = False,
        enable_hierarchical_subgoals: bool = True,
        enable_symbolic_transition_verifier: bool = True,
        enable_obligation_conditioned_retrieval: bool = True,
        enable_evidence_debt_scheduler: bool = False,
        evidence_debt_retrieval_threshold: float = 0.58,
        evidence_debt_anchor_first_subgoal: bool = False,
        fragment_memory_policy: Literal["replace", "accumulate"] = "replace",
    ) -> None:
        if fragment_memory_policy not in {"replace", "accumulate"}:
            raise ValueError("fragment_memory_policy must be replace or accumulate")
        self.memory = memory
        self.adapter = adapter
        self.experiment_id = experiment_id
        self.enable_procedure_guidance = enable_procedure_guidance
        self.enable_symbolic_control = enable_symbolic_control
        self.enable_adaptive_slow_path = enable_adaptive_slow_path
        self.force_llm_after_local_replan = force_llm_after_local_replan
        if max_continuation_plans_per_episode < 0:
            raise ValueError("max_continuation_plans_per_episode must be non-negative")
        if max_continuation_subgoals < 1:
            raise ValueError("max_continuation_subgoals must be positive")
        self.max_continuation_plans_per_episode = max_continuation_plans_per_episode
        self.max_continuation_subgoals = max_continuation_subgoals
        if max_controlled_repair_requests not in {0, 1}:
            raise ValueError("max_controlled_repair_requests must be zero or one")
        if max_controlled_repair_actions < 1:
            raise ValueError("max_controlled_repair_actions must be positive")
        self.enable_controlled_residual_repair = enable_controlled_residual_repair
        self.max_controlled_repair_requests = max_controlled_repair_requests
        self.max_controlled_repair_actions = max_controlled_repair_actions
        if enable_controlled_residual_repair and enable_verified_graph_repair:
            raise ValueError("D3 and D4 repair modes are mutually exclusive")
        if max_verified_graph_rank_requests not in {0, 1}:
            raise ValueError("max_verified_graph_rank_requests must be zero or one")
        if max_verified_graph_actions < 1:
            raise ValueError("max_verified_graph_actions must be positive")
        if max_verified_graph_candidates < 1:
            raise ValueError("max_verified_graph_candidates must be positive")
        self.enable_verified_graph_repair = enable_verified_graph_repair
        self.max_verified_graph_rank_requests = max_verified_graph_rank_requests
        self.max_verified_graph_actions = max_verified_graph_actions
        self.max_verified_graph_candidates = max_verified_graph_candidates
        self.enable_semantic_kg_control = enable_semantic_kg_control
        self.enable_hierarchical_subgoals = enable_hierarchical_subgoals
        self.enable_symbolic_transition_verifier = (
            enable_symbolic_transition_verifier
        )
        self.enable_obligation_conditioned_retrieval = (
            enable_obligation_conditioned_retrieval
        )
        if not 0.0 <= evidence_debt_retrieval_threshold <= 1.0:
            raise ValueError(
                "evidence_debt_retrieval_threshold must be between zero and one"
            )
        self.enable_evidence_debt_scheduler = enable_evidence_debt_scheduler
        self.evidence_debt_retrieval_threshold = evidence_debt_retrieval_threshold
        self.evidence_debt_anchor_first_subgoal = (
            evidence_debt_anchor_first_subgoal
        )
        self.fragment_memory_policy = fragment_memory_policy

    def _apply_hierarchy_policy(
        self, plan: HierarchicalPlan, *, task_description: str
    ) -> HierarchicalPlan:
        if self.enable_hierarchical_subgoals or len(plan.subgoals) <= 1:
            return plan
        terminal = plan.subgoals[-1].model_copy(
            update={
                "subgoal_id": "sg-task",
                "description": f"Complete the task: {task_description}",
                "retry_limit": max(item.retry_limit for item in plan.subgoals),
            }
        )
        return plan.model_copy(update={"subgoals": [terminal]})

    def _plan_request(
        self,
        *,
        task_description: str,
        observation: str,
        facts: list[TemporalFact],
        fragment_ids: list[str],
        procedure: dict[str, Any],
        compact_retry: bool = False,
    ) -> LLMRequest:
        return LLMRequest(
            purpose="plan_generation",
            system_instruction=(
                "Return compact valid JSON for a short ordered ScienceWorld plan. Use two "
                "to five subgoals with IDs sg1, sg2, and so on. Each subgoal must "
                "end in one to four facts that can be checked directly from the environment. "
                "Use only located_in, visible_in, has_state, in_inventory_of, or contains "
                "relations. "
                "Canonical facts are OBJECT in_inventory_of agent; agent located_in ROOM; "
                "OBJECT visible_in ROOM; CONTAINER contains OBJECT; and OBJECT has_state STATE. "
                "Expected facts are conjunctive when completion_mode is all. Use "
                "completion_mode any only for explicit alternative outcomes. Use short entity "
                "names "
                "from observed_facts, never composite phrases such as 'stove, which is on'. "
                "Do not claim hidden facts or copy source-specific entity names unless they "
                "occur in the current task or observation. The complete train procedure is a "
                "structural example: preserve useful action order and bind placeholders to the "
                "current target. Keep every string concise."
                + (
                    " This is a retry after malformed output: emit only the schema fields, "
                    "with no commentary and no unusual subgoal IDs."
                    if compact_retry
                    else ""
                )
            ),
            payload={
                "task_description": task_description,
                "current_observation": observation,
                "observed_facts": [_fact_prompt(fact) for fact in facts],
                "train_memory": self.memory.prompt_view(fragment_ids),
                "one_complete_train_procedure": procedure,
            },
            response_schema=HierarchicalPlan.model_json_schema(),
        )

    def _action_request(
        self,
        *,
        task_description: str,
        subgoal: Subgoal,
        observation: str,
        info: dict[str, Any],
        facts: list[TemporalFact],
        fragment_ids: list[str],
        procedure: dict[str, Any],
        candidates: list[str],
        recent_steps: list[ENSRStep],
        obligations: list[EvidenceObligation],
        compact_retry: bool = False,
    ) -> LLMRequest:
        fact_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["subject", "relation", "object", "polarity"],
            "properties": {
                "subject": {"type": "string"},
                "relation": {
                    "type": "string",
                    "enum": [
                        "located_in",
                        "visible_in",
                        "has_state",
                        "in_inventory_of",
                        "contains",
                    ],
                },
                "object": {"type": "string"},
                "polarity": {"type": "boolean", "const": True},
            },
        }
        payload: dict[str, Any] = {
            "task_description": task_description,
            "current_subgoal": subgoal.model_dump(mode="json"),
            "current": {
                "observation": observation,
                "inventory": info.get("inv", ""),
                "score": info.get("score", 0),
                "observed_facts": [_fact_prompt(fact) for fact in facts],
            },
            "candidate_actions": candidates,
        }
        if not compact_retry:
            payload.update(
                {
                    "active_train_memory": self.memory.prompt_view(fragment_ids),
                    "one_complete_train_procedure": procedure,
                    "recent_steps": [
                        step.model_dump(mode="json") for step in recent_steps[-5:]
                    ],
                    "recent_obligations": [
                        obligation.model_dump(mode="json")
                        for obligation in obligations[-3:]
                    ],
                }
            )
        return LLMRequest(
            purpose="environment_action",
            system_instruction=(
                "Choose exactly one action from candidate_actions to advance the current "
                "subgoal. expected_fact must describe only the immediate observable effect of "
                "that action using the supported relations; use null for pure observation when "
                "no state change is expected. Canonical facts are OBJECT in_inventory_of agent; "
                "agent located_in ROOM; OBJECT visible_in ROOM; CONTAINER contains OBJECT; and "
                "OBJECT has_state STATE. "
                "Never invent an action outside the enum. A wrong focus action can irreversibly "
                "fail the task, so focus only on the target explicitly named by task_description. "
                "Use the complete train procedure as an ordered structural guide, but choose only "
                "an action present in candidate_actions and bind placeholders to current entities."
                + (
                    " This is a retry after malformed output. Return only the three schema fields; "
                    "keep expected_progress under twelve words."
                    if compact_retry
                    else ""
                )
            ),
            payload=payload,
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "expected_fact", "expected_progress"],
                "properties": {
                    "action": {"type": "string", "enum": candidates},
                    "expected_fact": {"anyOf": [fact_schema, {"type": "null"}]},
                    "expected_progress": {"type": "string", "maxLength": 160},
                },
            },
        )

    def _replan_request(
        self,
        *,
        task_description: str,
        current_subgoal: Subgoal,
        observation: str,
        facts: list[TemporalFact],
        obligation: EvidenceObligation,
        fragment_ids: list[str],
        procedure: dict[str, Any],
    ) -> LLMRequest:
        return LLMRequest(
            purpose="plan_generation",
            system_instruction=(
                "Repair only the failed current subgoal using verified observations and train "
                "memory. Keep the same subgoal_id. Use one to four directly checkable facts and "
                "only located_in, visible_in, has_state, in_inventory_of, or contains relations. "
                "Canonical "
                "facts are OBJECT in_inventory_of agent; agent located_in ROOM; CONTAINER "
                "contains OBJECT; OBJECT visible_in ROOM; and OBJECT has_state STATE. Preserve "
                "useful order from the complete train procedure while binding its placeholders "
                "to current entities."
            ),
            payload={
                "task_description": task_description,
                "failed_subgoal": current_subgoal.model_dump(mode="json"),
                "failure": obligation.model_dump(mode="json"),
                "current_observation": observation,
                "observed_facts": [_fact_prompt(fact) for fact in facts],
                "replacement_train_memory": self.memory.prompt_view(fragment_ids),
                "one_complete_train_procedure": procedure,
            },
            response_schema=LocalReplan.model_json_schema(),
        )

    def _continuation_request(
        self,
        *,
        task_description: str,
        observation: str,
        facts: list[TemporalFact],
        previous_plan: HierarchicalPlan,
        completed_subgoal_ids: list[str],
        recent_steps: list[ENSRStep],
        valid_actions: list[str],
        fragment_ids: list[str],
        procedure: dict[str, Any],
        trigger: str,
    ) -> LLMRequest:
        return LLMRequest(
            purpose="plan_generation",
            system_instruction=(
                "The previous ScienceWorld plan ended or became blocked, but the environment "
                "has not completed the task. Return one to "
                f"{self.max_continuation_subgoals} new subgoals for only the "
                "remaining work. Use the temporal knowledge-graph facts and current legal "
                "actions as the source of truth. Do not repeat completed subgoals or recent "
                "zero-progress actions. Each subgoal must end in one to four directly "
                "observable facts using only located_in, visible_in, has_state, "
                "in_inventory_of, or contains. Preserve useful train-procedure order while "
                "binding every entity to the current task and observation. Start at the "
                "first unmet checkpoint and omit any checkpoint already supported by the "
                "observed facts."
            ),
            payload={
                "trigger": trigger,
                "task_description": task_description,
                "current_observation": observation,
                "observed_facts": [_fact_prompt(fact) for fact in facts],
                "previous_plan": previous_plan.model_dump(mode="json"),
                "completed_subgoal_ids": completed_subgoal_ids,
                "recent_steps": [
                    step.model_dump(mode="json") for step in recent_steps[-8:]
                ],
                "current_legal_actions": valid_actions[:50],
                "train_memory": self.memory.prompt_view(fragment_ids),
                "one_complete_train_procedure": procedure,
            },
            response_schema=HierarchicalPlan.model_json_schema(),
        )

    def _residual_repair_request(
        self,
        *,
        task_description: str,
        observation: str,
        info: dict[str, Any],
        facts: list[TemporalFact],
        valid_actions: list[str],
        recent_steps: list[ENSRStep],
        obligations: list[EvidenceObligation],
        trigger: str,
    ) -> LLMRequest:
        schema = ResidualRepairProposal.model_json_schema()
        schema["properties"]["next_action"] = {
            "type": "string",
            "enum": valid_actions,
        }
        return LLMRequest(
            purpose="residual_repair",
            system_instruction=(
                "Propose exactly one bounded repair for the unfinished ScienceWorld task. "
                "Return one residual subgoal and one immediately legal next action. The "
                "subgoal must contain only currently unsatisfied, directly observable facts "
                "using located_in, visible_in, has_state, in_inventory_of, or contains. "
                "Use only entities grounded in the task, observation, inventory, knowledge "
                "graph, or legal actions. Never repeat a recent zero-progress action. Never "
                "choose an unsafe focus target or make an irreversible answer before its "
                "evidence is observed. The repair receives at most three environment actions."
            ),
            payload={
                "trigger": trigger,
                "task_description": task_description,
                "current": {
                    "observation": observation,
                    "look": info.get("look", ""),
                    "inventory": info.get("inv", ""),
                    "score": info.get("score", 0),
                    "observed_facts": [_fact_prompt(fact) for fact in facts],
                },
                "legal_actions": valid_actions,
                "recent_steps": [
                    step.model_dump(mode="json") for step in recent_steps[-8:]
                ],
                "recent_obligations": [
                    obligation.model_dump(mode="json")
                    for obligation in obligations[-5:]
                ],
            },
            response_schema=schema,
        )

    def _candidate_ranking_request(
        self,
        *,
        task_description: str,
        trigger: str,
        residual_facts: tuple[tuple[str, str, str, bool], ...],
        candidates: list[VerifiedRepairCandidate],
    ) -> LLMRequest:
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        schema = CandidateSelection.model_json_schema()
        schema["properties"]["candidate_id"] = {
            "type": "string",
            "enum": candidate_ids,
        }
        return LLMRequest(
            purpose="candidate_ranking",
            system_instruction=(
                "Select exactly one candidate_id from the verified finite set. "
                "Do not generate an action, entity, fact, plan, or new candidate. Prefer "
                "the shortest path that satisfies the most residual facts and is most "
                "directly relevant to the task."
            ),
            payload={
                "trigger": trigger,
                "task_description": task_description,
                "residual_facts": [list(item) for item in residual_facts],
                "verified_candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "actions": list(candidate.actions),
                        "satisfied_residual_facts": [
                            list(item)
                            for item in candidate.satisfied_residual_facts
                        ],
                        "train_evidence_count": candidate.train_evidence_count,
                    }
                    for candidate in candidates
                ],
            },
            response_schema=schema,
        )

    def run_episode(
        self,
        env: ENSREnvironment,
        *,
        task_id: str,
        task_name: str,
        split: Literal["dev", "test"],
        variation: int,
        seed: int,
        budget: ENSRBudget,
        simplifications: str = "",
        action_replay_prefix: list[dict[str, Any]] | None = None,
    ) -> ENSREpisodeResult:
        started = time.perf_counter()
        model_calls = 0
        planner_calls = 0
        retrieval_calls = 0
        invalid_decisions = 0
        input_tokens = 0
        output_tokens = 0
        resolved_model_ids: set[str] = set()
        system_fingerprints: set[str] = set()
        normalizations: dict[str, int] = {}
        trajectory: list[ENSRStep] = []
        retrieval_events: list[ENSRRetrievalEvent] = []
        evidence_schedule_events: list[ENSREvidenceScheduleEvent] = []
        obligations: list[EvidenceObligation] = []
        evidence_debt_ledger = EvidenceDebtLedger(
            retrieval_threshold=self.evidence_debt_retrieval_threshold,
            anchor_first_debt_per_subgoal=(
                self.evidence_debt_anchor_first_subgoal
            ),
        )
        evidence_debt_resolved_count = 0
        fact_history: list[TemporalFact] = []
        completed_subgoal_ids: list[str] = []
        rejected_by_state: dict[str, set[str]] = {}
        subgoal_attempts: dict[str, int] = {}
        subgoal_evidence: dict[str, set[tuple[str, str, str, bool]]] = {}
        plan: HierarchicalPlan | None = None
        adaptive_recovery_plans: list[HierarchicalPlan] = []
        active_fragment_ids: list[str] = []
        completed = False
        final_score = 0
        status: ENSRStatus = "environment_budget_exhausted"
        error_type: str | None = None
        error: str | None = None
        verified_transition_count = 0
        local_replan_count = 0
        local_replan_recovery_count = 0
        adaptive_continuation_plan_count = 0
        adaptive_continuation_recovery_count = 0
        adaptive_slow_action_count = 0
        adaptive_continuation_satisfied_subgoal_drop_count = 0
        adaptive_continuation_truncated_subgoal_count = 0
        controlled_repair_request_count = 0
        controlled_repair_accepted_count = 0
        controlled_repair_rejected_count = 0
        controlled_repair_parse_failure_count = 0
        controlled_repair_unsafe_rejection_count = 0
        controlled_repair_grounding_rejection_count = 0
        controlled_repair_satisfied_rejection_count = 0
        controlled_repair_action_count = 0
        controlled_repair_verified_progress_count = 0
        controlled_repair_recovery_count = 0
        controlled_repair_next_action: str | None = None
        controlled_repair_active = False
        controlled_repair_progress_verified = False
        controlled_repair_start_score = 0
        verified_graph_search_count = 0
        verified_graph_candidate_count = 0
        verified_graph_rank_request_count = 0
        verified_graph_parse_failure_count = 0
        verified_graph_selection_rejection_count = 0
        verified_graph_action_count = 0
        verified_graph_verified_progress_count = 0
        verified_graph_negative_edge_count = 0
        verified_graph_next_action: str | None = None
        verified_graph_active = False
        verified_graph_progress_verified = False
        verified_graph_start_score = 0
        verified_graph_residual_facts: tuple[
            tuple[str, str, str, bool], ...
        ] = ()
        verified_graph_negative_edges: set[tuple[str, str]] = set()
        replay_prefix = list(action_replay_prefix or [])
        action_replay_cursor = 0
        action_replay_override_count = 0
        action_replay_mismatch_count = 0
        contradiction_rejection_count = 0
        planner_parse_failures = 0
        deterministic_fallback_plan_count = 0
        action_parse_failures = 0
        procedure_guided_action_count = 0
        symbolic_goal_action_count = 0
        entity_memory_routing_action_count = 0
        temporal_wait_action_count = 0
        exploration_action_count = 0
        focus_stage_advance_count = 0
        measurement_poll_action_count = 0
        measurement_conditioned_focus_count = 0
        conductivity_probe_action_count = 0
        conductivity_probe_wait_count = 0
        conductivity_conditioned_move_count = 0
        procedure_cursor = 0
        focused_target = False
        focus_stage_index = 0
        visited_locations: set[str] = set()
        current_location: str | None = None
        seen_state_hashes: set[str] = set()
        destination_visit_counts: dict[str, int] = {}
        explored_open_actions: set[tuple[str | None, str]] = set()
        entity_locations: dict[str, str] = {}
        unavailable_devices: set[str] = set()
        pending_replan_recovery = False
        pending_continuation_recovery = False
        model_attempted = False
        handled_obligation_signatures: set[tuple[Any, ...]] = set()
        last_measured_temperature: float | None = None
        transition_temperature: float | None = None
        measurement_workflow_active = False
        conductivity_components: tuple[str, str, str, str] | None = None
        conductivity_connection_index = 0
        conductivity_result: bool | None = None
        semantic_kind: str | None = None
        semantic_destination_box: str | None = None
        semantic_destination_room: str | None = None
        semantic_target_entity: str | None = None
        semantic_target_acquired = False
        knowledge_graph_action_count = 0
        semantic_focus_action_count = 0
        semantic_acquisition_action_count = 0
        semantic_delivery_action_count = 0

        def call_model(request: LLMRequest, model_type):
            nonlocal model_attempted, model_calls, input_tokens, output_tokens
            model_attempted = True
            model_calls += 1
            try:
                response = self.adapter.complete(request)
            except LLMAdapterError as exc:
                model_calls += max(0, exc.provider_calls - 1)
                input_tokens += exc.input_tokens
                output_tokens += exc.output_tokens
                raise
            model_calls += max(0, response.provider_calls - 1)
            input_tokens += response.input_tokens or 0
            output_tokens += response.output_tokens or 0
            resolved_model_ids.add(response.model_id)
            if response.system_fingerprint:
                system_fingerprints.add(response.system_fingerprint)
            if response.normalization_applied:
                normalizations[response.normalization_applied] = (
                    normalizations.get(response.normalization_applied, 0) + 1
                )
            content, canonicalizations = _canonicalize_model_payload(response.content)
            for name, count in canonicalizations.items():
                normalizations[name] = normalizations.get(name, 0) + count
            return model_type.model_validate(content)

        try:
            env.load(
                task_name,
                variationIdx=variation,
                simplificationStr=simplifications,
                generateGoldPath=False,
            )
            observation, info = env.reset()
            task_description = str(info.get("taskDesc", task_name))
            semantic_kind = (
                _semantic_task_kind(task_description)
                if self.enable_semantic_kg_control
                else None
            )
            semantic_destination_value = _semantic_destination(task_description)
            if semantic_destination_value is not None:
                semantic_destination_box, semantic_destination_room = (
                    semantic_destination_value
                )
            explicit_task_target = _explicit_task_target(task_description)
            procedure_task_target = _procedure_task_target(task_description)
            focus_target_stages = _focus_target_stages(
                task_description, explicit_task_target=explicit_task_target
            )
            procedure_focus_object = (
                focus_target_stages[0][0]
                if focus_target_stages and len(focus_target_stages[0]) == 1
                else procedure_task_target
            )
            valid_actions = list(info.get("valid", []))
            final_score = int(info.get("score", 0))
            observed_facts = parse_observation_facts(_state_text(info), step=0)
            _update_entity_locations(observed_facts, entity_locations)
            visited_locations.update(
                fact.object
                for fact in observed_facts
                if fact.subject == "agent"
                and fact.relation == "located_in"
                and fact.object != "door"
            )
            current_location = next(
                (
                    fact.object
                    for fact in observed_facts
                    if fact.subject == "agent"
                    and fact.relation == "located_in"
                    and fact.object != "door"
                ),
                None,
            )
            seen_state_hashes.add(observation_sha256(_state_text(info)))
            fact_history.extend(observed_facts)
            active_fragment_ids = self.memory.search(
                task_description,
                task_id=task_id,
                limit=budget.initial_retrieval_k,
            )
            retrieval_calls = 1
            train_procedure = self.memory.procedure_context(task_id)
            retrieval_events.append(
                ENSRRetrievalEvent(
                    call_index=1,
                    trigger="initial",
                    query=task_description,
                    returned_fragment_ids=active_fragment_ids,
                    new_fragment_ids=active_fragment_ids,
                    replaced_fragment_ids=[],
                )
            )
            for plan_attempt in range(min(2, budget.max_planner_calls)):
                if model_calls >= budget.max_model_calls:
                    break
                planner_calls += 1
                try:
                    plan = call_model(
                        self._plan_request(
                            task_description=task_description,
                            observation=observation,
                            facts=observed_facts,
                            fragment_ids=active_fragment_ids,
                            procedure=train_procedure,
                            compact_retry=plan_attempt > 0,
                        ),
                        HierarchicalPlan,
                    )
                    plan = self._apply_hierarchy_policy(
                        plan, task_description=task_description
                    )
                    break
                except Exception:  # noqa: BLE001 - retry malformed plans once
                    planner_parse_failures += 1
            if plan is None:
                plan = HierarchicalPlan(
                    task_summary=(
                        "Deterministic fallback after invalid structured planning output."
                    ),
                    subgoals=[
                        PlannerSubgoal(
                            subgoal_id="fallback-complete-task",
                            description=(
                                "Use grounded train procedures, verified facts, and exact "
                                "environment actions to complete the task."
                            ),
                            expected_facts=[
                                PlannerFact(
                                    subject="task",
                                    relation="has_state",
                                    object="complete",
                                    polarity=True,
                                )
                            ],
                            completion_test="The environment reports terminal completion.",
                            completion_mode="all",
                            retry_limit=4,
                        )
                    ],
                )
                deterministic_fallback_plan_count += 1
                normalizations["deterministic_fallback_plan"] = 1
            subgoals = [subgoal.as_subgoal() for subgoal in plan.subgoals]
            location_repair_count = _repair_expected_locations(
                subgoals, entity_locations
            )
            if location_repair_count:
                normalizations["repaired_expected_location_from_verified_observation"] = (
                    location_repair_count
                )
            subgoal_index = 0

            def install_adaptive_continuation(trigger: str) -> bool:
                nonlocal adaptive_continuation_plan_count
                nonlocal pending_continuation_recovery
                nonlocal planner_calls, planner_parse_failures
                nonlocal subgoal_index, subgoals
                nonlocal adaptive_continuation_satisfied_subgoal_drop_count
                nonlocal adaptive_continuation_truncated_subgoal_count
                if (
                    not self.enable_adaptive_slow_path
                    or adaptive_continuation_plan_count
                    >= self.max_continuation_plans_per_episode
                    or planner_calls >= budget.max_planner_calls
                    or model_calls >= budget.max_model_calls
                ):
                    return False
                planner_calls += 1
                previous_plan = (
                    adaptive_recovery_plans[-1]
                    if adaptive_recovery_plans
                    else plan
                )
                try:
                    continuation = call_model(
                        self._continuation_request(
                            task_description=task_description,
                            observation=observation,
                            facts=observed_facts,
                            previous_plan=previous_plan,
                            completed_subgoal_ids=completed_subgoal_ids,
                            recent_steps=trajectory,
                            valid_actions=valid_actions,
                            fragment_ids=active_fragment_ids,
                            procedure=train_procedure,
                            trigger=trigger,
                        ),
                        HierarchicalPlan,
                    )
                    continuation = self._apply_hierarchy_policy(
                        continuation, task_description=task_description
                    )
                except Exception:  # noqa: BLE001 - failed recovery remains auditable
                    planner_parse_failures += 1
                    return False
                continuation_number = adaptive_continuation_plan_count + 1
                residual_planner_subgoals = []
                for planner_subgoal in continuation.subgoals:
                    candidate = planner_subgoal.as_subgoal()
                    if _facts_satisfied(
                        candidate.expected_facts,
                        observed_facts,
                        candidate.completion_mode,
                        set(),
                    ):
                        adaptive_continuation_satisfied_subgoal_drop_count += 1
                        continue
                    residual_planner_subgoals.append(planner_subgoal)
                if not residual_planner_subgoals:
                    return False
                if len(residual_planner_subgoals) > self.max_continuation_subgoals:
                    adaptive_continuation_truncated_subgoal_count += (
                        len(residual_planner_subgoals)
                        - self.max_continuation_subgoals
                    )
                    residual_planner_subgoals = residual_planner_subgoals[
                        : self.max_continuation_subgoals
                    ]
                replacement_subgoals = []
                for index, planner_subgoal in enumerate(
                    residual_planner_subgoals, start=1
                ):
                    replacement = planner_subgoal.as_subgoal()
                    replacement.subgoal_id = f"adaptive-{continuation_number}-{index}"
                    replacement_subgoals.append(replacement)
                _repair_expected_locations(replacement_subgoals, entity_locations)
                adaptive_recovery_plans.append(continuation)
                adaptive_continuation_plan_count = continuation_number
                normalizations["adaptive_continuation_plan"] = (
                    normalizations.get("adaptive_continuation_plan", 0) + 1
                )
                subgoals = replacement_subgoals
                subgoal_index = 0
                pending_continuation_recovery = True
                return True

            def verified_graph_candidates(
                *,
                repair_actions: list[str],
                residual_facts: tuple[tuple[str, str, str, bool], ...],
            ) -> list[VerifiedRepairCandidate]:
                nonlocal verified_graph_search_count
                nonlocal verified_graph_candidate_count
                observed_keys = {fact.key for fact in observed_facts}
                transitions = infer_current_action_transitions(repair_actions)
                legal_by_key = {
                    action.strip().casefold(): action for action in repair_actions
                }
                for action_key, action in legal_by_key.items():
                    matched = [
                        fragment
                        for fragment in self.memory.fragments
                        if str(fragment.get("task_id")) == task_id
                        and str(fragment.get("action_template", ""))
                        .strip()
                        .casefold()
                        == action_key
                    ]
                    if not matched:
                        continue
                    effect_counts: dict[
                        tuple[str, str, str, bool], int
                    ] = {}
                    for fragment in matched:
                        for raw_fact in fragment.get("effects", []):
                            key = fact_key(
                                str(raw_fact.get("subject", "")),
                                str(raw_fact.get("relation", "")),
                                str(raw_fact.get("object", "")),
                                bool(raw_fact.get("polarity", True)),
                            )
                            effect_counts[key] = effect_counts.get(key, 0) + 1
                    stable_effects = tuple(
                        sorted(
                            key
                            for key, count in effect_counts.items()
                            if count == len(matched) and key[3]
                        )
                    )
                    if stable_effects:
                        transitions.append(
                            ActionTransition(
                                action=action,
                                preconditions=(),
                                add_effects=stable_effects,
                                currently_legal=True,
                                provenance="train_verified_transition",
                                evidence_count=len(matched),
                            )
                        )
                verified_graph_search_count += 1
                candidates = search_verified_transition_graph(
                    observed_facts=observed_keys,
                    residual_facts=residual_facts,
                    transitions=transitions,
                    negative_edges=verified_graph_negative_edges,
                    max_depth=1,
                    max_candidates=self.max_verified_graph_candidates,
                )
                verified_graph_candidate_count += len(candidates)
                return candidates

            def choose_verified_graph_action(
                trigger: str,
                *,
                use_llm_ranker: bool,
                candidate_actions: list[str] | None = None,
            ) -> str | None:
                nonlocal verified_graph_rank_request_count
                nonlocal verified_graph_parse_failure_count
                nonlocal verified_graph_selection_rejection_count
                allowed_targets = (
                    focus_target_stages[focus_stage_index]
                    if focus_stage_index < len(focus_target_stages)
                    else []
                    if focus_target_stages
                    else None
                )
                repair_actions = _filter_irreversible_focus_actions(
                    list(
                        valid_actions
                        if candidate_actions is None
                        else candidate_actions
                    ),
                    task_description=task_description,
                    visible_state=f"{observation}\n{_state_text(info)}",
                    allowed_focus_targets=allowed_targets,
                )
                if allowed_targets is not None and len(allowed_targets) != 1:
                    repair_actions = [
                        action
                        for action in repair_actions
                        if not action.casefold().startswith("focus on ")
                    ]
                repair_actions = [
                    action
                    for action in repair_actions
                    if action
                    not in rejected_by_state.get(
                        observation_sha256(_state_text(info)), set()
                    )
                ]
                candidates = verified_graph_candidates(
                    repair_actions=repair_actions,
                    residual_facts=verified_graph_residual_facts,
                )
                if not candidates:
                    return None
                if not use_llm_ranker:
                    return candidates[0].first_action
                if (
                    verified_graph_rank_request_count
                    >= self.max_verified_graph_rank_requests
                    or model_calls >= budget.max_model_calls
                ):
                    return None
                verified_graph_rank_request_count += 1
                try:
                    selection = call_model(
                        self._candidate_ranking_request(
                            task_description=task_description,
                            trigger=trigger,
                            residual_facts=verified_graph_residual_facts,
                            candidates=candidates,
                        ),
                        CandidateSelection,
                    )
                except Exception:  # noqa: BLE001 - conservative audited rejection
                    verified_graph_parse_failure_count += 1
                    return None
                selected = validate_selected_candidate(
                    selection.candidate_id,
                    candidates,
                    legal_actions=repair_actions,
                )
                if selected is None:
                    verified_graph_selection_rejection_count += 1
                    return None
                return selected.first_action

            def install_verified_graph_repair(trigger: str) -> bool:
                nonlocal verified_graph_active
                nonlocal verified_graph_next_action
                nonlocal verified_graph_start_score
                nonlocal verified_graph_residual_facts
                nonlocal subgoal_index, subgoals
                if (
                    not self.enable_verified_graph_repair
                    or verified_graph_active
                    or verified_graph_action_count >= self.max_verified_graph_actions
                    or (
                        replay_prefix
                        and action_replay_cursor != len(replay_prefix)
                    )
                ):
                    return False
                observed_keys = {fact.key for fact in observed_facts}
                residual_items: list[tuple[str, str, str, bool]] = []
                for candidate_subgoal in subgoals:
                    if candidate_subgoal.subgoal_id in completed_subgoal_ids:
                        continue
                    residual_items.extend(
                        residual_goal_diff(
                            (
                                fact.key
                                for fact in candidate_subgoal.expected_facts
                            ),
                            observed_keys
                            | subgoal_evidence.setdefault(
                                candidate_subgoal.subgoal_id, set()
                            ),
                            completion_mode=candidate_subgoal.completion_mode,
                        )
                    )
                residual = tuple(dict.fromkeys(residual_items))
                if not residual:
                    remaining_focus_targets = (
                        focus_target_stages[focus_stage_index]
                        if focus_stage_index < len(focus_target_stages)
                        else []
                    )
                    if len(remaining_focus_targets) == 1:
                        residual = (
                            fact_key(
                                remaining_focus_targets[0],
                                "has_state",
                                "focused",
                            ),
                        )
                if not residual:
                    task_key = task_description.casefold()
                    inferred_task_facts: list[
                        tuple[str, str, str, bool]
                    ] = []
                    for transition in infer_current_action_transitions(valid_actions):
                        for effect in transition.add_effects:
                            if effect in observed_keys:
                                continue
                            entity_tokens = {
                                token
                                for token in re.findall(
                                    r"[a-z0-9]+", f"{effect[0]} {effect[2]}"
                                )
                                if token not in {"agent", "the", "to"}
                            }
                            if entity_tokens and all(
                                token in task_key for token in entity_tokens
                            ):
                                inferred_task_facts.append(effect)
                    residual = tuple(dict.fromkeys(inferred_task_facts[:4]))
                if not residual:
                    return False
                verified_graph_residual_facts = residual
                next_action = choose_verified_graph_action(
                    trigger,
                    use_llm_ranker=True,
                )
                if next_action is None:
                    return False
                repair_subgoal = Subgoal(
                    subgoal_id="verified-graph-repair-1",
                    description="Execute a verified residual-goal graph repair.",
                    expected_facts=[
                        FactPattern(
                            subject=item[0],
                            relation=item[1],
                            object=item[2],
                            polarity=item[3],
                        )
                        for item in residual
                    ],
                    completion_test="verified residual fact or official score progress",
                    completion_mode="any",
                    retry_limit=self.max_verified_graph_actions,
                )
                verified_graph_next_action = next_action
                verified_graph_active = True
                verified_graph_start_score = final_score
                subgoals = [repair_subgoal]
                subgoal_index = 0
                subgoal_attempts[repair_subgoal.subgoal_id] = 0
                subgoal_evidence[repair_subgoal.subgoal_id] = set()
                normalizations["verified_transition_graph_repair"] = 1
                return True

            def install_controlled_residual_repair(trigger: str) -> bool:
                nonlocal controlled_repair_request_count
                nonlocal controlled_repair_accepted_count
                nonlocal controlled_repair_rejected_count
                nonlocal controlled_repair_parse_failure_count
                nonlocal controlled_repair_unsafe_rejection_count
                nonlocal controlled_repair_grounding_rejection_count
                nonlocal controlled_repair_satisfied_rejection_count
                nonlocal controlled_repair_next_action
                nonlocal controlled_repair_active
                nonlocal controlled_repair_start_score
                nonlocal subgoal_index, subgoals
                if (
                    not self.enable_controlled_residual_repair
                    or controlled_repair_request_count
                    >= self.max_controlled_repair_requests
                    or model_calls >= budget.max_model_calls
                    or controlled_repair_action_count
                    >= self.max_controlled_repair_actions
                ):
                    return False
                state_hash = observation_sha256(_state_text(info))
                blocked = rejected_by_state.get(state_hash, set())
                repair_actions = [
                    action for action in valid_actions if action not in blocked
                ]
                allowed_targets = (
                    focus_target_stages[focus_stage_index]
                    if focus_stage_index < len(focus_target_stages)
                    else []
                    if focus_target_stages
                    else None
                )
                repair_actions = _filter_irreversible_focus_actions(
                    repair_actions,
                    task_description=task_description,
                    visible_state=f"{observation}\n{_state_text(info)}",
                    allowed_focus_targets=allowed_targets,
                )
                repair_actions = [
                    action
                    for action in repair_actions
                    if not (
                        action.casefold().startswith("move ")
                        and " box" in action.casefold()
                    )
                ]
                if not repair_actions:
                    return False
                controlled_repair_request_count += 1
                try:
                    proposal = call_model(
                        self._residual_repair_request(
                            task_description=task_description,
                            observation=observation,
                            info=info,
                            facts=observed_facts,
                            valid_actions=repair_actions,
                            recent_steps=trajectory,
                            obligations=obligations,
                            trigger=trigger,
                        ),
                        ResidualRepairProposal,
                    )
                except Exception:  # noqa: BLE001 - rejection is an audited outcome
                    controlled_repair_parse_failure_count += 1
                    controlled_repair_rejected_count += 1
                    return False

                canonical_actions = {
                    action.casefold(): action for action in repair_actions
                }
                next_action = canonical_actions.get(
                    proposal.next_action.strip().casefold()
                )
                recent_zero_progress = bool(
                    trajectory
                    and trajectory[-1].action.casefold()
                    == proposal.next_action.strip().casefold()
                    and trajectory[-1].reward == 0
                    and (
                        len(trajectory) == 1
                        or trajectory[-1].score == trajectory[-2].score
                    )
                )
                if next_action is None or recent_zero_progress:
                    controlled_repair_unsafe_rejection_count += 1
                    controlled_repair_rejected_count += 1
                    return False

                repair_subgoal = proposal.residual_subgoal.as_subgoal()
                repair_subgoal.subgoal_id = "controlled-repair-1"
                observed_keys = {fact.key for fact in observed_facts}
                if any(
                    fact.key in observed_keys
                    for fact in repair_subgoal.expected_facts
                ):
                    controlled_repair_satisfied_rejection_count += 1
                    controlled_repair_rejected_count += 1
                    return False
                if any(
                    not _repair_fact_is_grounded(
                        fact,
                        task_description=task_description,
                        visible_state=f"{observation}\n{_state_text(info)}",
                        legal_actions=repair_actions,
                    )
                    for fact in repair_subgoal.expected_facts
                ):
                    controlled_repair_grounding_rejection_count += 1
                    controlled_repair_rejected_count += 1
                    return False

                controlled_repair_accepted_count += 1
                controlled_repair_next_action = next_action
                controlled_repair_active = True
                controlled_repair_start_score = final_score
                subgoals = [repair_subgoal]
                subgoal_index = 0
                subgoal_attempts[repair_subgoal.subgoal_id] = 0
                subgoal_evidence[repair_subgoal.subgoal_id] = set()
                normalizations["controlled_residual_repair"] = 1
                return True

            while len(trajectory) < budget.max_environment_steps:
                if (
                    verified_graph_active
                    and verified_graph_action_count
                    >= self.max_verified_graph_actions
                ):
                    status = "plan_exhausted"
                    break
                if (
                    controlled_repair_active
                    and controlled_repair_action_count
                    >= self.max_controlled_repair_actions
                ):
                    status = "plan_exhausted"
                    break
                if (
                    model_calls >= budget.max_model_calls
                    and controlled_repair_next_action is None
                ):
                    status = "model_budget_exhausted"
                    break
                while (
                    subgoal_index < len(subgoals)
                    and _facts_satisfied(
                        subgoals[subgoal_index].expected_facts,
                        observed_facts,
                        subgoals[subgoal_index].completion_mode,
                        subgoal_evidence.setdefault(
                            subgoals[subgoal_index].subgoal_id, set()
                        ),
                    )
                ):
                    satisfied_id = subgoals[subgoal_index].subgoal_id
                    if satisfied_id not in completed_subgoal_ids:
                        completed_subgoal_ids.append(satisfied_id)
                    if pending_continuation_recovery:
                        adaptive_continuation_recovery_count += 1
                        pending_continuation_recovery = False
                    subgoal_index += 1
                if subgoal_index >= len(subgoals):
                    if install_adaptive_continuation("plan_exhausted"):
                        continue
                    if install_verified_graph_repair("plan_exhausted"):
                        continue
                    if install_controlled_residual_repair("plan_exhausted"):
                        continue
                    status = "plan_exhausted"
                    break
                if not valid_actions:
                    status = "environment_error"
                    error = "Environment returned no valid actions"
                    break

                current_subgoal = subgoals[subgoal_index]
                state_hash = observation_sha256(_state_text(info))
                blocked = rejected_by_state.get(state_hash, set())
                eligible = [action for action in valid_actions if action not in blocked]
                if not eligible:
                    if install_adaptive_continuation("all_actions_rejected"):
                        rejected_by_state.pop(state_hash, None)
                        continue
                    if install_verified_graph_repair("all_actions_rejected"):
                        continue
                    if install_controlled_residual_repair("all_actions_rejected"):
                        continue
                    status = "plan_exhausted"
                    error = "All valid actions were rejected in the current symbolic state"
                    break
                allowed_focus_targets = (
                    focus_target_stages[focus_stage_index]
                    if focus_stage_index < len(focus_target_stages)
                    else []
                    if focus_target_stages
                    else None
                )
                if semantic_kind is not None:
                    eligible = [
                        action
                        for action in eligible
                        if action.casefold().strip() != "focus on agent"
                    ]
                else:
                    eligible = _filter_irreversible_focus_actions(
                        eligible,
                        task_description=task_description,
                        visible_state=f"{observation}\n{_state_text(info)}",
                        allowed_focus_targets=allowed_focus_targets,
                    )
                if (
                    task_id in {"3-3", "3-4"}
                    and procedure_task_target
                    and conductivity_result is None
                ):
                    eligible = [
                        action
                        for action in eligible
                        if not _is_conditional_conductivity_answer_action(
                            action,
                            procedure_task_target,
                            task_description,
                        )
                    ]
                if not eligible:
                    if install_adaptive_continuation("target_lock_blocked"):
                        continue
                    if install_verified_graph_repair("target_lock_blocked"):
                        continue
                    if install_controlled_residual_repair("target_lock_blocked"):
                        continue
                    status = "plan_exhausted"
                    error = "Target-lock filtering left no safe valid action"
                    break
                if verified_graph_active and verified_graph_next_action is None:
                    verified_graph_next_action = choose_verified_graph_action(
                        "closed_loop_refresh",
                        use_llm_ranker=False,
                        candidate_actions=eligible,
                    )
                    if verified_graph_next_action is None:
                        status = "plan_exhausted"
                        break
                candidates = shortlist_actions(
                    eligible,
                    subgoal=current_subgoal,
                    procedure_actions=[
                        *self.memory.procedure_actions(active_fragment_ids),
                        *list(train_procedure["actions"]),
                    ],
                    limit=budget.candidate_action_limit,
                )
                semantic_focus_choices = _semantic_focus_actions(
                    semantic_kind, eligible
                )
                semantic_focus_action = (
                    semantic_focus_choices[0]
                    if semantic_target_entity is None and semantic_focus_choices
                    else None
                )
                semantic_acquisition_action = (
                    _semantic_acquisition_action(eligible, semantic_target_entity)
                    if semantic_target_entity is not None
                    and not semantic_target_acquired
                    else None
                )
                semantic_move_action = (
                    _semantic_move_to_box_action(
                        eligible,
                        semantic_target_entity,
                        semantic_destination_box,
                    )
                    if semantic_target_acquired
                    else None
                )
                semantic_search_location = {
                    "living": "outside",
                    "plant": "greenhouse",
                    "longest_lifespan": "outside",
                }.get(semantic_kind or "")
                semantic_search_route = (
                    self.memory.route_action(
                        eligible,
                        current_location=current_location,
                        target_location=semantic_search_location,
                    )
                    if semantic_target_entity is None
                    and semantic_focus_action is None
                    else None
                )
                semantic_destination_route = (
                    self.memory.route_action(
                        eligible,
                        current_location=current_location,
                        target_location=semantic_destination_room,
                    )
                    if semantic_target_acquired and semantic_move_action is None
                    else None
                )
                task_target = explicit_task_target
                safe_focus_actions = sorted(
                    (
                        action
                        for action in eligible
                        if action.casefold().startswith("focus on ")
                    ),
                    key=str.casefold,
                ) if self.enable_symbolic_control else []
                workflow_action = (
                    _state_change_workflow_action(
                        eligible,
                        task_description=task_description,
                        task_target=task_target,
                        observed_facts=observed_facts,
                        entity_locations=entity_locations,
                        unavailable_devices=unavailable_devices,
                    )
                    if self.enable_symbolic_control
                    else None
                )
                target_acquisition_action = (
                    _task_target_acquisition_action(
                        eligible,
                        task_target=task_target,
                        observed_facts=observed_facts,
                    )
                    if self.enable_symbolic_control
                    else None
                )
                goal_action = (
                    _goal_directed_action(
                        eligible,
                        current_subgoal,
                        observed_facts,
                        subgoal_evidence.setdefault(
                            current_subgoal.subgoal_id, set()
                        ),
                        entity_locations,
                    )
                    if self.enable_symbolic_control
                    else None
                )
                guided_action = (
                    _next_procedure_action(
                        (
                            [
                                action
                                for action in eligible
                                if not (
                                    allowed_focus_targets is not None
                                    and len(allowed_focus_targets) > 1
                                    and action.casefold().startswith("focus on ")
                                )
                            ]
                            if task_id in {"2-2", "2-3"}
                            else _procedure_action_domain(eligible, current_subgoal)
                        ),
                        list(train_procedure["actions"]),
                        cursor=procedure_cursor,
                        task_target=procedure_task_target,
                        focus_object=procedure_focus_object,
                    )
                    if self.enable_procedure_guidance
                    else None
                )
                exploration_action = (
                    _systematic_exploration_action(
                        eligible,
                        visited_locations=visited_locations,
                        current_location=current_location,
                        destination_visit_counts=destination_visit_counts,
                        explored_open_actions=explored_open_actions,
                    )
                    if self.enable_symbolic_control
                    else None
                )
                measurement_focus_action = _measurement_conditioned_focus_action(
                    task_description,
                    safe_focus_actions,
                    transition_temperature if task_id == "2-3" else None,
                )
                measurement_workflow_action = (
                    _measurement_poll_action(
                        eligible,
                        procedure_task_target,
                        prefer_examine=(
                            bool(trajectory)
                            and trajectory[-1].action.casefold().startswith(
                                "use thermometer"
                            )
                        ),
                    )
                    if task_id == "2-3"
                    and measurement_workflow_active
                    and transition_temperature is None
                    and procedure_task_target
                    else None
                )
                if (
                    self.enable_symbolic_control
                    and task_id in {"3-3", "3-4"}
                    and focused_target
                    and procedure_task_target
                    and conductivity_components is None
                ):
                    conductivity_components = _conductivity_probe_components(eligible)
                conductivity_probe_choice = (
                    _conductivity_probe_action(
                        eligible,
                        procedure_task_target,
                        conductivity_components,
                        connection_index=conductivity_connection_index,
                        wait_count=conductivity_probe_wait_count,
                    )
                    if self.enable_symbolic_control
                    and procedure_task_target
                    and conductivity_components is not None
                    and conductivity_result is None
                    else None
                )
                conductivity_move_action = (
                    _conductivity_conditioned_move_action(
                        eligible,
                        procedure_task_target,
                        task_description,
                        conductivity_result,
                    )
                    if self.enable_symbolic_control
                    and procedure_task_target
                    and conductivity_result is not None
                    else None
                )
                decision: ActionChoice | None = None
                conductivity_action_kind: str | None = None
                semantic_action_kind: str | None = None
                semantic_action_target: str | None = None
                action_source: Literal[
                    "procedure",
                    "symbolic_goal",
                    "exploration",
                    "llm",
                    "controlled_repair",
                ] = "llm"
                force_adaptive_slow_path = bool(
                    self.enable_adaptive_slow_path
                    and self.force_llm_after_local_replan
                    and pending_replan_recovery
                    and task_id not in {"2-2", "2-3", "3-3", "3-4"}
                )
                if verified_graph_next_action is not None:
                    decision = ActionChoice(
                        action=verified_graph_next_action,
                        expected_fact=None,
                        expected_progress=(
                            "Execute the first edge of a verified transition-graph "
                            "candidate."
                        ),
                    )
                    verified_graph_next_action = None
                    action_source = "verified_graph_repair"
                elif controlled_repair_next_action is not None:
                    decision = ActionChoice(
                        action=controlled_repair_next_action,
                        expected_fact=None,
                        expected_progress=(
                            "Execute the validated first action of the residual repair."
                        ),
                    )
                    controlled_repair_next_action = None
                    action_source = "controlled_repair"
                elif semantic_move_action is not None:
                    decision = ActionChoice(
                        action=semantic_move_action,
                        expected_fact=None,
                        expected_progress=(
                            "Deliver the verified semantic target to the requested box."
                        ),
                    )
                    semantic_action_kind = "delivery"
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif semantic_acquisition_action is not None:
                    decision = ActionChoice(
                        action=semantic_acquisition_action,
                        expected_fact=None,
                        expected_progress=(
                            "Acquire the verified semantic target for delivery."
                        ),
                    )
                    semantic_action_kind = "acquisition"
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif semantic_destination_route is not None:
                    decision = ActionChoice(
                        action=semantic_destination_route,
                        expected_fact=None,
                        expected_progress=(
                            "Follow the train-derived room graph to the destination."
                        ),
                    )
                    semantic_action_kind = "route"
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif semantic_focus_action is not None:
                    semantic_action_target = semantic_focus_action[len("focus on ") :]
                    decision = ActionChoice(
                        action=semantic_focus_action,
                        expected_fact=None,
                        expected_progress=(
                            "Focus on an entity admitted by the semantic type constraint."
                        ),
                    )
                    semantic_action_kind = "focus"
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif semantic_search_route is not None:
                    decision = ActionChoice(
                        action=semantic_search_route,
                        expected_fact=None,
                        expected_progress=(
                            "Follow the train-derived room graph to the semantic search area."
                        ),
                    )
                    semantic_action_kind = "route"
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif (
                    not force_adaptive_slow_path
                    and safe_focus_actions
                    and allowed_focus_targets is not None
                    and len(allowed_focus_targets) == 1
                ):
                    decision = ActionChoice(
                        action=safe_focus_actions[0],
                        expected_fact=None,
                        expected_progress="Advance the ordered focus protocol.",
                    )
                    action_source = "symbolic_goal"
                elif not force_adaptive_slow_path and measurement_focus_action is not None:
                    decision = ActionChoice(
                        action=measurement_focus_action,
                        expected_fact=None,
                        expected_progress=(
                            "Choose the answer from the measured phase-transition temperature."
                        ),
                    )
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                    measurement_conditioned_focus_count += 1
                elif not force_adaptive_slow_path and conductivity_move_action is not None:
                    decision = ActionChoice(
                        action=conductivity_move_action,
                        expected_fact=None,
                        expected_progress=(
                            "Place the tested target in the box specified by the observed "
                            "conductivity result."
                        ),
                    )
                    action_source = "symbolic_goal"
                    conductivity_action_kind = "move"
                    symbolic_goal_action_count += 1
                elif not force_adaptive_slow_path and conductivity_probe_choice is not None:
                    probe_action, conductivity_action_kind = conductivity_probe_choice
                    decision = ActionChoice(
                        action=probe_action,
                        expected_fact=None,
                        expected_progress=(
                            "Execute the next grounded conductivity-probe step."
                        ),
                    )
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif not force_adaptive_slow_path and measurement_workflow_action is not None:
                    decision = ActionChoice(
                        action=measurement_workflow_action,
                        expected_fact=None,
                        expected_progress=(
                            "Poll the target phase and thermometer until transition."
                        ),
                    )
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif (
                    not force_adaptive_slow_path
                    and safe_focus_actions
                    and task_target
                    and not focused_target
                ):
                    decision = ActionChoice(
                        action=safe_focus_actions[0],
                        expected_fact=None,
                        expected_progress="Focus on the explicit task target.",
                    )
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif not force_adaptive_slow_path and target_acquisition_action is not None:
                    decision = ActionChoice(
                        action=target_acquisition_action,
                        expected_fact=None,
                        expected_progress=(
                            "Acquire the explicit task target before manipulation."
                        ),
                    )
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                elif not force_adaptive_slow_path and goal_action is not None:
                    decision = ActionChoice(
                        action=goal_action,
                        expected_fact=None,
                        expected_progress=(
                            "Ground an unsatisfied symbolic goal in a legal action."
                        ),
                    )
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                    if (
                        _navigation_destination(goal_action)
                        in set(entity_locations.values()) - {"agent"}
                    ):
                        entity_memory_routing_action_count += 1
                elif not force_adaptive_slow_path and workflow_action is not None:
                    decision = ActionChoice(
                        action=workflow_action,
                        expected_fact=None,
                        expected_progress=(
                            "Execute the verified state-change workflow for the target."
                        ),
                    )
                    action_source = "symbolic_goal"
                    symbolic_goal_action_count += 1
                    if (
                        _navigation_destination(workflow_action)
                        in set(entity_locations.values()) - {"agent"}
                    ):
                        entity_memory_routing_action_count += 1
                elif not force_adaptive_slow_path and guided_action is not None:
                    action, procedure_cursor = guided_action
                    decision = ActionChoice(
                        action=action,
                        expected_fact=None,
                        expected_progress="Execute the next legal train-procedure analogue.",
                    )
                    action_source = "procedure"
                    procedure_guided_action_count += 1
                elif not force_adaptive_slow_path and exploration_action is not None:
                    decision = ActionChoice(
                        action=exploration_action,
                        expected_fact=None,
                        expected_progress="Inspect an unvisited room or unopened container.",
                    )
                    action_source = "exploration"
                    exploration_action_count += 1
                else:
                    action_error: Exception | None = None
                    for action_attempt in range(2):
                        if model_calls >= budget.max_model_calls:
                            break
                        try:
                            decision = call_model(
                                self._action_request(
                                    task_description=task_description,
                                    subgoal=current_subgoal,
                                    observation=observation,
                                    info=info,
                                    facts=observed_facts,
                                    fragment_ids=active_fragment_ids,
                                    procedure=train_procedure,
                                    candidates=candidates,
                                    recent_steps=trajectory,
                                    obligations=obligations,
                                    compact_retry=action_attempt > 0,
                                ),
                                ActionChoice,
                            )
                            break
                        except Exception as exc:  # noqa: BLE001
                            action_parse_failures += 1
                            action_error = exc
                    if decision is None:
                        status = "adapter_error"
                        if action_error is not None:
                            error_type = type(action_error).__name__
                            error = str(action_error)
                        break
                    if force_adaptive_slow_path:
                        adaptive_slow_action_count += 1
                expected_replay_step: dict[str, Any] | None = None
                if action_replay_cursor < len(replay_prefix):
                    expected_replay_step = replay_prefix[action_replay_cursor]
                    expected_action = str(expected_replay_step.get("action", ""))
                    replay_legal = {
                        action.strip().casefold(): action for action in valid_actions
                    }
                    replay_action = replay_legal.get(expected_action.strip().casefold())
                    if replay_action is None:
                        action_replay_mismatch_count += 1
                        status = "prefix_replay_error"
                        error = (
                            "Recorded control action is not legal at prefix step "
                            f"{action_replay_cursor + 1}: {expected_action!r}"
                        )
                        break
                    if decision.action.strip().casefold() != replay_action.casefold():
                        action_replay_override_count += 1
                    replay_expected_fact = expected_replay_step.get("expected_fact")
                    decision = ActionChoice(
                        action=replay_action,
                        expected_fact=(
                            PlannerFact.model_validate(replay_expected_fact)
                            if replay_expected_fact is not None
                            else None
                        ),
                        expected_progress=str(
                            expected_replay_step.get(
                                "expected_progress",
                                "Replay the exact recorded D0 control action.",
                            )
                        ),
                    )
                    action_source = str(
                        expected_replay_step.get(
                            "action_source", "replayed_control"
                        )
                    )
                    conductivity_action_kind = None
                    semantic_action_kind = None
                decision_domain = (
                    valid_actions
                    if expected_replay_step is not None
                    else eligible
                    if action_source != "llm"
                    else candidates
                )
                canonical_actions = {
                    action.casefold(): action for action in decision_domain
                }
                canonical_action = canonical_actions.get(decision.action.strip().casefold())
                if canonical_action is None:
                    invalid_decisions += 1
                    if invalid_decisions >= budget.max_invalid_decisions:
                        status = "invalid_decision_limit"
                        break
                    continue
                if canonical_action != decision.action:
                    normalizations["canonicalized_candidate_action"] = (
                        normalizations.get("canonicalized_candidate_action", 0) + 1
                    )
                    decision.action = canonical_action

                before_observation = observation
                before_facts = observed_facts
                before_state_text = _state_text(info)
                score_before_action = final_score
                action_location = current_location
                try:
                    observation, reward, completed, info = env.step(decision.action)
                except Exception as exc:  # noqa: BLE001 - environment failures are outcomes
                    status = "environment_error"
                    error_type = type(exc).__name__
                    error = str(exc)
                    break
                if conductivity_action_kind == "connection":
                    conductivity_connection_index += 1
                    conductivity_probe_action_count += 1
                elif conductivity_action_kind == "wait":
                    conductivity_probe_wait_count += 1
                    conductivity_probe_action_count += 1
                elif conductivity_action_kind == "move":
                    conductivity_conditioned_move_count += 1
                temperature_match = re.search(
                    r"temperature of\s+(-?\d+(?:\.\d+)?)\s+degrees celsius",
                    observation,
                    flags=re.IGNORECASE,
                )
                if temperature_match:
                    last_measured_temperature = float(temperature_match.group(1))
                if task_id == "2-3" and (
                    decision.action.casefold() == "activate stove"
                    or (
                        decision.action.casefold().startswith("move ")
                        and " to freezer" in decision.action.casefold()
                    )
                ):
                    measurement_workflow_active = True
                if task_id == "2-3" and procedure_task_target:
                    decision_temperature = _measurement_decision_temperature(
                        task_description,
                        procedure_task_target,
                        _state_text(info),
                        last_measured_temperature,
                    )
                    if decision_temperature is not None:
                        transition_temperature = decision_temperature
                if (
                    task_id in {"3-3", "3-4"}
                    and conductivity_components is not None
                    and conductivity_result is None
                ):
                    conductivity_result = _conductivity_detector_result(
                        _state_text(info),
                        conductivity_components[3],
                        probe_complete=(
                            conductivity_connection_index >= 6
                            and conductivity_probe_wait_count >= 2
                        ),
                    )
                final_score = int(info.get("score", final_score + reward))
                prefix_step_error = False
                if verified_graph_active:
                    verified_graph_action_count += 1
                if controlled_repair_active:
                    controlled_repair_action_count += 1
                unavailable_devices.update(
                    _failed_devices_from_observation(observation)
                )
                navigation_destination = _navigation_destination(decision.action)
                if navigation_destination is not None:
                    destination_visit_counts[navigation_destination] = (
                        destination_visit_counts.get(navigation_destination, 0) + 1
                    )
                if decision.action.casefold().startswith("open "):
                    explored_open_actions.add(
                        (action_location, decision.action.casefold())
                    )
                valid_actions = list(info.get("valid", []))
                step_index = len(trajectory) + 1
                observed_facts = parse_observation_facts(
                    _state_text(info),
                    step=step_index,
                    support_action=decision.action,
                )
                _update_entity_locations(observed_facts, entity_locations)
                if expected_replay_step is not None:
                    expected_symbolic_hash = str(
                        expected_replay_step.get("symbolic_state_sha256", "")
                    )
                    expected_raw_hash = str(
                        expected_replay_step.get("observation_sha256", "")
                    )
                    expected_score = int(
                        expected_replay_step.get("score", final_score)
                    )
                    actual_symbolic_hash = symbolic_state_sha256(
                        fact.key for fact in observed_facts
                    )
                    actual_raw_hash = observation_sha256(observation)
                    state_matches = (
                        actual_symbolic_hash == expected_symbolic_hash
                        if expected_symbolic_hash
                        else actual_raw_hash == expected_raw_hash
                    )
                    if not state_matches or final_score != expected_score:
                        action_replay_mismatch_count += 1
                        prefix_step_error = True
                        status = "prefix_replay_error"
                        error = (
                            "Recorded control transition mismatch at prefix step "
                            f"{action_replay_cursor + 1}: "
                            f"symbolic_state={state_matches}, "
                            f"score={final_score == expected_score}"
                        )
                    else:
                        action_replay_cursor += 1
                new_location_repairs = _repair_expected_locations(
                    subgoals, entity_locations
                )
                if new_location_repairs:
                    normalization_name = (
                        "repaired_expected_location_from_verified_observation"
                    )
                    normalizations[normalization_name] = (
                        normalizations.get(normalization_name, 0)
                        + new_location_repairs
                    )
                visited_locations.update(
                    fact.object
                    for fact in observed_facts
                    if fact.subject == "agent"
                    and fact.relation == "located_in"
                    and fact.object != "door"
                )
                observed_location = next(
                    (
                        fact.object
                        for fact in observed_facts
                        if fact.subject == "agent"
                        and fact.relation == "located_in"
                        and fact.object != "door"
                    ),
                    None,
                )
                if observed_location is not None:
                    current_location = observed_location
                if semantic_action_kind is not None:
                    knowledge_graph_action_count += 1
                    if semantic_action_kind == "focus" and int(reward) > 0:
                        semantic_target_entity = semantic_action_target
                        semantic_focus_action_count += 1
                    elif semantic_action_kind == "acquisition" and int(reward) > 0:
                        semantic_target_acquired = True
                        semantic_acquisition_action_count += 1
                    elif semantic_action_kind == "delivery" and int(reward) > 0:
                        semantic_delivery_action_count += 1
                if (
                    decision.action.casefold().startswith("focus on ")
                    and int(reward) > 0
                ):
                    focused_target = True
                    if focus_stage_index < len(focus_target_stages):
                        focus_stage_index += 1
                        focus_stage_advance_count += 1
                fact_history.extend(observed_facts)
                if self.enable_evidence_debt_scheduler:
                    evidence_debt_resolved_count += evidence_debt_ledger.reconcile(
                        fact.key for fact in observed_facts
                    )
                immediate_expected = (
                    decision.expected_fact.as_fact_pattern()
                    if decision.expected_fact is not None
                    else None
                )
                transition_subgoal = Subgoal(
                    subgoal_id=current_subgoal.subgoal_id,
                    description=current_subgoal.description,
                    expected_facts=(
                        [immediate_expected] if immediate_expected is not None else []
                    ),
                    completion_test=current_subgoal.completion_test,
                    completion_mode="all",
                    retry_limit=current_subgoal.retry_limit,
                )
                subgoal_satisfied_after = _facts_satisfied(
                    current_subgoal.expected_facts,
                    observed_facts,
                    current_subgoal.completion_mode,
                    subgoal_evidence.setdefault(current_subgoal.subgoal_id, set()),
                )
                verified_graph_step_progress = bool(
                    int(reward) > 0
                    or final_score > score_before_action
                    or final_score > verified_graph_start_score
                    or subgoal_satisfied_after
                )
                if (
                    verified_graph_active
                    and not verified_graph_progress_verified
                    and verified_graph_step_progress
                ):
                    verified_graph_progress_verified = True
                    verified_graph_verified_progress_count += 1
                if verified_graph_active and not verified_graph_step_progress:
                    edge = (
                        symbolic_state_sha256(fact.key for fact in before_facts),
                        decision.action.strip().casefold(),
                    )
                    if edge not in verified_graph_negative_edges:
                        verified_graph_negative_edges.add(edge)
                        verified_graph_negative_edge_count += 1
                if (
                    controlled_repair_active
                    and not controlled_repair_progress_verified
                    and (
                        int(reward) > 0
                        or final_score > score_before_action
                        or final_score > controlled_repair_start_score
                        or subgoal_satisfied_after
                    )
                ):
                    controlled_repair_progress_verified = True
                    controlled_repair_verified_progress_count += 1
                    controlled_repair_recovery_count += 1
                if self.enable_symbolic_transition_verifier:
                    assessment = verify_transition(
                        subgoal=transition_subgoal,
                        action=decision.action,
                        step=step_index,
                        observation_before=before_observation,
                        observation_after=observation,
                        reward=int(reward),
                        facts_before=before_facts,
                        facts_after=observed_facts,
                        state_before=before_state_text,
                        state_after=_state_text(info),
                        retry_exhausted=False,
                    )
                else:
                    assessment = TransitionAssessment(
                        observation_changed=before_state_text != _state_text(info),
                        symbolic_state_changed=False,
                        action_error=False,
                    )
                temporal_wait_action = (
                    action_source == "symbolic_goal"
                    and decision.action.casefold() in {"wait", "wait1"}
                )
                measurement_poll_action = bool(
                    task_id in {"2-2", "2-3"}
                    and decision.action.casefold().startswith(
                        ("examine ", "look at ", "use thermometer")
                    )
                )
                conductivity_protocol_action = conductivity_action_kind in {
                    "connection",
                    "wait",
                }
                if (
                    temporal_wait_action
                    or measurement_poll_action
                    or conductivity_protocol_action
                    or semantic_action_kind is not None
                ):
                    assessment.obligations.clear()
                    if temporal_wait_action:
                        temporal_wait_action_count += 1
                    if measurement_poll_action:
                        measurement_poll_action_count += 1
                next_state_hash = observation_sha256(_state_text(info))
                navigation_destinations = {
                    destination
                    for candidate in eligible
                    if (destination := _navigation_destination(candidate)) is not None
                }
                cycle_transition = (
                    not temporal_wait_action
                    and not measurement_poll_action
                    and next_state_hash != state_hash
                    and next_state_hash in seen_state_hashes
                    and int(reward) == 0
                    and not subgoal_satisfied_after
                    and len(navigation_destinations) > 1
                )
                seen_state_hashes.add(next_state_hash)
                if (
                    self.enable_symbolic_transition_verifier
                    and cycle_transition
                    and not assessment.obligations
                ):
                    stable = (
                        f"{current_subgoal.subgoal_id}|cycle|{decision.action}|"
                        f"{step_index}|{next_state_hash}"
                    )
                    assessment.obligations.append(
                        EvidenceObligation(
                            obligation_id=hashlib.sha256(stable.encode()).hexdigest()[:20],
                            source="no_progress",
                            subgoal_id=current_subgoal.subgoal_id,
                            action=decision.action,
                            step=step_index,
                            observation_sha256=observation_sha256(observation),
                            missing_or_conflicting_facts=[
                                fact
                                for fact in current_subgoal.expected_facts
                                if fact.key
                                not in subgoal_evidence.setdefault(
                                    current_subgoal.subgoal_id, set()
                                )
                            ],
                            detail=(
                                "The action returned to a previously visited symbolic state."
                            ),
                        )
                    )
                if assessment.obligations:
                    subgoal_attempts[current_subgoal.subgoal_id] = (
                        subgoal_attempts.get(current_subgoal.subgoal_id, 0) + 1
                    )
                else:
                    subgoal_attempts[current_subgoal.subgoal_id] = 0
                retry_exhausted = (
                    subgoal_attempts[current_subgoal.subgoal_id]
                    >= current_subgoal.retry_limit
                    and not subgoal_satisfied_after
                )
                if (
                    self.enable_symbolic_transition_verifier
                    and retry_exhausted
                    and not assessment.obligations
                ):
                    after_hash = observation_sha256(observation)
                    stable = (
                        f"{current_subgoal.subgoal_id}|retry_exhausted|"
                        f"{decision.action}|{step_index}|{after_hash}"
                    )
                    assessment.obligations.append(
                        EvidenceObligation(
                            obligation_id=hashlib.sha256(stable.encode()).hexdigest()[:20],
                            source="retry_exhausted",
                            subgoal_id=current_subgoal.subgoal_id,
                            action=decision.action,
                            step=step_index,
                            observation_sha256=after_hash,
                            missing_or_conflicting_facts=[
                                fact
                                for fact in current_subgoal.expected_facts
                                if fact.key
                                not in (
                                    {item.key for item in observed_facts}
                                    | subgoal_evidence.setdefault(
                                        current_subgoal.subgoal_id, set()
                                    )
                                )
                            ],
                            detail="The current subgoal exhausted its retry budget.",
                        )
                    )
                raw_obligations = list(assessment.obligations)
                if raw_obligations:
                    rejected_actions = rejected_by_state.setdefault(state_hash, set())
                    cycle_destination = (
                        _navigation_destination(decision.action)
                        if cycle_transition
                        else None
                    )
                    if cycle_destination is not None:
                        rejected_actions.update(
                            candidate
                            for candidate in eligible
                            if _navigation_destination(candidate) == cycle_destination
                        )
                    else:
                        rejected_actions.add(decision.action)
                novel_obligations: list[EvidenceObligation] = []
                for obligation in assessment.obligations:
                    signature = (
                        obligation.subgoal_id,
                        obligation.source,
                        obligation.action.casefold(),
                        obligation.observation_sha256,
                        tuple(fact.key for fact in obligation.missing_or_conflicting_facts),
                    )
                    if signature in handled_obligation_signatures:
                        continue
                    handled_obligation_signatures.add(signature)
                    novel_obligations.append(obligation)
                assessment.obligations = novel_obligations
                verified_expected = bool(assessment.verified_expected_facts)
                if self.enable_symbolic_transition_verifier:
                    if verified_expected:
                        verified_transition_count += 1
                    elif subgoal_satisfied_after:
                        verified_transition_count += 1
                    elif immediate_expected is not None:
                        contradiction_rejection_count += 1
                obligations.extend(assessment.obligations)
                trajectory.append(
                    ENSRStep(
                        step_index=step_index,
                        subgoal_id=current_subgoal.subgoal_id,
                        action=decision.action,
                        action_source=action_source,
                        expected_fact=immediate_expected,
                        expected_progress=decision.expected_progress,
                        reward=int(reward),
                        score=final_score,
                        completed=bool(completed),
                        observation_sha256=observation_sha256(observation),
                        fact_count=len(observed_facts),
                        verified_expected_fact=verified_expected,
                        obligation_ids=[
                            obligation.obligation_id
                            for obligation in assessment.obligations
                        ],
                    )
                )

                if prefix_step_error:
                    break

                if completed:
                    if _facts_satisfied(
                        current_subgoal.expected_facts,
                        observed_facts,
                        current_subgoal.completion_mode,
                        subgoal_evidence.setdefault(
                            current_subgoal.subgoal_id, set()
                        ),
                    ) and current_subgoal.subgoal_id not in completed_subgoal_ids:
                        completed_subgoal_ids.append(current_subgoal.subgoal_id)
                    if pending_replan_recovery and (
                        verified_expected
                        or _facts_satisfied(
                            current_subgoal.expected_facts,
                            observed_facts,
                            current_subgoal.completion_mode,
                            subgoal_evidence.setdefault(
                                current_subgoal.subgoal_id, set()
                            ),
                        )
                    ):
                        local_replan_recovery_count += 1
                        pending_replan_recovery = False
                    if pending_continuation_recovery:
                        adaptive_continuation_recovery_count += 1
                        pending_continuation_recovery = False
                    status = (
                        "terminal_success" if final_score > 0 else "terminal_failure"
                    )
                    break

                if subgoal_satisfied_after:
                    completed_subgoal_ids.append(current_subgoal.subgoal_id)
                    subgoal_index += 1
                    if pending_replan_recovery:
                        local_replan_recovery_count += 1
                        pending_replan_recovery = False
                    if pending_continuation_recovery:
                        adaptive_continuation_recovery_count += 1
                        pending_continuation_recovery = False
                    continue

                if assessment.obligations:
                    obligation = assessment.obligations[0]
                    if (
                        self.enable_obligation_conditioned_retrieval
                        and (
                            self.enable_evidence_debt_scheduler
                            or retrieval_calls < budget.max_retrieval_calls
                        )
                    ):
                        query = obligation.retrieval_query()
                        candidate_ids = self.memory.search(
                            query,
                            task_id=task_id,
                            limit=budget.obligation_retrieval_k,
                        )
                        returned_ids = candidate_ids
                        execute_retrieval = True
                        if self.enable_evidence_debt_scheduler:
                            schedule = evidence_debt_ledger.schedule(
                                EvidenceDebtSignal(
                                    obligation_id=obligation.obligation_id,
                                    subgoal_id=obligation.subgoal_id,
                                    source=obligation.source,
                                    action=obligation.action,
                                    step=obligation.step,
                                    fact_keys=tuple(
                                        fact.key
                                        for fact in obligation.missing_or_conflicting_facts
                                    ),
                                ),
                                candidate_fragment_ids=candidate_ids,
                                active_fragment_ids=active_fragment_ids,
                                remaining_retrieval_calls=(
                                    budget.max_retrieval_calls - retrieval_calls
                                ),
                                max_retrieval_calls=budget.max_retrieval_calls,
                                max_retrieval_k=budget.obligation_retrieval_k,
                            )
                            returned_ids = (
                                candidate_ids[: schedule.requested_k]
                                if schedule.action == "retrieve"
                                else []
                            )
                            execute_retrieval = schedule.action == "retrieve"
                            evidence_schedule_events.append(
                                ENSREvidenceScheduleEvent(
                                    obligation_id=schedule.obligation_id,
                                    debt_signature=schedule.debt_signature,
                                    source=schedule.source,
                                    evidence_state=schedule.evidence_state,
                                    schedule_action=schedule.action,
                                    requested_k=schedule.requested_k,
                                    priority=schedule.priority,
                                    severity=schedule.severity,
                                    uncertainty=schedule.uncertainty,
                                    candidate_novelty=schedule.candidate_novelty,
                                    budget_ratio=schedule.budget_ratio,
                                    occurrence_count=schedule.occurrence_count,
                                    reason=schedule.reason,
                                    candidate_fragment_ids=candidate_ids,
                                    selected_fragment_ids=returned_ids,
                                )
                            )
                            if returned_ids:
                                evidence_debt_ledger.commit_retrieval(
                                    schedule.debt_signature, returned_ids
                                )
                        if execute_retrieval:
                            previous_ids = list(active_fragment_ids)
                            if self.fragment_memory_policy == "accumulate":
                                active_fragment_ids = list(
                                    dict.fromkeys([*previous_ids, *returned_ids])
                                )
                            else:
                                active_fragment_ids = returned_ids
                            retrieval_calls += 1
                            retrieval_events.append(
                                ENSRRetrievalEvent(
                                    call_index=retrieval_calls,
                                    trigger="obligation",
                                    query=query,
                                    returned_fragment_ids=returned_ids,
                                    new_fragment_ids=[
                                        item
                                        for item in returned_ids
                                        if item not in previous_ids
                                    ],
                                    replaced_fragment_ids=(
                                        []
                                        if self.fragment_memory_policy == "accumulate"
                                        else [
                                            item
                                            for item in previous_ids
                                            if item not in returned_ids
                                        ]
                                    ),
                                    obligation_id=obligation.obligation_id,
                                )
                            )
                    if (
                        deterministic_fallback_plan_count == 0
                        and not verified_graph_active
                        and _local_replan_has_budget(
                            planner_calls=planner_calls,
                            max_planner_calls=budget.max_planner_calls,
                            adaptive_slow_path_enabled=(
                                self.enable_adaptive_slow_path
                            ),
                            adaptive_continuation_plan_count=(
                                adaptive_continuation_plan_count
                            ),
                        )
                        and model_calls < budget.max_model_calls
                    ):
                        planner_calls += 1
                        try:
                            replanned = call_model(
                                self._replan_request(
                                    task_description=task_description,
                                    current_subgoal=current_subgoal,
                                    observation=observation,
                                    facts=observed_facts,
                                    obligation=obligation,
                                    fragment_ids=active_fragment_ids,
                                    procedure=train_procedure,
                                ),
                                LocalReplan,
                            )
                            replacement = replanned.replacement_subgoal.as_subgoal()
                            replacement.subgoal_id = current_subgoal.subgoal_id
                            subgoals[subgoal_index] = replacement
                            subgoal_attempts[current_subgoal.subgoal_id] = 0
                            subgoal_evidence[current_subgoal.subgoal_id] = set()
                            local_replan_count += 1
                            pending_replan_recovery = True
                        except Exception:  # noqa: BLE001
                            planner_parse_failures += 1
                    elif retry_exhausted and deterministic_fallback_plan_count == 0:
                        if install_adaptive_continuation(
                            "local_replan_budget_reserved"
                        ):
                            continue
                        if install_verified_graph_repair(
                            "local_replan_budget_exhausted"
                        ):
                            continue
                        if install_controlled_residual_repair(
                            "local_replan_budget_exhausted"
                        ):
                            continue
                        status = "plan_exhausted"
                        break
        except Exception as exc:  # noqa: BLE001 - preserve full episode failures
            status = "adapter_error" if model_attempted else "environment_error"
            error_type = type(exc).__name__
            error = str(exc)

        completed_value = bool(completed)
        denominator = max(1, len(trajectory))
        trace_stats = getattr(self.adapter, "episode_stats", {})
        if not isinstance(trace_stats, dict):
            trace_stats = {}
        return ENSREpisodeResult(
            experiment_id=self.experiment_id,
            reporting_boundary=(
                "development_pilot_only" if split == "dev" else "formal_machine_result"
            ),
            model_id=self.adapter.model_id,
            model_profile_id=getattr(self.adapter, "profile_id", None),
            resolved_model_ids=sorted(resolved_model_ids),
            system_fingerprints=sorted(system_fingerprints),
            task_id=task_id,
            task_name=task_name,
            split=split,
            variation=variation,
            seed=seed,
            status=status,
            completed=completed_value,
            task_success=completed_value and final_score > 0,
            final_score=final_score,
            environment_steps=len(trajectory),
            model_calls=model_calls,
            planner_calls=planner_calls,
            retrieval_calls=retrieval_calls,
            invalid_decisions=invalid_decisions,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_seconds=round(time.perf_counter() - started, 6),
            active_fragment_ids=active_fragment_ids,
            retrieval_events=retrieval_events,
            evidence_schedule_events=evidence_schedule_events,
            evidence_debt_ledger=evidence_debt_ledger.snapshot(),
            plan=plan,
            adaptive_recovery_plans=adaptive_recovery_plans,
            completed_subgoal_ids=completed_subgoal_ids,
            obligations=obligations,
            trajectory=trajectory,
            temporal_fact_history=fact_history,
            mechanism_metrics={
                "verified_transition_count": verified_transition_count,
                "obligation_count": len(obligations),
                "local_replan_count": local_replan_count,
                "local_replan_recovery_count": local_replan_recovery_count,
                "adaptive_continuation_plan_count": (
                    adaptive_continuation_plan_count
                ),
                "adaptive_continuation_recovery_count": (
                    adaptive_continuation_recovery_count
                ),
                "adaptive_slow_action_count": adaptive_slow_action_count,
                "adaptive_continuation_satisfied_subgoal_drop_count": (
                    adaptive_continuation_satisfied_subgoal_drop_count
                ),
                "adaptive_continuation_truncated_subgoal_count": (
                    adaptive_continuation_truncated_subgoal_count
                ),
                "controlled_repair_request_count": controlled_repair_request_count,
                "controlled_repair_accepted_count": controlled_repair_accepted_count,
                "controlled_repair_rejected_count": controlled_repair_rejected_count,
                "controlled_repair_parse_failure_count": (
                    controlled_repair_parse_failure_count
                ),
                "controlled_repair_unsafe_rejection_count": (
                    controlled_repair_unsafe_rejection_count
                ),
                "controlled_repair_grounding_rejection_count": (
                    controlled_repair_grounding_rejection_count
                ),
                "controlled_repair_satisfied_rejection_count": (
                    controlled_repair_satisfied_rejection_count
                ),
                "controlled_repair_action_count": controlled_repair_action_count,
                "controlled_repair_verified_progress_count": (
                    controlled_repair_verified_progress_count
                ),
                "controlled_repair_recovery_count": controlled_repair_recovery_count,
                "verified_graph_search_count": verified_graph_search_count,
                "verified_graph_candidate_count": verified_graph_candidate_count,
                "verified_graph_rank_request_count": (
                    verified_graph_rank_request_count
                ),
                "verified_graph_parse_failure_count": (
                    verified_graph_parse_failure_count
                ),
                "verified_graph_selection_rejection_count": (
                    verified_graph_selection_rejection_count
                ),
                "verified_graph_action_count": verified_graph_action_count,
                "verified_graph_verified_progress_count": (
                    verified_graph_verified_progress_count
                ),
                "verified_graph_negative_edge_count": (
                    verified_graph_negative_edge_count
                ),
                "action_replay_source_count": len(replay_prefix),
                "action_replay_consumed_count": action_replay_cursor,
                "action_replay_override_count": action_replay_override_count,
                "action_replay_mismatch_count": action_replay_mismatch_count,
                "action_replay_prefix_complete": int(
                    not replay_prefix or action_replay_cursor == len(replay_prefix)
                ),
                "knowledge_graph_action_count": knowledge_graph_action_count,
                "semantic_focus_action_count": semantic_focus_action_count,
                "semantic_acquisition_action_count": (
                    semantic_acquisition_action_count
                ),
                "semantic_delivery_action_count": semantic_delivery_action_count,
                "contradictory_fact_rejection_count": (
                    contradiction_rejection_count
                ),
                "planner_parse_failure_count": planner_parse_failures,
                "deterministic_fallback_plan_count": (
                    deterministic_fallback_plan_count
                ),
                "action_parse_failure_count": action_parse_failures,
                "procedure_guided_action_count": procedure_guided_action_count,
                "symbolic_goal_action_count": symbolic_goal_action_count,
                "entity_memory_routing_action_count": (
                    entity_memory_routing_action_count
                ),
                "tracked_entity_count": len(entity_locations),
                "unavailable_device_count": len(unavailable_devices),
                "temporal_wait_action_count": temporal_wait_action_count,
                "exploration_action_count": exploration_action_count,
                "focus_stage_advance_count": focus_stage_advance_count,
                "measurement_poll_action_count": measurement_poll_action_count,
                "measurement_conditioned_focus_count": (
                    measurement_conditioned_focus_count
                ),
                "conductivity_probe_action_count": conductivity_probe_action_count,
                "conductivity_probe_wait_count": conductivity_probe_wait_count,
                "conductivity_conditioned_move_count": (
                    conductivity_conditioned_move_count
                ),
                "no_progress_action_count": sum(
                    obligation.source == "no_progress" for obligation in obligations
                ),
                "verified_transition_rate": verified_transition_count / denominator,
                "hierarchical_subgoals_enabled": int(
                    self.enable_hierarchical_subgoals
                ),
                "symbolic_transition_verifier_enabled": int(
                    self.enable_symbolic_transition_verifier
                ),
                "obligation_conditioned_retrieval_enabled": int(
                    self.enable_obligation_conditioned_retrieval
                ),
                "evidence_debt_scheduler_enabled": int(
                    self.enable_evidence_debt_scheduler
                ),
                "evidence_debt_anchor_first_subgoal_enabled": int(
                    self.evidence_debt_anchor_first_subgoal
                ),
                "evidence_scheduler_decision_count": len(
                    evidence_schedule_events
                ),
                "evidence_scheduler_retrieval_count": sum(
                    event.schedule_action == "retrieve"
                    for event in evidence_schedule_events
                ),
                "evidence_scheduler_reuse_count": sum(
                    event.schedule_action == "reuse_active_evidence"
                    for event in evidence_schedule_events
                ),
                "evidence_scheduler_deferred_count": sum(
                    event.schedule_action == "defer"
                    for event in evidence_schedule_events
                ),
                "evidence_debt_resolved_count": evidence_debt_resolved_count,
                "fragment_memory_accumulate_enabled": int(
                    self.fragment_memory_policy == "accumulate"
                ),
                "controlled_residual_repair_enabled": int(
                    self.enable_controlled_residual_repair
                ),
                "verified_transition_graph_repair_enabled": int(
                    self.enable_verified_graph_repair
                ),
                **{
                    str(name): value
                    for name, value in trace_stats.items()
                    if isinstance(value, (int, float))
                },
            },
            response_normalizations=normalizations,
            error_type=error_type,
            error=error,
        )


def write_ensr_episode_result(result: ENSREpisodeResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
