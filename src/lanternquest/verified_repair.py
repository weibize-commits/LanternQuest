from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

FactKey = tuple[str, str, str, bool]
TransitionProvenance = Literal[
    "current_legal_action",
    "train_verified_transition",
]


def normalize_entity(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    normalized = re.sub(r"^(?:the|an|a)\s+", "", normalized)
    normalized = re.sub(r"^substance called\s+", "", normalized)
    return normalized.strip(" .,:;")


def fact_key(
    subject: str,
    relation: str,
    object_: str,
    polarity: bool = True,
) -> FactKey:
    return (
        normalize_entity(subject),
        relation.strip().casefold(),
        normalize_entity(object_),
        bool(polarity),
    )


def symbolic_state_sha256(facts: Iterable[FactKey]) -> str:
    canonical = sorted([list(item) for item in set(facts)])
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ActionTransition:
    action: str
    preconditions: tuple[FactKey, ...]
    add_effects: tuple[FactKey, ...]
    delete_effects: tuple[FactKey, ...] = ()
    currently_legal: bool = False
    grounded: bool = True
    provenance: TransitionProvenance = "current_legal_action"
    evidence_count: int = 1
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedRepairCandidate:
    candidate_id: str
    actions: tuple[str, ...]
    predicted_add_effects: tuple[FactKey, ...]
    satisfied_residual_facts: tuple[FactKey, ...]
    final_state_sha256: str
    train_evidence_count: int

    @property
    def first_action(self) -> str:
        return self.actions[0]


def residual_goal_diff(
    goal_facts: Iterable[FactKey],
    observed_facts: Iterable[FactKey],
    *,
    completion_mode: Literal["all", "any"] = "all",
) -> tuple[FactKey, ...]:
    goals = tuple(dict.fromkeys(goal_facts))
    observed = set(observed_facts)
    if completion_mode == "any" and any(goal in observed for goal in goals):
        return ()
    return tuple(goal for goal in goals if goal not in observed)


def infer_current_action_transitions(
    legal_actions: Iterable[str],
) -> list[ActionTransition]:
    """Infer conservative one-step effects from exact ScienceWorld action strings."""
    transitions: list[ActionTransition] = []
    for action in sorted(set(legal_actions), key=str.casefold):
        lowered = re.sub(r"\s+", " ", action.strip().casefold())
        effects: list[FactKey] = []
        match = re.fullmatch(r"(?:pick up|take) (.+)", lowered)
        if match:
            effects.append(fact_key(match.group(1), "in_inventory_of", "agent"))
        match = re.fullmatch(r"move (.+?) to inventory", lowered)
        if match:
            effects.append(fact_key(match.group(1), "in_inventory_of", "agent"))
        match = re.fullmatch(r"(?:move|put|place|pour) (.+?) to (.+)", lowered)
        if match and normalize_entity(match.group(2)) != "inventory":
            effects.append(fact_key(match.group(2), "contains", match.group(1)))
        match = re.fullmatch(r"open (.+)", lowered)
        if match:
            effects.append(fact_key(match.group(1), "has_state", "open"))
        match = re.fullmatch(r"close (.+)", lowered)
        if match:
            effects.append(fact_key(match.group(1), "has_state", "closed"))
        match = re.fullmatch(r"activate (.+)", lowered)
        if match:
            effects.append(fact_key(match.group(1), "has_state", "activated"))
        match = re.fullmatch(r"deactivate (.+)", lowered)
        if match:
            effects.append(fact_key(match.group(1), "has_state", "deactivated"))
        match = re.fullmatch(r"turn on (.+)", lowered)
        if match:
            effects.append(fact_key(match.group(1), "has_state", "on"))
        match = re.fullmatch(r"turn off (.+)", lowered)
        if match:
            effects.append(fact_key(match.group(1), "has_state", "off"))
        match = re.fullmatch(r"go to (?:door to )?(.+)", lowered)
        if match:
            effects.append(fact_key("agent", "located_in", match.group(1)))
        match = re.fullmatch(r"focus on (.+)", lowered)
        if match:
            effects.append(fact_key(match.group(1), "has_state", "focused"))
        if not effects:
            continue
        transitions.append(
            ActionTransition(
                action=action,
                preconditions=(),
                add_effects=tuple(dict.fromkeys(effects)),
                currently_legal=True,
                provenance="current_legal_action",
            )
        )
    return transitions


def _apply_transition(
    state: frozenset[FactKey], transition: ActionTransition
) -> frozenset[FactKey]:
    updated = set(state)
    updated.difference_update(transition.delete_effects)
    updated.update(transition.add_effects)
    return frozenset(updated)


def _candidate_id(
    actions: tuple[str, ...], satisfied: tuple[FactKey, ...]
) -> str:
    payload = {"actions": actions, "satisfied": satisfied}
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:12]
    return f"candidate-{digest}"


def search_verified_transition_graph(
    *,
    observed_facts: Iterable[FactKey],
    residual_facts: Iterable[FactKey],
    transitions: Iterable[ActionTransition],
    negative_edges: set[tuple[str, str]] | None = None,
    max_depth: int = 3,
    max_candidates: int = 8,
) -> list[VerifiedRepairCandidate]:
    """Search bounded symbolic paths; only current legal actions may start a path."""
    if not 1 <= max_depth <= 3:
        raise ValueError("max_depth must be between one and three")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    start = frozenset(observed_facts)
    residual = tuple(dict.fromkeys(residual_facts))
    if not residual:
        return []
    blocked = negative_edges or set()
    models = sorted(
        (
            item
            for item in transitions
            if item.grounded
            and item.evidence_count > 0
            and not item.risk_flags
            and item.add_effects
        ),
        key=lambda item: (
            0 if item.provenance == "train_verified_transition" else 1,
            item.action.casefold(),
            item.add_effects,
        ),
    )
    queue = deque([(start, tuple(), tuple(), 0)])
    visited: dict[frozenset[FactKey], int] = {start: 0}
    candidates: list[VerifiedRepairCandidate] = []
    seen_first_actions: set[str] = set()
    while queue:
        state, actions, added, evidence = queue.popleft()
        if len(actions) >= max_depth:
            continue
        state_hash = symbolic_state_sha256(state)
        for transition in models:
            if not actions and not transition.currently_legal:
                continue
            if transition.action.casefold() in {
                action.casefold() for action in actions
            }:
                continue
            if not set(transition.preconditions).issubset(state):
                continue
            edge = (state_hash, transition.action.strip().casefold())
            if edge in blocked:
                continue
            next_state = _apply_transition(state, transition)
            if next_state == state:
                continue
            next_actions = (*actions, transition.action)
            next_added = tuple(dict.fromkeys((*added, *transition.add_effects)))
            newly_satisfied = tuple(fact for fact in residual if fact in next_state)
            if newly_satisfied:
                first_key = next_actions[0].casefold()
                candidate = VerifiedRepairCandidate(
                    candidate_id=_candidate_id(next_actions, newly_satisfied),
                    actions=next_actions,
                    predicted_add_effects=next_added,
                    satisfied_residual_facts=newly_satisfied,
                    final_state_sha256=symbolic_state_sha256(next_state),
                    train_evidence_count=(
                        evidence
                        + (
                            transition.evidence_count
                            if transition.provenance == "train_verified_transition"
                            else 0
                        )
                    ),
                )
                if first_key not in seen_first_actions:
                    candidates.append(candidate)
                    seen_first_actions.add(first_key)
            depth = len(next_actions)
            if depth < max_depth and visited.get(next_state, max_depth + 1) > depth:
                visited[next_state] = depth
                queue.append(
                    (
                        next_state,
                        next_actions,
                        next_added,
                        evidence
                        + (
                            transition.evidence_count
                            if transition.provenance == "train_verified_transition"
                            else 0
                        ),
                    )
                )
    candidates.sort(
        key=lambda item: (
            -len(item.satisfied_residual_facts),
            len(item.actions),
            -item.train_evidence_count,
            tuple(action.casefold() for action in item.actions),
        )
    )
    return candidates[:max_candidates]


def validate_selected_candidate(
    candidate_id: str,
    candidates: Iterable[VerifiedRepairCandidate],
    *,
    legal_actions: Iterable[str],
) -> VerifiedRepairCandidate | None:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = by_id.get(candidate_id.strip())
    if selected is None:
        return None
    legal = {action.strip().casefold() for action in legal_actions}
    if selected.first_action.strip().casefold() not in legal:
        return None
    return selected
