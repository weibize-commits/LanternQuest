import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FactProvenance = Literal[
    "train_gold_transition",
    "environment_transition",
    "symbolic_inference",
    "llm_candidate",
]
ObligationSource = Literal[
    "missing_precondition",
    "action_error",
    "no_progress",
    "effect_mismatch",
    "retry_exhausted",
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _normalize_entity(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip().casefold())
    value = re.sub(r"^(?:the|an|a)\s+", "", value)
    value = re.sub(r"^substance called\s+", "", value)
    value = re.sub(r"^recipe titled\s+", "", value)
    value = re.sub(r"^(?:solid|liquid|gaseous|gas|boiling|burning)\s+", "", value)
    return value.strip(" .,:;")


def _split_top_level_items(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and character == ",":
            items.append(text[start:index].strip())
            start = index + 1
        elif depth == 0 and text[index : index + 5].casefold() == " and ":
            items.append(text[start:index].strip())
            start = index + 5
            index += 4
        index += 1
    items.append(text[start:].strip().rstrip("."))
    return [item for item in items if item]


def _entity_label(text: str) -> str:
    prefix = re.split(
        r"\s*\(|,\s+(?:which|currently)\b|\.\s+(?:The|In|On)\b",
        text.strip(),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _normalize_entity(prefix)


def _matter_state(text: str) -> str | None:
    cleaned = re.sub(r"^(?:the|an|a)\s+", "", text.strip().casefold())
    match = re.match(r"(solid|liquid|gaseous|gas|boiling|burning)\b", cleaned)
    return match.group(1) if match else None


class FactPattern(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str
    relation: str
    object: str
    polarity: bool = True

    @property
    def key(self) -> tuple[str, str, str, bool]:
        return (
            _normalize_entity(self.subject),
            self.relation.casefold().strip(),
            _normalize_entity(self.object),
            self.polarity,
        )


class TemporalFact(FactPattern):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(ge=0)
    observation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    provenance: FactProvenance
    confidence: float = Field(ge=0.0, le=1.0)
    support_action: str | None = None


class Subgoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoal_id: str
    description: str
    expected_facts: list[FactPattern] = Field(default_factory=list)
    completion_test: str
    completion_mode: Literal["all", "any"] = "all"
    retry_limit: int = Field(default=2, ge=0, le=10)


class EvidenceObligation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation_id: str
    source: ObligationSource
    subgoal_id: str
    action: str
    step: int = Field(ge=1)
    observation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    missing_or_conflicting_facts: list[FactPattern] = Field(default_factory=list)
    detail: str

    def retrieval_query(self) -> str:
        fact_text = "; ".join(
            f"{fact.subject} {fact.relation} {fact.object} polarity={fact.polarity}"
            for fact in self.missing_or_conflicting_facts
        )
        return " | ".join(
            item
            for item in (
                f"subgoal={self.subgoal_id}",
                f"failure={self.source}",
                f"action={self.action}",
                f"facts={fact_text}" if fact_text else "",
                f"detail={self.detail}",
            )
            if item
        )


class TransitionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verified_expected_facts: list[FactPattern] = Field(default_factory=list)
    missing_expected_facts: list[FactPattern] = Field(default_factory=list)
    observation_changed: bool
    symbolic_state_changed: bool
    action_error: bool
    obligations: list[EvidenceObligation] = Field(default_factory=list)


_LOCATION_PATTERNS = (
    re.compile(r"This room is called the ([^.]+)\.", re.IGNORECASE),
    re.compile(r"This [^.]*?location is called the ([^.]+)\.", re.IGNORECASE),
)
_ACTION_ERROR_MARKERS = (
    "i don't understand",
    "you can't",
    "you cannot",
    "not a valid action",
    "that action is not",
    "nothing happens",
)


def observation_sha256(observation: str) -> str:
    return hashlib.sha256(observation.encode()).hexdigest()


def parse_observation_facts(
    observation: str,
    *,
    step: int,
    provenance: FactProvenance = "environment_transition",
    support_action: str | None = None,
) -> list[TemporalFact]:
    """Extract only conservative facts that are explicit in a ScienceWorld observation."""
    observation_hash = observation_sha256(observation)
    location: str | None = None
    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(observation)
        if match:
            location = _normalize_entity(match.group(1))
            break

    facts: dict[tuple[str, str, str, bool], TemporalFact] = {}

    def add(subject: str, relation: str, object_: str) -> None:
        fact = TemporalFact(
            subject=_normalize_entity(subject),
            relation=relation,
            object=_normalize_entity(object_),
            step=step,
            observation_sha256=observation_hash,
            provenance=provenance,
            confidence=1.0,
            support_action=support_action,
        )
        facts[fact.key] = fact

    if location:
        add("agent", "located_in", location)

    inventory_mode = False
    inventory_entities: set[str] = set()
    container_contents: dict[str, set[str]] = {}
    for raw_line in observation.splitlines():
        if "in your inventory, you see:" in raw_line.casefold():
            inventory_mode = True
            continue
        if not raw_line.startswith("\t"):
            if any(pattern.search(raw_line) for pattern in _LOCATION_PATTERNS):
                inventory_mode = False
            continue
        line = raw_line.strip()
        if not line:
            continue
        entity = _entity_label(line)
        if not entity or entity in {"agent", "air"}:
            continue
        if inventory_mode:
            add(entity, "in_inventory_of", "agent")
            inventory_entities.add(entity)
        elif location:
            add(entity, "visible_in", location)

        state_match = re.search(
            r"\b(?:that|which) is (?:turned )?"
            r"(open|closed|activated|deactivated|on|off)\b",
            line,
            flags=re.IGNORECASE,
        )
        if state_match:
            add(entity, "has_state", state_match.group(1))

        door_state_match = re.search(
            r"\bthe .+? door is (open|closed)\b", line, flags=re.IGNORECASE
        )
        if door_state_match:
            add(entity, "has_state", door_state_match.group(1))

        def add_contents(container: str, contents_text: str) -> None:
            for raw_content in _split_top_level_items(contents_text):
                content = _entity_label(raw_content)
                if not content or content == "nothing":
                    continue
                add(container, "contains", content)
                container_contents.setdefault(container, set()).add(content)
                matter_state = _matter_state(raw_content)
                if matter_state:
                    add(content, "has_state", matter_state)
                nested_match = re.search(
                    r"\(containing (.+)\)\s*$", raw_content, flags=re.IGNORECASE
                )
                if nested_match:
                    add_contents(content, nested_match.group(1))

        direct_content_match = re.match(
            r"^.+?\s+\(containing (.+)\)\s*$", line, flags=re.IGNORECASE
        )
        if direct_content_match:
            add_contents(entity, direct_content_match.group(1))

        placed_contents_match = re.search(
            r"\b(?:in|on) the .+? is:\s*(.+?)\.?$", line, flags=re.IGNORECASE
        )
        if placed_contents_match:
            add_contents(entity, placed_contents_match.group(1))

    # ScienceWorld reports objects inside a carried container as explicit contents of
    # that inventory item. This rule-derived closure lets the symbolic verifier treat
    # those contents as available to the agent while retaining distinct provenance.
    for container in sorted(inventory_entities):
        for content in sorted(container_contents.get(container, set())):
            fact = TemporalFact(
                subject=content,
                relation="in_inventory_of",
                object="agent",
                step=step,
                observation_sha256=observation_hash,
                provenance="symbolic_inference",
                confidence=1.0,
                support_action=support_action,
            )
            facts[fact.key] = fact

    return sorted(facts.values(), key=lambda fact: fact.key)


def verify_transition(
    *,
    subgoal: Subgoal,
    action: str,
    step: int,
    observation_before: str,
    observation_after: str,
    reward: int,
    facts_before: list[TemporalFact],
    facts_after: list[TemporalFact],
    state_before: str | None = None,
    state_after: str | None = None,
    retry_exhausted: bool = False,
) -> TransitionAssessment:
    after_keys = {fact.key for fact in facts_after}
    before_keys = {fact.key for fact in facts_before}
    verified = [fact for fact in subgoal.expected_facts if fact.key in after_keys]
    missing = [fact for fact in subgoal.expected_facts if fact.key not in after_keys]
    observation_changed = (
        state_before if state_before is not None else observation_before
    ) != (state_after if state_after is not None else observation_after)
    symbolic_state_changed = before_keys != after_keys
    lowered = observation_after.casefold()
    action_error = any(marker in lowered for marker in _ACTION_ERROR_MARKERS)
    source: ObligationSource | None = None
    detail = ""

    if action_error:
        source = (
            "missing_precondition"
            if any(token in lowered for token in ("need", "must", "first"))
            else "action_error"
        )
        detail = observation_after[:500]
    elif not observation_changed and not symbolic_state_changed and reward == 0:
        source = "no_progress"
        detail = "The action produced no observable or symbolic state change."
    elif subgoal.expected_facts and not verified:
        source = "effect_mismatch"
        detail = "None of the expected facts became observable after the action."
    elif retry_exhausted and not verified:
        source = "retry_exhausted"
        detail = "The current subgoal exhausted its retry budget."

    obligations: list[EvidenceObligation] = []
    if source:
        after_hash = observation_sha256(observation_after)
        stable = (
            f"{subgoal.subgoal_id}|{source}|{action}|{step}|{after_hash}"
        )
        obligations.append(
            EvidenceObligation(
                obligation_id=hashlib.sha256(stable.encode()).hexdigest()[:20],
                source=source,
                subgoal_id=subgoal.subgoal_id,
                action=action,
                step=step,
                observation_sha256=after_hash,
                missing_or_conflicting_facts=missing,
                detail=detail,
            )
        )

    return TransitionAssessment(
        verified_expected_facts=verified,
        missing_expected_facts=missing,
        observation_changed=observation_changed,
        symbolic_state_changed=symbolic_state_changed,
        action_error=action_error,
        obligations=obligations,
    )


def shortlist_actions(
    valid_actions: list[str],
    *,
    subgoal: Subgoal,
    procedure_actions: list[str],
    limit: int = 20,
) -> list[str]:
    if limit < 1:
        raise ValueError("limit must be positive")
    context_tokens = _tokens(subgoal.description)
    for fact in subgoal.expected_facts:
        context_tokens.update(_tokens(" ".join(fact.key[:3])))
    procedure_operators = {
        action.casefold().split(maxsplit=1)[0] for action in procedure_actions if action
    }
    procedure_patterns = [
        re.compile(
            "^"
            + re.escape(action.casefold()).replace(
                re.escape("<task_target>"), r".+?"
            ).replace(re.escape("<focus_object>"), r".+?")
            + "$"
        )
        for action in procedure_actions
        if action
    ]

    def score(action: str) -> tuple[float, str]:
        lowered = action.casefold()
        action_tokens = _tokens(lowered)
        lexical = len(context_tokens & action_tokens)
        operator = lowered.split(maxsplit=1)[0] if lowered else ""
        procedure_bonus = 2.0 if operator in procedure_operators else 0.0
        exact_procedure_bonus = (
            6.0 if any(pattern.match(lowered) for pattern in procedure_patterns) else 0.0
        )
        exploration_bonus = (
            0.5
            if lowered.startswith(("look ", "inventory", "go to ", "open door"))
            else 0.0
        )
        return (
            lexical + procedure_bonus + exact_procedure_bonus + exploration_bonus,
            lowered,
        )

    unique_actions = sorted(
        set(valid_actions),
        key=lambda action: (-score(action)[0], score(action)[1]),
    )
    selected = unique_actions[:limit]
    must_keep = [
        action
        for action in unique_actions
        if action.casefold() in {"look around", "look in inventory", "inventory"}
    ]
    for action in must_keep:
        if action in selected:
            continue
        if len(selected) == limit:
            selected[-1] = action
        else:
            selected.append(action)
    return list(dict.fromkeys(selected))[:limit]
