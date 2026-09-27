import hashlib
import random
import re
import time
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lanternquest.llm import LLMAdapter, LLMAdapterError, LLMRequest

ScienceWorldMethod = Literal[
    "random_sanity",
    "gold_oracle",
    "b0_full_skills",
    "b1_static_rag",
    "b2_task_graph_rag",
    "b4_sequential_planner",
    "iper_rag",
]
AutomaticObligationPolicy = Literal["disabled", "unsupported_or_stalled"]
ObligationSource = Literal["model_request", "unsupported_action", "execution_stall"]
EpisodeStatus = Literal[
    "terminal_success",
    "terminal_failure",
    "model_stop",
    "environment_budget_exhausted",
    "model_budget_exhausted",
    "retrieval_budget_exhausted",
    "invalid_decision_limit",
    "adapter_error",
    "environment_error",
]


class ScienceWorldEnvironment(Protocol):
    def load(
        self,
        task_name: str,
        variationIdx: int = 0,
        simplificationStr: str = "",
        generateGoldPath: bool = False,
    ) -> None: ...

    def reset(self) -> tuple[str, dict[str, Any]]: ...

    def step(self, action: str) -> tuple[str, int, bool, dict[str, Any]]: ...

    def get_gold_action_sequence(self) -> list[str]: ...


class ScienceWorldSkill(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    task_name: str
    variation: int
    task_description: str
    gold_actions: list[str]
    skill_fingerprint: str


class SkillHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_name: str
    skill_fingerprint: str
    score: float


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_type: Literal["act", "request_evidence", "stop"]
    action: str | None
    evidence_query: str | None
    expected_progress: str | None

    @model_validator(mode="after")
    def validate_decision(self) -> "ActionDecision":
        if self.decision_type == "act" and not self.action:
            raise ValueError("act decisions require an action")
        if self.decision_type == "request_evidence" and not self.evidence_query:
            raise ValueError("request_evidence decisions require a query")
        if self.decision_type != "act" and self.action is not None:
            raise ValueError("Only act decisions may include an action")
        if self.decision_type != "request_evidence" and self.evidence_query is not None:
            raise ValueError("Only request_evidence decisions may include a query")
        return self


class EpisodeBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_environment_steps: int = Field(default=100, ge=1, le=500)
    max_model_calls: int = Field(default=100, ge=1, le=500)
    max_retrieval_calls: int = Field(default=8, ge=1, le=100)
    initial_retrieval_k: int = Field(default=5, ge=1, le=100)
    obligation_retrieval_k: int = Field(default=3, ge=1, le=100)
    max_invalid_decisions: int = Field(default=5, ge=1, le=50)
    automatic_obligation_policy: AutomaticObligationPolicy = "disabled"
    max_automatic_obligations: int = Field(default=3, ge=1, le=50)
    candidate_action_limit: int | None = Field(default=None, ge=1, le=100)


class RetrievalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_index: int
    query: str
    trigger: Literal["initial", "task_graph", "obligation"]
    returned_skill_ids: list[str]
    new_skill_ids: list[str]
    obligation_source: ObligationSource | None = None


class EnvironmentStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    action: str
    reward: int
    score: int
    completed: bool
    valid_action_count: int
    observation_sha256: str
    expected_progress: str | None = None


class ScienceWorldEpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    experiment_id: str = "E11"
    reporting_boundary: str
    method: ScienceWorldMethod
    model_id: str | None = None
    model_profile_id: str | None = None
    resolved_model_ids: list[str] = Field(default_factory=list)
    system_fingerprints: list[str] = Field(default_factory=list)
    task_id: str
    task_name: str
    split: Literal["dev", "test"]
    variation: int
    seed: int
    status: EpisodeStatus
    completed: bool
    task_success: bool
    final_score: int
    environment_steps: int
    model_calls: int
    retrieval_calls: int
    invalid_decisions: int
    rejected_actions: list[str] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    wall_seconds: float = 0.0
    response_normalizations: dict[str, int] = Field(default_factory=dict)
    retrieved_skill_ids: list[str]
    retrieval_events: list[RetrievalEvent]
    trajectory: list[EnvironmentStep]
    error_type: str | None = None
    error: str | None = None


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _observation_hash(observation: str) -> str:
    return hashlib.sha256(observation.encode("utf-8")).hexdigest()


def _action_operator(action: str) -> str:
    """Return a stable operator signature without binding variation-specific entities."""
    words = re.findall(r"[a-z0-9]+", action.casefold())
    if not words:
        return ""
    if len(words) >= 2 and words[0] in {"focus", "go", "look", "pick", "put"}:
        return " ".join(words[:2])
    return words[0]


def _action_requires_procedure_support(action: str) -> bool:
    operator = _action_operator(action)
    return operator not in {
        "close",
        "go to",
        "inventory",
        "look around",
        "look at",
        "look in",
        "open",
    }


def _normalize_action_decision(content: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized = dict(content)
    changed = False
    for field in ("action", "evidence_query", "expected_progress"):
        if field not in normalized:
            normalized[field] = None
    decision_type = normalized.get("decision_type")
    if decision_type == "act" and normalized.get("evidence_query") is not None:
        normalized["evidence_query"] = None
        changed = True
    elif decision_type == "request_evidence" and normalized.get("action") is not None:
        normalized["action"] = None
        changed = True
    elif decision_type == "stop":
        for field in ("action", "evidence_query"):
            if normalized.get(field) is not None:
                normalized[field] = None
                changed = True
    return normalized, changed


def _explicit_task_target(task_description: str) -> str | None:
    patterns = (
        r"your task is to (?:boil|melt|freeze) (.+?)\.",
        r"your task is to change the state of matter of (.+?)\.",
    )
    for pattern in patterns:
        match = re.search(pattern, task_description, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _skill_prompt_view(skill: ScienceWorldSkill) -> dict[str, object]:
    """Abstract source-variation entities before showing a skill to the model."""
    source_target = _explicit_task_target(skill.task_description)
    focus_object = next(
        (
            action.removeprefix("focus on ").strip()
            for action in skill.gold_actions
            if action.casefold().startswith("focus on ")
        ),
        None,
    )
    replacements: dict[str, str] = {}
    if source_target:
        replacements[source_target.casefold()] = "<TASK_TARGET>"
    if focus_object and focus_object.casefold() not in replacements:
        replacements[focus_object.casefold()] = "<FOCUS_OBJECT>"

    if replacements:
        pattern = re.compile(
            "|".join(re.escape(term) for term in sorted(replacements, key=len, reverse=True)),
            flags=re.IGNORECASE,
        )

        def abstract(text: str) -> str:
            return pattern.sub(lambda match: replacements[match.group(0).casefold()], text)

    else:

        def abstract(text: str) -> str:
            return text

    return {
        "task_id": skill.task_id,
        "task_name": skill.task_name,
        "task_description_pattern": abstract(skill.task_description),
        "procedure_actions": [abstract(action) for action in skill.gold_actions],
        "skill_fingerprint": skill.skill_fingerprint,
    }


def load_scienceworld_skills(skill_dir: Path) -> list[ScienceWorldSkill]:
    skills: list[ScienceWorldSkill] = []
    for path in sorted(skill_dir.glob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        skills.append(
            ScienceWorldSkill.model_validate_json(path.read_text(encoding="utf-8"))
        )
    if not skills:
        raise ValueError(f"No ScienceWorld skills found in {skill_dir}")
    if len({skill.skill_fingerprint for skill in skills}) != len(skills):
        raise ValueError("ScienceWorld skill fingerprints must be unique")
    return skills


class ScienceWorldSkillRetriever:
    def __init__(self, skills: list[ScienceWorldSkill]) -> None:
        self.skills = list(skills)
        self._text = {
            skill.skill_fingerprint: " ".join(
                [skill.task_name, skill.task_description, *skill.gold_actions]
            )
            for skill in self.skills
        }

    def search(self, query: str, limit: int) -> list[SkillHit]:
        query_tokens = _tokens(query)
        scored: list[tuple[float, str, ScienceWorldSkill]] = []
        for skill in self.skills:
            skill_tokens = _tokens(self._text[skill.skill_fingerprint])
            overlap = len(query_tokens & skill_tokens)
            union = len(query_tokens | skill_tokens) or 1
            score = overlap / union
            scored.append((-score, skill.skill_fingerprint, skill))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [
            SkillHit(
                task_id=skill.task_id,
                task_name=skill.task_name,
                skill_fingerprint=skill.skill_fingerprint,
                score=-score,
            )
            for score, _, skill in scored[:limit]
        ]

    def task_graph(self, task_id: str, limit: int) -> list[SkillHit]:
        topic_id = task_id.split("-", maxsplit=1)[0]
        same_topic = [
            skill for skill in self.skills if skill.task_id.split("-", maxsplit=1)[0] == topic_id
        ]
        ordered = sorted(
            same_topic or self.skills,
            key=lambda skill: (skill.task_id != task_id, skill.task_id),
        )
        return [
            SkillHit(
                task_id=skill.task_id,
                task_name=skill.task_name,
                skill_fingerprint=skill.skill_fingerprint,
                score=1.0 if skill.task_id == task_id else 0.5,
            )
            for skill in ordered[:limit]
        ]

    def obligation_search(
        self,
        query: str,
        task_id: str,
        limit: int,
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[SkillHit]:
        """Retrieve evidence with an explicit exact-task and task-topic structural prior."""
        query_tokens = _tokens(query)
        topic_id = task_id.split("-", maxsplit=1)[0]
        excluded = exclude_ids or set()
        scored: list[tuple[float, str, ScienceWorldSkill]] = []
        for skill in self.skills:
            if skill.skill_fingerprint in excluded:
                continue
            skill_tokens = _tokens(self._text[skill.skill_fingerprint])
            overlap = len(query_tokens & skill_tokens)
            union = len(query_tokens | skill_tokens) or 1
            lexical_score = overlap / union
            structural_score = (
                1.0
                if skill.task_id == task_id
                else 0.25
                if skill.task_id.split("-", maxsplit=1)[0] == topic_id
                else 0.0
            )
            score = lexical_score + structural_score
            scored.append((-score, skill.skill_fingerprint, skill))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [
            SkillHit(
                task_id=skill.task_id,
                task_name=skill.task_name,
                skill_fingerprint=skill.skill_fingerprint,
                score=-score,
            )
            for score, _, skill in scored[:limit]
        ]


class ScienceWorldExperimentHarness:
    def __init__(
        self,
        skills: list[ScienceWorldSkill],
        adapter: LLMAdapter | None = None,
        experiment_id: str = "E11",
    ) -> None:
        self.skills = list(skills)
        self.skills_by_id = {skill.skill_fingerprint: skill for skill in skills}
        self.retriever = ScienceWorldSkillRetriever(skills)
        self.adapter = adapter
        self.experiment_id = experiment_id

    def _has_task_procedure_support(
        self,
        *,
        task_id: str,
        action: str,
        retrieved_ids: list[str],
    ) -> bool:
        operator = _action_operator(action)
        return any(
            skill.task_id == task_id
            and operator in {_action_operator(item) for item in skill.gold_actions}
            for skill_id in retrieved_ids
            for skill in [self.skills_by_id[skill_id]]
        )

    def _automatic_obligation_query(
        self,
        *,
        task_description: str,
        observation: str,
        action: str,
        expected_progress: str | None,
        source: ObligationSource,
    ) -> str:
        return " ".join(
            part
            for part in (
                task_description,
                f"evidence for action {action}",
                f"expected progress {expected_progress}" if expected_progress else "",
                "recover from unchanged state" if source == "execution_stall" else "",
                observation[:500],
            )
            if part
        )

    def run_episode(
        self,
        env: ScienceWorldEnvironment,
        *,
        method: ScienceWorldMethod,
        task_id: str,
        task_name: str,
        split: Literal["dev", "test"],
        variation: int,
        seed: int,
        budget: EpisodeBudget,
        simplifications: str = "",
    ) -> ScienceWorldEpisodeResult:
        episode_started = time.perf_counter()
        if method not in {"random_sanity", "gold_oracle"} and self.adapter is None:
            raise ValueError(f"Method {method} requires an LLM adapter")

        trajectory: list[EnvironmentStep] = []
        retrieval_events: list[RetrievalEvent] = []
        retrieved_ids: list[str] = []
        model_calls = 0
        retrieval_calls = 0
        invalid_decisions = 0
        input_tokens = 0
        output_tokens = 0
        resolved_model_ids: set[str] = set()
        system_fingerprints: set[str] = set()
        response_normalizations: dict[str, int] = {}
        final_score = 0
        completed = False
        status: EpisodeStatus = "environment_budget_exhausted"
        error_type: str | None = None
        error: str | None = None

        try:
            env.load(
                task_name,
                variationIdx=variation,
                simplificationStr=simplifications,
                generateGoldPath=method == "gold_oracle",
            )
            observation, info = env.reset()
            final_score = int(info.get("score", 0))
            task_description = str(info.get("taskDesc", task_name))
            valid_actions = list(info.get("valid", []))

            if method == "gold_oracle":
                actions = env.get_gold_action_sequence()
                return self._run_fixed_actions(
                    env=env,
                    method=method,
                    task_id=task_id,
                    task_name=task_name,
                    split=split,
                    variation=variation,
                    seed=seed,
                    actions=actions,
                    initial_valid_actions=valid_actions,
                    budget=budget,
                )

            if method == "random_sanity":
                rng = random.Random(f"{seed}|{task_id}|{variation}")
                for step_index in range(1, budget.max_environment_steps + 1):
                    if not valid_actions:
                        status = "environment_error"
                        error = "Environment returned no valid actions"
                        break
                    action = rng.choice(valid_actions)
                    observation, reward, completed, info = env.step(action)
                    final_score = int(info.get("score", final_score + reward))
                    valid_actions = list(info.get("valid", []))
                    trajectory.append(
                        EnvironmentStep(
                            step_index=step_index,
                            action=action,
                            reward=reward,
                            score=final_score,
                            completed=completed,
                            valid_action_count=len(valid_actions),
                            observation_sha256=_observation_hash(observation),
                        )
                    )
                    if completed:
                        status = (
                            "terminal_success" if final_score > 0 else "terminal_failure"
                        )
                        break
                return self._result(
                    method=method,
                    task_id=task_id,
                    task_name=task_name,
                    split=split,
                    variation=variation,
                    seed=seed,
                    status=status,
                    completed=completed,
                    final_score=final_score,
                    model_calls=0,
                    retrieval_calls=0,
                    invalid_decisions=0,
                    input_tokens=0,
                    output_tokens=0,
                    retrieved_ids=[],
                    retrieval_events=[],
                    trajectory=trajectory,
                    error_type=error_type,
                    error=error,
                    wall_seconds=time.perf_counter() - episode_started,
                )

            if method == "b0_full_skills":
                retrieved_ids = [skill.skill_fingerprint for skill in self.skills]
            elif method == "b2_task_graph_rag":
                hits = self.retriever.task_graph(task_id, budget.initial_retrieval_k)
                retrieved_ids = [hit.skill_fingerprint for hit in hits]
                retrieval_calls = 1
                retrieval_events.append(
                    RetrievalEvent(
                        call_index=1,
                        query=f"task graph topic {task_id}",
                        trigger="task_graph",
                        returned_skill_ids=retrieved_ids,
                        new_skill_ids=retrieved_ids,
                    )
                )
            else:
                hits = self.retriever.search(task_description, budget.initial_retrieval_k)
                retrieved_ids = [hit.skill_fingerprint for hit in hits]
                retrieval_calls = 1
                retrieval_events.append(
                    RetrievalEvent(
                        call_index=1,
                        query=task_description,
                        trigger="initial",
                        returned_skill_ids=retrieved_ids,
                        new_skill_ids=retrieved_ids,
                    )
                )

            unresolved_obligations: list[str] = []
            rejected_actions: list[str] = []
            no_progress_actions_by_state: dict[str, set[str]] = {}
            automatic_obligation_keys: set[str] = set()
            automatic_obligation_count = 0
            while len(trajectory) < budget.max_environment_steps:
                if model_calls >= budget.max_model_calls:
                    status = "model_budget_exhausted"
                    break
                request = self._action_request(
                    method=method,
                    task_id=task_id,
                    task_description=task_description,
                    observation=observation,
                    info=info,
                    valid_actions=valid_actions,
                    retrieved_ids=retrieved_ids,
                    trajectory=trajectory,
                    unresolved_obligations=unresolved_obligations,
                    rejected_actions=rejected_actions,
                    no_progress_actions=sorted(
                        no_progress_actions_by_state.get(
                            _observation_hash(observation), set()
                        )
                    ),
                    automatic_obligation_policy=budget.automatic_obligation_policy,
                    candidate_action_limit=budget.candidate_action_limit,
                )
                try:
                    assert self.adapter is not None
                    model_calls += 1
                    response = self.adapter.complete(request)
                    model_calls += max(0, response.provider_calls - 1)
                    input_tokens += response.input_tokens or 0
                    output_tokens += response.output_tokens or 0
                    resolved_model_ids.add(response.model_id)
                    if response.system_fingerprint:
                        system_fingerprints.add(response.system_fingerprint)
                    if response.normalization_applied:
                        response_normalizations[response.normalization_applied] = (
                            response_normalizations.get(response.normalization_applied, 0)
                            + 1
                        )
                    decision_content, decision_was_normalized = (
                        _normalize_action_decision(response.content)
                    )
                    if decision_was_normalized:
                        normalization_name = "cleared_irrelevant_decision_fields"
                        response_normalizations[normalization_name] = (
                            response_normalizations.get(normalization_name, 0) + 1
                        )
                    decision = ActionDecision.model_validate(decision_content)
                except Exception as exc:  # noqa: BLE001 - benchmark failures are outcomes
                    if isinstance(exc, LLMAdapterError):
                        model_calls += max(0, exc.provider_calls - 1)
                        input_tokens += exc.input_tokens
                        output_tokens += exc.output_tokens
                    status = "adapter_error"
                    error_type = type(exc).__name__
                    error = str(exc)
                    break

                if decision.decision_type == "stop":
                    status = "model_stop"
                    break

                if decision.decision_type == "request_evidence":
                    if (
                        method != "iper_rag"
                        or budget.automatic_obligation_policy != "disabled"
                    ):
                        invalid_decisions += 1
                    elif retrieval_calls >= budget.max_retrieval_calls:
                        status = "retrieval_budget_exhausted"
                        break
                    else:
                        assert decision.evidence_query is not None
                        hits = self.retriever.search(
                            decision.evidence_query, budget.obligation_retrieval_k
                        )
                        returned_ids = [hit.skill_fingerprint for hit in hits]
                        new_ids = [
                            skill_id
                            for skill_id in returned_ids
                            if skill_id not in retrieved_ids
                        ]
                        retrieved_ids.extend(new_ids)
                        retrieval_calls += 1
                        unresolved_obligations.append(decision.evidence_query)
                        retrieval_events.append(
                            RetrievalEvent(
                                call_index=retrieval_calls,
                                query=decision.evidence_query,
                                trigger="obligation",
                                returned_skill_ids=returned_ids,
                                new_skill_ids=new_ids,
                                obligation_source="model_request",
                            )
                        )
                        continue

                elif (
                    method == "iper_rag"
                    and budget.automatic_obligation_policy == "unsupported_or_stalled"
                    and decision.action is not None
                    and _action_requires_procedure_support(decision.action)
                    and not self._has_task_procedure_support(
                        task_id=task_id,
                        action=decision.action,
                        retrieved_ids=retrieved_ids,
                    )
                ):
                    before_observation_hash = _observation_hash(observation)
                    obligation_key = (
                        f"unsupported_action|{before_observation_hash}|{decision.action}"
                    )
                    if (
                        obligation_key not in automatic_obligation_keys
                        and retrieval_calls < budget.max_retrieval_calls
                        and automatic_obligation_count
                        < budget.max_automatic_obligations
                    ):
                        automatic_obligation_keys.add(obligation_key)
                        automatic_obligation_count += 1
                        query = self._automatic_obligation_query(
                            task_description=task_description,
                            observation=observation,
                            action=decision.action,
                            expected_progress=decision.expected_progress,
                            source="unsupported_action",
                        )
                        hits = self.retriever.obligation_search(
                            query,
                            task_id,
                            budget.obligation_retrieval_k,
                            exclude_ids=set(retrieved_ids),
                        )
                        returned_ids = [hit.skill_fingerprint for hit in hits]
                        new_ids = [
                            skill_id
                            for skill_id in returned_ids
                            if skill_id not in retrieved_ids
                        ]
                        retrieved_ids.extend(new_ids)
                        retrieval_calls += 1
                        unresolved_obligations.append(query)
                        retrieval_events.append(
                            RetrievalEvent(
                                call_index=retrieval_calls,
                                query=query,
                                trigger="obligation",
                                returned_skill_ids=returned_ids,
                                new_skill_ids=new_ids,
                                obligation_source="unsupported_action",
                            )
                        )
                        continue

                elif decision.action not in valid_actions:
                    invalid_decisions += 1
                    if decision.action:
                        rejected_actions.append(decision.action)
                elif decision.action in no_progress_actions_by_state.get(
                    _observation_hash(observation), set()
                ):
                    invalid_decisions += 1
                    assert decision.action is not None
                    rejected_actions.append(decision.action)
                else:
                    assert decision.action is not None
                    before_observation_hash = _observation_hash(observation)
                    observation, reward, completed, info = env.step(decision.action)
                    final_score = int(info.get("score", final_score + reward))
                    valid_actions = list(info.get("valid", []))
                    after_observation_hash = _observation_hash(observation)
                    if (
                        not completed
                        and reward == 0
                        and decision.action != "wait1"
                    ):
                        no_progress_actions_by_state.setdefault(
                            before_observation_hash, set()
                        ).add(decision.action)
                    trajectory.append(
                        EnvironmentStep(
                            step_index=len(trajectory) + 1,
                            action=decision.action,
                            reward=reward,
                            score=final_score,
                            completed=completed,
                            valid_action_count=len(valid_actions),
                            observation_sha256=after_observation_hash,
                            expected_progress=decision.expected_progress,
                        )
                    )
                    if (
                        method == "iper_rag"
                        and budget.automatic_obligation_policy
                        == "unsupported_or_stalled"
                        and not completed
                        and reward == 0
                        and before_observation_hash == after_observation_hash
                        and decision.action != "wait1"
                    ):
                        obligation_key = (
                            f"execution_stall|{before_observation_hash}|{decision.action}"
                        )
                        if (
                            obligation_key not in automatic_obligation_keys
                            and retrieval_calls < budget.max_retrieval_calls
                            and automatic_obligation_count
                            < budget.max_automatic_obligations
                        ):
                            automatic_obligation_keys.add(obligation_key)
                            automatic_obligation_count += 1
                            query = self._automatic_obligation_query(
                                task_description=task_description,
                                observation=observation,
                                action=decision.action,
                                expected_progress=decision.expected_progress,
                                source="execution_stall",
                            )
                            hits = self.retriever.obligation_search(
                                query,
                                task_id,
                                budget.obligation_retrieval_k,
                                exclude_ids=set(retrieved_ids),
                            )
                            returned_ids = [hit.skill_fingerprint for hit in hits]
                            new_ids = [
                                skill_id
                                for skill_id in returned_ids
                                if skill_id not in retrieved_ids
                            ]
                            retrieved_ids.extend(new_ids)
                            retrieval_calls += 1
                            unresolved_obligations.append(query)
                            retrieval_events.append(
                                RetrievalEvent(
                                    call_index=retrieval_calls,
                                    query=query,
                                    trigger="obligation",
                                    returned_skill_ids=returned_ids,
                                    new_skill_ids=new_ids,
                                    obligation_source="execution_stall",
                                )
                            )
                    if completed:
                        status = (
                            "terminal_success" if final_score > 0 else "terminal_failure"
                        )
                        break

                if invalid_decisions >= budget.max_invalid_decisions:
                    status = "invalid_decision_limit"
                    break
        except Exception as exc:  # noqa: BLE001 - benchmark failures are outcomes
            status = "environment_error"
            error_type = type(exc).__name__
            error = str(exc)

        return self._result(
            method=method,
            task_id=task_id,
            task_name=task_name,
            split=split,
            variation=variation,
            seed=seed,
            status=status,
            completed=completed,
            final_score=final_score,
            model_calls=model_calls,
            retrieval_calls=retrieval_calls,
            invalid_decisions=invalid_decisions,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retrieved_ids=retrieved_ids,
            retrieval_events=retrieval_events,
            trajectory=trajectory,
            error_type=error_type,
            error=error,
            resolved_model_ids=sorted(resolved_model_ids),
            system_fingerprints=sorted(system_fingerprints),
            response_normalizations=response_normalizations,
            rejected_actions=rejected_actions,
            wall_seconds=time.perf_counter() - episode_started,
        )

    def _run_fixed_actions(
        self,
        *,
        env: ScienceWorldEnvironment,
        method: ScienceWorldMethod,
        task_id: str,
        task_name: str,
        split: Literal["dev", "test"],
        variation: int,
        seed: int,
        actions: list[str],
        initial_valid_actions: list[str],
        budget: EpisodeBudget,
    ) -> ScienceWorldEpisodeResult:
        episode_started = time.perf_counter()
        trajectory: list[EnvironmentStep] = []
        valid_actions = initial_valid_actions
        completed = False
        final_score = 0
        status: EpisodeStatus = "environment_budget_exhausted"
        for step_index, action in enumerate(
            actions[: budget.max_environment_steps], start=1
        ):
            observation, reward, completed, info = env.step(action)
            final_score = int(info.get("score", final_score + reward))
            valid_actions = list(info.get("valid", []))
            trajectory.append(
                EnvironmentStep(
                    step_index=step_index,
                    action=action,
                    reward=reward,
                    score=final_score,
                    completed=completed,
                    valid_action_count=len(valid_actions),
                    observation_sha256=_observation_hash(observation),
                )
            )
            if completed:
                status = "terminal_success" if final_score > 0 else "terminal_failure"
                break
        return self._result(
            method=method,
            task_id=task_id,
            task_name=task_name,
            split=split,
            variation=variation,
            seed=seed,
            status=status,
            completed=completed,
            final_score=final_score,
            model_calls=0,
            retrieval_calls=0,
            invalid_decisions=0,
            input_tokens=0,
            output_tokens=0,
            retrieved_ids=[],
            retrieval_events=[],
            trajectory=trajectory,
            error_type=None,
            error=None,
            wall_seconds=time.perf_counter() - episode_started,
        )

    def _action_request(
        self,
        *,
        method: ScienceWorldMethod,
        task_id: str,
        task_description: str,
        observation: str,
        info: dict[str, Any],
        valid_actions: list[str],
        retrieved_ids: list[str],
        trajectory: list[EnvironmentStep],
        unresolved_obligations: list[str],
        rejected_actions: list[str] | None = None,
        no_progress_actions: list[str] | None = None,
        automatic_obligation_policy: AutomaticObligationPolicy = "disabled",
        candidate_action_limit: int | None = None,
    ) -> LLMRequest:
        skills = [self.skills_by_id[skill_id] for skill_id in retrieved_ids]
        can_model_request_evidence = (
            method == "iper_rag" and automatic_obligation_policy == "disabled"
        )
        blocked_actions = set(rejected_actions or []) | set(no_progress_actions or [])
        available_actions = [
            action for action in valid_actions if action not in blocked_actions
        ]
        task_target = _explicit_task_target(task_description)
        if task_target:
            visible_state = f"{observation}\n{info.get('inv', '')}".casefold()
            target_is_visible = task_target.casefold() in visible_state

            def focus_action_is_eligible(action: str) -> bool:
                if not action.casefold().startswith("focus on "):
                    return True
                if not target_is_visible:
                    return False
                focus_object = action.removeprefix("focus on ").casefold()
                return task_target.casefold() in focus_object or (
                    "substance" in focus_object and focus_object in visible_state
                )

            available_actions = [
                action for action in available_actions if focus_action_is_eligible(action)
            ]
        if candidate_action_limit is not None and len(available_actions) > candidate_action_limit:
            target_tokens = _tokens(task_description)

            def action_priority(action: str) -> tuple[int, int, str]:
                action_tokens = _tokens(action)
                target_overlap = len(target_tokens & action_tokens)
                operator = _action_operator(action)
                exploration = int(
                    operator in {"go to", "look around", "look at", "look in", "open"}
                )
                return (-target_overlap, -exploration, action.casefold())

            available_actions = sorted(available_actions, key=action_priority)[
                :candidate_action_limit
            ]
        instruction = (
            "Act in ScienceWorld using only the supplied public environment state, valid "
            "actions, and train-split skills. Never use a test gold path. Choose an action "
            "exactly as written in valid_actions. Before returning act, verify that the "
            "action occurs verbatim in current.valid_actions. Never repeat an action listed "
            "in locally_rejected_actions or current.no_progress_actions; choose a listed "
            "navigation or exploration action that can change the state instead. Unknown "
            "is not false. Treat the target "
            "named in task_description as locked. A wrong focus action can irreversibly "
            "fail the episode: never focus on an object that differs from the named target. "
            "If the target is absent from the current observation, navigate or explore "
            "instead of focusing on a different object. Training skills are abstracted "
            "source-variation examples: bind <TASK_TARGET> and <FOCUS_OBJECT> to the "
            "current task and never copy a source-specific entity name. "
        )
        if can_model_request_evidence:
            instruction += (
                "If the available skills are insufficient for the next justified action, "
                "request one specific evidence query instead of guessing."
            )
        elif method == "iper_rag":
            instruction += (
                "An automatic evidence controller checks procedural support and execution "
                "stalls. Return act or stop, never request_evidence. Retrieval only searches "
                "train-split procedures and cannot reveal hidden current-variation locations."
            )
        else:
            instruction += "Evidence is fixed; return act or stop, never request_evidence."
        if method in {"b4_sequential_planner", "iper_rag"}:
            instruction += (
                " Explicitly compare the current state and recent trajectory with action "
                "preconditions before acting."
            )

        response_properties: dict[str, object] = {
            "decision_type": {
                "type": "string",
                "enum": (
                    ["act", "request_evidence", "stop"]
                    if can_model_request_evidence
                    else ["act", "stop"]
                ),
            },
            "action": {
                "anyOf": [
                    {"type": "string", "enum": available_actions},
                    {"type": "null"},
                ]
            },
            "expected_progress": {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            },
        }
        required_fields = ["decision_type", "action", "expected_progress"]
        if can_model_request_evidence:
            response_properties["evidence_query"] = {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            }
            required_fields.append("evidence_query")
        response_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": required_fields,
            "properties": response_properties,
        }
        return LLMRequest(
            purpose="environment_action",
            system_instruction=instruction,
            payload={
                "method": method,
                "task_id": task_id,
                "task_description": task_description,
                "current": {
                    "observation": observation,
                    "inventory": info.get("inv", ""),
                    "score": info.get("score", 0),
                    "valid_actions": available_actions,
                    "no_progress_actions": no_progress_actions or [],
                },
                "recent_trajectory": [
                    step.model_dump(mode="json") for step in trajectory[-6:]
                ],
                "training_skills": [
                    _skill_prompt_view(skill)
                    for skill in skills
                ],
                "unresolved_evidence_obligations": unresolved_obligations,
                "locally_rejected_actions": rejected_actions or [],
            },
            response_schema=response_schema,
        )

    def _result(
        self,
        *,
        method: ScienceWorldMethod,
        task_id: str,
        task_name: str,
        split: Literal["dev", "test"],
        variation: int,
        seed: int,
        status: EpisodeStatus,
        completed: bool,
        final_score: int,
        model_calls: int,
        retrieval_calls: int,
        invalid_decisions: int,
        input_tokens: int,
        output_tokens: int,
        retrieved_ids: list[str],
        retrieval_events: list[RetrievalEvent],
        trajectory: list[EnvironmentStep],
        error_type: str | None,
        error: str | None,
        resolved_model_ids: list[str] | None = None,
        system_fingerprints: list[str] | None = None,
        response_normalizations: dict[str, int] | None = None,
        rejected_actions: list[str] | None = None,
        wall_seconds: float = 0.0,
    ) -> ScienceWorldEpisodeResult:
        return ScienceWorldEpisodeResult(
            experiment_id=self.experiment_id,
            reporting_boundary=(
                "development_pilot_only"
                if split == "dev"
                else (
                    "sanity_or_oracle_only"
                    if method in {"random_sanity", "gold_oracle"}
                    else "formal_machine_result"
                )
            ),
            method=method,
            model_id=(
                self.adapter.model_id
                if self.adapter is not None
                and method not in {"random_sanity", "gold_oracle"}
                else None
            ),
            model_profile_id=(
                getattr(self.adapter, "profile_id", None)
                if self.adapter is not None
                and method not in {"random_sanity", "gold_oracle"}
                else None
            ),
            resolved_model_ids=resolved_model_ids or [],
            system_fingerprints=system_fingerprints or [],
            task_id=task_id,
            task_name=task_name,
            split=split,
            variation=variation,
            seed=seed,
            status=status,
            completed=completed,
            task_success=completed and final_score > 0,
            final_score=final_score,
            environment_steps=len(trajectory),
            model_calls=model_calls,
            retrieval_calls=retrieval_calls,
            invalid_decisions=invalid_decisions,
            rejected_actions=rejected_actions or [],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_seconds=wall_seconds,
            response_normalizations=response_normalizations or {},
            retrieved_skill_ids=retrieved_ids,
            retrieval_events=retrieval_events,
            trajectory=trajectory,
            error_type=error_type,
            error=error,
        )


def write_episode_result(result: ScienceWorldEpisodeResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def summarize_episode_results(
    results: list[ScienceWorldEpisodeResult],
) -> dict[str, object]:
    by_method: dict[str, list[ScienceWorldEpisodeResult]] = {}
    for result in results:
        by_method.setdefault(result.method, []).append(result)
    methods: dict[str, object] = {}
    for method, rows in sorted(by_method.items()):
        methods[method] = {
            "episodes": len(rows),
            "environment_terminal": sum(row.completed for row in rows),
            "task_successes": sum(row.task_success for row in rows),
            "task_success_rate": sum(row.task_success for row in rows) / len(rows),
            "mean_final_score": sum(row.final_score for row in rows) / len(rows),
            "mean_clipped_score": sum(max(0, row.final_score) for row in rows)
            / len(rows),
            "environment_steps": sum(row.environment_steps for row in rows),
            "model_calls": sum(row.model_calls for row in rows),
            "retrieval_calls": sum(row.retrieval_calls for row in rows),
            "input_tokens": sum(row.input_tokens for row in rows),
            "output_tokens": sum(row.output_tokens for row in rows),
            "wall_seconds": sum(row.wall_seconds for row in rows),
            "mean_wall_seconds": sum(row.wall_seconds for row in rows) / len(rows),
            "requested_model_ids": sorted(
                {row.model_id for row in rows if row.model_id is not None}
            ),
            "resolved_model_ids": sorted(
                {
                    model_id
                    for row in rows
                    for model_id in row.resolved_model_ids
                }
            ),
            "system_fingerprints": sorted(
                {
                    fingerprint
                    for row in rows
                    for fingerprint in row.system_fingerprints
                }
            ),
            "response_normalizations": {
                normalization: sum(
                    row.response_normalizations.get(normalization, 0) for row in rows
                )
                for normalization in sorted(
                    {
                        name
                        for row in rows
                        for name in row.response_normalizations
                    }
                )
            },
            "status_counts": {
                status: sum(row.status == status for row in rows)
                for status in sorted({row.status for row in rows})
            },
        }
    experiment_ids = sorted({result.experiment_id for result in results})
    return {
        "experiment_id": experiment_ids[0] if len(experiment_ids) == 1 else experiment_ids,
        "episode_count": len(results),
        "methods": methods,
    }
