from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from lanternquest.llm import LLMAdapter, LLMAdapterError, LLMRequest
from lanternquest.scienceworld_experiment import (
    EnvironmentStep,
    ScienceWorldSkill,
    _explicit_task_target,
    _skill_prompt_view,
    _tokens,
)

PublicBaselineMethod = Literal["react", "reflexion"]
PublicBaselineStatus = Literal[
    "terminal_success",
    "terminal_failure",
    "model_stop",
    "loop_detected",
    "attempts_exhausted",
    "environment_budget_exhausted",
    "model_budget_exhausted",
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


class PublicBaselineBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_environment_steps: int = Field(default=100, ge=1, le=500)
    max_model_calls: int = Field(default=100, ge=1, le=500)
    candidate_action_limit: int = Field(default=12, ge=1, le=100)
    demonstration_count: int = Field(default=2, ge=0, le=5)
    reflexion_max_attempts: int = Field(default=4, ge=1, le=10)
    recent_history_limit: int = Field(default=8, ge=1, le=30)


class PublicBaselineTrial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_index: int
    status: str
    score: int
    environment_steps: int
    model_calls: int
    reflection: str | None = None


class PublicBaselineEpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    experiment_id: str = "E72"
    reporting_boundary: str
    method: PublicBaselineMethod
    adaptation_label: str
    model_id: str
    model_profile_id: str | None = None
    resolved_model_ids: list[str] = Field(default_factory=list)
    system_fingerprints: list[str] = Field(default_factory=list)
    task_id: str
    task_name: str
    split: Literal["dev", "test"]
    variation: int
    seed: int
    status: PublicBaselineStatus
    completed: bool
    task_success: bool
    final_score: int
    environment_steps: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    invalid_decisions: int
    repeated_actions: int
    no_progress_actions: int
    wall_seconds: float
    demonstration_skill_ids: list[str]
    reflections: list[str]
    trials: list[PublicBaselineTrial]
    trajectory: list[EnvironmentStep]
    response_normalizations: dict[str, int] = Field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None


def _observation_hash(observation: str) -> str:
    return hashlib.sha256(observation.encode("utf-8")).hexdigest()


def _operator(action: str) -> str:
    words = action.casefold().split()
    if not words:
        return ""
    if len(words) >= 2 and words[0] in {"focus", "go", "look", "pick", "put"}:
        return " ".join(words[:2])
    return words[0]


def _candidate_actions(
    *,
    task_description: str,
    observation: str,
    inventory: str,
    valid_actions: list[str],
    limit: int,
) -> list[str]:
    """Apply the same target lock and deterministic shortlist used by the main harness."""
    available = list(dict.fromkeys(str(action) for action in valid_actions))
    task_target = _explicit_task_target(task_description)
    if task_target:
        visible_state = f"{observation}\n{inventory}".casefold()
        target_is_visible = task_target.casefold() in visible_state
        filtered: list[str] = []
        for action in available:
            if not action.casefold().startswith("focus on "):
                filtered.append(action)
                continue
            focus_object = action[len("focus on ") :].casefold()
            if target_is_visible and (
                task_target.casefold() in focus_object
                or ("substance" in focus_object and focus_object in visible_state)
            ):
                filtered.append(action)
        available = filtered

    if len(available) <= limit:
        return available
    target_tokens = _tokens(task_description)

    def priority(action: str) -> tuple[int, int, str]:
        overlap = len(target_tokens & _tokens(action))
        exploration = int(
            _operator(action) in {"go to", "look around", "look at", "look in", "open"}
        )
        return (-overlap, -exploration, action.casefold())

    return sorted(available, key=priority)[:limit]


class PublicBaselineHarness:
    """Compute-bounded ScienceWorld adaptations of public ReAct and Reflexion agents."""

    def __init__(
        self,
        skills: list[ScienceWorldSkill],
        adapter: LLMAdapter,
        *,
        experiment_id: str = "E72",
    ) -> None:
        self.skills = list(skills)
        self.adapter = adapter
        self.experiment_id = experiment_id

    def _demonstrations(
        self, task_id: str, count: int
    ) -> tuple[list[dict[str, object]], list[str]]:
        same_task = sorted(
            (skill for skill in self.skills if skill.task_id == task_id),
            key=lambda skill: (skill.variation, skill.skill_fingerprint),
        )
        selected = same_task[:count]
        return (
            [_skill_prompt_view(skill) for skill in selected],
            [skill.skill_fingerprint for skill in selected],
        )

    def _action_request(
        self,
        *,
        method: PublicBaselineMethod,
        task_id: str,
        task_description: str,
        observation: str,
        info: dict[str, Any],
        candidates: list[str],
        demonstrations: list[dict[str, object]],
        recent_history: list[dict[str, object]],
        reflections: list[str],
        trial_index: int,
    ) -> LLMRequest:
        instruction = (
            "Act in the public ScienceWorld environment as a ReAct agent. At each step, "
            "state one concise next subgoal and select exactly one action from "
            "current.valid_actions. Use the observation returned after each action to update "
            "the next subgoal. Do not claim hidden state, use a test gold path, or copy an "
            "entity from a training demonstration when it is absent from the current state. "
            "Treat the task target as locked because a wrong focus action can terminate the "
            "episode. Avoid repeating an action that produced no state change unless it is a "
            "necessary wait action. The next_subgoal field is a brief action rationale, not a "
            "private chain-of-thought transcript."
        )
        if method == "reflexion":
            instruction += (
                " You are in a Reflexion trial. Use the supplied verbal reflections from "
                "earlier failed trials as episodic memory and deliberately avoid the failed "
                "strategy they identify."
            )
        return LLMRequest(
            purpose="environment_action",
            system_instruction=instruction,
            payload={
                "method": method,
                "task_id": task_id,
                "task_description": task_description,
                "trial_index": trial_index,
                "current": {
                    "observation": observation,
                    "inventory": info.get("inv", ""),
                    "score": info.get("score", 0),
                    "valid_actions": candidates,
                },
                "recent_history": recent_history,
                "training_demonstrations": demonstrations,
                "episodic_reflections": reflections,
            },
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["decision_type", "next_subgoal", "action"],
                "properties": {
                    "decision_type": {
                        "type": "string",
                        "enum": ["act", "stop"],
                    },
                    "next_subgoal": {"type": "string"},
                    "action": {
                        "anyOf": [
                            {"type": "string", "enum": candidates},
                            {"type": "null"},
                        ]
                    },
                },
            },
        )

    def _reflection_request(
        self,
        *,
        task_description: str,
        trial_index: int,
        score: int,
        completed: bool,
        history: list[dict[str, object]],
        prior_reflections: list[str],
    ) -> LLMRequest:
        return LLMRequest(
            purpose="residual_repair",
            system_instruction=(
                "Produce one concise Reflexion memory after an unsuccessful ScienceWorld "
                "trial. Identify the most consequential failed strategy from the observable "
                "trajectory and give one concrete correction for the next trial. Do not infer "
                "hidden locations or use a gold path."
            ),
            payload={
                "task_description": task_description,
                "trial_index": trial_index,
                "terminal_score": score,
                "environment_completed": completed,
                "trajectory": history,
                "prior_reflections": prior_reflections,
            },
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["reflection"],
                "properties": {"reflection": {"type": "string"}},
            },
        )

    def run_episode(
        self,
        env: ScienceWorldEnvironment,
        *,
        method: PublicBaselineMethod,
        task_id: str,
        task_name: str,
        split: Literal["dev", "test"],
        variation: int,
        seed: int,
        budget: PublicBaselineBudget,
        simplifications: str = "",
    ) -> PublicBaselineEpisodeResult:
        started = time.perf_counter()
        demonstrations, demonstration_ids = self._demonstrations(
            task_id, budget.demonstration_count
        )
        max_attempts = 1 if method == "react" else budget.reflexion_max_attempts
        trajectory: list[EnvironmentStep] = []
        trials: list[PublicBaselineTrial] = []
        reflections: list[str] = []
        model_calls = 0
        input_tokens = 0
        output_tokens = 0
        invalid_decisions = 0
        repeated_actions = 0
        no_progress_actions = 0
        best_score = 0
        task_success = False
        completed = False
        status: PublicBaselineStatus = "attempts_exhausted"
        error_type: str | None = None
        error: str | None = None
        resolved_model_ids: set[str] = set()
        system_fingerprints: set[str] = set()
        normalizations: Counter[str] = Counter()

        try:
            env.load(
                task_name,
                variationIdx=variation,
                simplificationStr=simplifications,
                generateGoldPath=False,
            )
            for trial_index in range(1, max_attempts + 1):
                if model_calls >= budget.max_model_calls:
                    status = "model_budget_exhausted"
                    break
                if len(trajectory) >= budget.max_environment_steps:
                    status = "environment_budget_exhausted"
                    break

                observation, info = env.reset()
                trial_score = int(info.get("score", 0))
                task_description = str(info.get("taskDesc", task_name))
                valid_actions = [str(item) for item in info.get("valid", [])]
                trial_history: list[dict[str, object]] = []
                trial_model_start = model_calls
                trial_step_start = len(trajectory)
                trial_status = "step_budget_exhausted"
                remaining_attempts = max_attempts - trial_index + 1
                remaining_steps = budget.max_environment_steps - len(trajectory)
                trial_step_limit = (
                    remaining_steps
                    if method == "react"
                    else max(1, remaining_steps // remaining_attempts)
                )

                for _ in range(trial_step_limit):
                    if model_calls >= budget.max_model_calls:
                        status = "model_budget_exhausted"
                        trial_status = status
                        break
                    candidates = _candidate_actions(
                        task_description=task_description,
                        observation=observation,
                        inventory=str(info.get("inv", "")),
                        valid_actions=valid_actions,
                        limit=budget.candidate_action_limit,
                    )
                    if not candidates:
                        status = "environment_error"
                        trial_status = status
                        error = "Environment returned no admissible candidate actions"
                        break
                    request = self._action_request(
                        method=method,
                        task_id=task_id,
                        task_description=task_description,
                        observation=observation,
                        info=info,
                        candidates=candidates,
                        demonstrations=demonstrations,
                        recent_history=trial_history[-budget.recent_history_limit :],
                        reflections=reflections,
                        trial_index=trial_index,
                    )
                    try:
                        model_calls += 1
                        response = self.adapter.complete(request)
                        model_calls += max(0, response.provider_calls - 1)
                        input_tokens += response.input_tokens or 0
                        output_tokens += response.output_tokens or 0
                        resolved_model_ids.add(response.model_id)
                        if response.system_fingerprint:
                            system_fingerprints.add(response.system_fingerprint)
                        if response.normalization_applied:
                            normalizations[response.normalization_applied] += 1
                        decision_type = str(response.content.get("decision_type", ""))
                        action_value = response.content.get("action")
                        next_subgoal = str(response.content.get("next_subgoal", ""))
                    except Exception as exc:  # noqa: BLE001
                        if isinstance(exc, LLMAdapterError):
                            model_calls += max(0, exc.provider_calls - 1)
                            input_tokens += exc.input_tokens
                            output_tokens += exc.output_tokens
                        status = "adapter_error"
                        trial_status = status
                        error_type = type(exc).__name__
                        error = str(exc)
                        break

                    if decision_type == "stop":
                        trial_status = "model_stop"
                        break
                    if decision_type != "act" or not isinstance(action_value, str):
                        invalid_decisions += 1
                        trial_history.append(
                            {
                                "next_subgoal": next_subgoal,
                                "action": action_value,
                                "feedback": "invalid structured decision",
                            }
                        )
                        continue
                    action = action_value
                    if action not in candidates:
                        invalid_decisions += 1
                        trial_history.append(
                            {
                                "next_subgoal": next_subgoal,
                                "action": action,
                                "feedback": "action was not in valid_actions",
                            }
                        )
                        continue
                    prior_trial_actions = {
                        str(item["action"])
                        for item in trial_history
                        if isinstance(item.get("action"), str)
                    }
                    if action in prior_trial_actions:
                        repeated_actions += 1
                    before_hash = _observation_hash(observation)
                    observation, reward, completed, info = env.step(action)
                    trial_score = int(info.get("score", trial_score + int(reward)))
                    valid_actions = [str(item) for item in info.get("valid", [])]
                    after_hash = _observation_hash(observation)
                    no_progress = not completed and int(reward) == 0 and before_hash == after_hash
                    if no_progress and action != "wait1":
                        no_progress_actions += 1
                    trial_history.append(
                        {
                            "next_subgoal": next_subgoal,
                            "action": action,
                            "reward": int(reward),
                            "score": trial_score,
                            "completed": bool(completed),
                            "observation_changed": before_hash != after_hash,
                        }
                    )
                    trajectory.append(
                        EnvironmentStep(
                            step_index=len(trajectory) + 1,
                            action=action,
                            reward=int(reward),
                            score=trial_score,
                            completed=bool(completed),
                            valid_action_count=len(valid_actions),
                            observation_sha256=after_hash,
                            expected_progress=next_subgoal or None,
                        )
                    )
                    best_score = max(best_score, trial_score)
                    if completed:
                        if trial_score > 0:
                            task_success = True
                            status = "terminal_success"
                            trial_status = status
                        else:
                            trial_status = "terminal_failure"
                        break
                    recent_actions = [
                        str(item["action"])
                        for item in trial_history[-5:]
                        if isinstance(item.get("action"), str)
                    ]
                    if len(recent_actions) == 5 and len(set(recent_actions)) <= 2:
                        trial_status = "loop_detected"
                        break

                reflection_text: str | None = None
                can_reflect = (
                    method == "reflexion"
                    and not task_success
                    and trial_index < max_attempts
                    and model_calls < budget.max_model_calls
                )
                if can_reflect:
                    try:
                        model_calls += 1
                        response = self.adapter.complete(
                            self._reflection_request(
                                task_description=task_description,
                                trial_index=trial_index,
                                score=trial_score,
                                completed=completed,
                                history=trial_history,
                                prior_reflections=reflections,
                            )
                        )
                        model_calls += max(0, response.provider_calls - 1)
                        input_tokens += response.input_tokens or 0
                        output_tokens += response.output_tokens or 0
                        resolved_model_ids.add(response.model_id)
                        if response.system_fingerprint:
                            system_fingerprints.add(response.system_fingerprint)
                        if response.normalization_applied:
                            normalizations[response.normalization_applied] += 1
                        reflection_text = str(response.content["reflection"]).strip()
                        if reflection_text:
                            reflections.append(reflection_text)
                    except Exception as exc:  # noqa: BLE001
                        if isinstance(exc, LLMAdapterError):
                            model_calls += max(0, exc.provider_calls - 1)
                            input_tokens += exc.input_tokens
                            output_tokens += exc.output_tokens
                        status = "adapter_error"
                        trial_status = status
                        error_type = type(exc).__name__
                        error = str(exc)

                trials.append(
                    PublicBaselineTrial(
                        trial_index=trial_index,
                        status=trial_status,
                        score=trial_score,
                        environment_steps=len(trajectory) - trial_step_start,
                        model_calls=model_calls - trial_model_start,
                        reflection=reflection_text,
                    )
                )
                if task_success or status in {
                    "adapter_error",
                    "environment_error",
                    "model_budget_exhausted",
                }:
                    break
                if method == "react":
                    if trial_status == "terminal_failure":
                        status = "terminal_failure"
                    elif trial_status == "model_stop":
                        status = "model_stop"
                    elif trial_status == "loop_detected":
                        status = "loop_detected"
                    else:
                        status = "environment_budget_exhausted"
                    break
                completed = False

            if method == "reflexion" and not task_success and status == "attempts_exhausted":
                if len(trajectory) >= budget.max_environment_steps:
                    status = "environment_budget_exhausted"
                elif model_calls >= budget.max_model_calls:
                    status = "model_budget_exhausted"
                elif trials and all(trial.status == "terminal_failure" for trial in trials):
                    status = "terminal_failure"
        except Exception as exc:  # noqa: BLE001
            status = "environment_error"
            error_type = type(exc).__name__
            error = str(exc)

        return PublicBaselineEpisodeResult(
            experiment_id=self.experiment_id,
            reporting_boundary=(
                "development_only" if split == "dev" else "formal_external_baseline"
            ),
            method=method,
            adaptation_label=(
                "compute-matched ReAct adaptation"
                if method == "react"
                else "compute-matched four-trial Reflexion adaptation"
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
            completed=completed,
            task_success=task_success,
            final_score=best_score,
            environment_steps=len(trajectory),
            model_calls=model_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            invalid_decisions=invalid_decisions,
            repeated_actions=repeated_actions,
            no_progress_actions=no_progress_actions,
            wall_seconds=time.perf_counter() - started,
            demonstration_skill_ids=demonstration_ids,
            reflections=reflections,
            trials=trials,
            trajectory=trajectory,
            response_normalizations=dict(sorted(normalizations.items())),
            error_type=error_type,
            error=error,
        )


def write_public_baseline_result(
    result: PublicBaselineEpisodeResult, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
