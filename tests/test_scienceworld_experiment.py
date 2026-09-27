from pathlib import Path

from lanternquest.llm import LLMRequest, LLMResponse
from lanternquest.scienceworld_experiment import (
    EpisodeBudget,
    ScienceWorldExperimentHarness,
    ScienceWorldSkill,
    ScienceWorldSkillRetriever,
    summarize_episode_results,
)


def make_skill(task_id: str, task_name: str, description: str) -> ScienceWorldSkill:
    return ScienceWorldSkill(
        task_id=task_id,
        task_name=task_name,
        variation=0,
        task_description=description,
        gold_actions=["look around", "finish"],
        skill_fingerprint=f"fingerprint-{task_id}",
    )


class FakeEnvironment:
    def __init__(self) -> None:
        self.score = 0

    def load(self, *args: object, **kwargs: object) -> None:
        self.score = 0

    def reset(self) -> tuple[str, dict[str, object]]:
        return "A closed task.", {
            "score": 0,
            "taskDesc": "Task Description: finish the task",
            "inv": "empty",
            "valid": ["look around", "finish"],
        }

    def step(self, action: str) -> tuple[str, int, bool, dict[str, object]]:
        completed = action == "finish"
        reward = 100 if completed else 0
        self.score += reward
        return "done" if completed else "same", reward, completed, {
            "score": self.score,
            "valid": ["look around", "finish"],
            "inv": "empty",
        }

    def get_gold_action_sequence(self) -> list[str]:
        return ["finish"]


class FakeAdapter:
    model_id = "fake"

    def __init__(self, contents: list[dict[str, object]]) -> None:
        self.contents = contents
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            model_id=self.model_id,
            content=self.contents.pop(0),
            input_tokens=10,
            output_tokens=4,
        )


def test_skill_retrieval_is_deterministic() -> None:
    skills = [
        make_skill("1-1", "boil", "boil water with heat"),
        make_skill("5-1", "grow-plant", "grow a plant with water"),
    ]
    retriever = ScienceWorldSkillRetriever(skills)

    first = retriever.search("boil water", 1)
    second = retriever.search("boil water", 1)

    assert first == second
    assert first[0].task_id == "1-1"


def test_obligation_retrieval_excludes_already_seen_skills() -> None:
    skills = [
        make_skill("1-1", "boil", "boil water with heat"),
        make_skill("1-2", "melt", "melt ice with heat"),
    ]
    retriever = ScienceWorldSkillRetriever(skills)

    hits = retriever.obligation_search(
        "boil water",
        "1-1",
        1,
        exclude_ids={skills[0].skill_fingerprint},
    )

    assert [hit.skill_fingerprint for hit in hits] == [skills[1].skill_fingerprint]


def test_locked_target_filters_focus_actions_until_target_is_visible() -> None:
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "boil a target")]
    )
    request = harness._action_request(
        method="b4_sequential_planner",
        task_id="1-1",
        task_description="Your task is to boil rubber.",
        observation="A hallway containing air and a door to the kitchen.",
        info={"inv": "empty", "score": 0},
        valid_actions=["focus on air", "focus on rubber", "open door to kitchen"],
        retrieved_ids=[],
        trajectory=[],
        unresolved_obligations=[],
    )

    assert request.payload["current"]["valid_actions"] == [
        "open door to kitchen"
    ]


def test_candidate_action_limit_uses_deterministic_target_aware_ranking() -> None:
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "boil a target")]
    )
    request = harness._action_request(
        method="iper_rag",
        task_id="1-1",
        task_description="Your task is to boil rubber.",
        observation="Rubber is visible in the kitchen.",
        info={"inv": "empty", "score": 0},
        valid_actions=[
            "wait1",
            "look around",
            "pick up rubber",
            "open cupboard",
        ],
        retrieved_ids=[],
        trajectory=[],
        unresolved_obligations=[],
        candidate_action_limit=2,
    )

    actions = request.payload["current"]["valid_actions"]
    assert actions == ["pick up rubber", "look around"]
    assert request.response_schema["properties"]["action"]["anyOf"][0]["enum"] == actions


def test_source_variation_target_is_abstracted_in_prompt() -> None:
    source_skill = ScienceWorldSkill(
        task_id="1-4",
        task_name="change-the-state-of-matter-of",
        variation=0,
        task_description="Your task is to change the state of matter of orange juice.",
        gold_actions=["focus on orange juice", "move orange juice to metal pot"],
        skill_fingerprint="source-orange",
    )
    adapter = FakeAdapter(
        [
            {
                "decision_type": "stop",
                "action": None,
                "evidence_query": None,
                "expected_progress": None,
            }
        ]
    )
    harness = ScienceWorldExperimentHarness([source_skill], adapter=adapter)

    harness.run_episode(
        FakeEnvironment(),
        method="b0_full_skills",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(max_environment_steps=1, max_model_calls=1),
    )

    prompt_skill = adapter.requests[0].payload["training_skills"][0]
    prompt_text = str(prompt_skill)
    assert "orange juice" not in prompt_text
    assert "<TASK_TARGET>" in prompt_text


def test_static_planner_completes_with_valid_action() -> None:
    adapter = FakeAdapter(
        [
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": None,
                "expected_progress": "complete",
            }
        ]
    )
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "finish the task")], adapter=adapter
    )

    result = harness.run_episode(
        FakeEnvironment(),
        method="b4_sequential_planner",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(max_environment_steps=3, max_model_calls=3),
    )

    assert result.status == "terminal_success"
    assert result.task_success is True
    assert result.final_score == 100
    assert result.model_calls == 1
    assert result.retrieval_calls == 1
    assert result.input_tokens == 10
    assert result.model_id == "fake"
    assert result.resolved_model_ids == ["fake"]
    assert result.reporting_boundary == "development_pilot_only"
    assert "target named in task_description as locked" in (
        adapter.requests[0].system_instruction
    )


def test_followup_experiment_id_is_preserved_in_episode_and_summary() -> None:
    adapter = FakeAdapter(
        [
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": None,
                "expected_progress": "complete",
            }
        ]
    )
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "finish the task")],
        adapter=adapter,
        experiment_id="E11F",
    )
    result = harness.run_episode(
        FakeEnvironment(),
        method="iper_rag",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(max_environment_steps=1, max_model_calls=1),
    )

    assert result.experiment_id == "E11F"
    assert summarize_episode_results([result])["experiment_id"] == "E11F"


def test_iper_can_fill_evidence_obligation_before_acting() -> None:
    adapter = FakeAdapter(
        [
            {
                "decision_type": "request_evidence",
                "action": None,
                "evidence_query": "grow plant water",
                "expected_progress": None,
            },
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": None,
                "expected_progress": "complete",
            },
        ]
    )
    skills = [
        make_skill("1-1", "boil", "finish the task"),
        make_skill("5-1", "grow-plant", "grow plant water"),
    ]
    harness = ScienceWorldExperimentHarness(skills, adapter=adapter)

    result = harness.run_episode(
        FakeEnvironment(),
        method="iper_rag",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(
            max_environment_steps=3,
            max_model_calls=3,
            initial_retrieval_k=1,
            obligation_retrieval_k=1,
        ),
    )

    assert result.status == "terminal_success"
    assert result.task_success is True
    assert result.model_calls == 2
    assert result.retrieval_calls == 2
    assert len(result.retrieval_events) == 2
    assert result.retrieval_events[-1].trigger == "obligation"
    assert result.retrieval_events[-1].obligation_source == "model_request"


def test_iper_automatically_retrieves_for_unsupported_action() -> None:
    adapter = FakeAdapter(
        [
            {
                "decision_type": "act",
                "action": "activate stove",
                "evidence_query": None,
                "expected_progress": "enter the work area",
            },
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": None,
                "expected_progress": "complete",
            },
        ]
    )
    skill = make_skill("1-1", "boil", "finish the task")
    harness = ScienceWorldExperimentHarness([skill], adapter=adapter)

    class DoorEnvironment(FakeEnvironment):
        def reset(self) -> tuple[str, dict[str, object]]:
            observation, info = super().reset()
            info["valid"] = ["activate stove", "finish"]
            return observation, info

    result = harness.run_episode(
        DoorEnvironment(),
        method="iper_rag",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(
            max_environment_steps=3,
            max_model_calls=3,
            automatic_obligation_policy="unsupported_or_stalled",
        ),
    )

    assert result.status == "terminal_success"
    assert result.model_calls == 2
    assert result.retrieval_calls == 2
    assert result.retrieval_events[-1].trigger == "obligation"
    assert result.retrieval_events[-1].obligation_source == "unsupported_action"
    assert result.trajectory[0].action == "finish"


def test_automatic_controller_disables_free_form_evidence_requests() -> None:
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "finish the task")]
    )
    request = harness._action_request(
        method="iper_rag",
        task_id="1-1",
        task_description="Your task is to boil water.",
        observation="hallway",
        info={"inv": "empty", "score": 0},
        valid_actions=["look around", "finish"],
        retrieved_ids=[],
        trajectory=[],
        unresolved_obligations=[],
        automatic_obligation_policy="unsupported_or_stalled",
    )

    assert request.response_schema["properties"]["decision_type"]["enum"] == [
        "act",
        "stop",
    ]
    assert "evidence_query" not in request.response_schema["properties"]
    assert "cannot reveal hidden current-variation locations" in (
        request.system_instruction
    )


def test_iper_automatically_retrieves_after_execution_stall() -> None:
    adapter = FakeAdapter(
        [
            {
                "decision_type": "act",
                "action": "look around",
                "evidence_query": None,
                "expected_progress": "inspect",
            },
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": None,
                "expected_progress": "complete",
            },
        ]
    )
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "finish the task")], adapter=adapter
    )

    class StalledEnvironment(FakeEnvironment):
        def reset(self) -> tuple[str, dict[str, object]]:
            _, info = super().reset()
            return "same", info

    result = harness.run_episode(
        StalledEnvironment(),
        method="iper_rag",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(
            max_environment_steps=3,
            max_model_calls=3,
            automatic_obligation_policy="unsupported_or_stalled",
        ),
    )

    assert result.status == "terminal_success"
    assert result.retrieval_calls == 2
    assert result.retrieval_events[-1].obligation_source == "execution_stall"


def test_invalid_action_is_rejected_and_logged() -> None:
    adapter = FakeAdapter(
        [
            {
                "decision_type": "act",
                "action": "focus on missing object",
                "evidence_query": None,
                "expected_progress": "try an unavailable action",
            },
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": None,
                "expected_progress": "complete",
            },
        ]
    )
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "finish the task")], adapter=adapter
    )

    result = harness.run_episode(
        FakeEnvironment(),
        method="b4_sequential_planner",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(max_environment_steps=3, max_model_calls=3),
    )

    assert result.status == "terminal_success"
    assert result.invalid_decisions == 1
    assert result.rejected_actions == ["focus on missing object"]
    assert adapter.requests[1].payload["locally_rejected_actions"] == [
        "focus on missing object"
    ]


def test_irrelevant_evidence_query_is_cleared_for_act_decision() -> None:
    adapter = FakeAdapter(
        [
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": "unused query",
                "expected_progress": "complete",
            }
        ]
    )
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "finish the task")], adapter=adapter
    )

    result = harness.run_episode(
        FakeEnvironment(),
        method="b4_sequential_planner",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(max_environment_steps=2, max_model_calls=2),
    )

    assert result.status == "terminal_success"
    assert result.response_normalizations == {
        "cleared_irrelevant_decision_fields": 1
    }


def test_no_progress_action_is_not_executed_twice_in_same_state() -> None:
    class StableFakeEnvironment(FakeEnvironment):
        def reset(self) -> tuple[str, dict[str, object]]:
            _, info = super().reset()
            return "same", info

    adapter = FakeAdapter(
        [
            {
                "decision_type": "act",
                "action": "look around",
                "evidence_query": None,
                "expected_progress": "inspect",
            },
            {
                "decision_type": "act",
                "action": "look around",
                "evidence_query": None,
                "expected_progress": "inspect again",
            },
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": None,
                "expected_progress": "complete",
            },
        ]
    )
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "finish the task")], adapter=adapter
    )

    result = harness.run_episode(
        StableFakeEnvironment(),
        method="b4_sequential_planner",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(max_environment_steps=3, max_model_calls=4),
    )

    assert result.status == "terminal_success"
    assert [step.action for step in result.trajectory] == ["look around", "finish"]
    assert result.invalid_decisions == 1
    assert adapter.requests[1].payload["current"]["no_progress_actions"] == [
        "look around"
    ]
    assert "look around" not in adapter.requests[1].payload["current"][
        "valid_actions"
    ]


def test_zero_reward_transition_is_not_repeated_after_returning_to_state() -> None:
    class AlternatingEnvironment(FakeEnvironment):
        def reset(self) -> tuple[str, dict[str, object]]:
            _, info = super().reset()
            info["valid"] = ["look at drawing", "look around", "finish"]
            return "room", info

        def step(self, action: str) -> tuple[str, int, bool, dict[str, object]]:
            if action == "finish":
                return "done", 100, True, {
                    "score": 100,
                    "valid": ["look at drawing", "look around", "finish"],
                    "inv": "empty",
                }
            observation = "drawing" if action == "look at drawing" else "room"
            return observation, 0, False, {
                "score": 0,
                "valid": ["look at drawing", "look around", "finish"],
                "inv": "empty",
            }

    adapter = FakeAdapter(
        [
            {
                "decision_type": "act",
                "action": "look at drawing",
                "evidence_query": None,
                "expected_progress": "inspect",
            },
            {
                "decision_type": "act",
                "action": "look around",
                "evidence_query": None,
                "expected_progress": "return",
            },
            {
                "decision_type": "act",
                "action": "look at drawing",
                "evidence_query": None,
                "expected_progress": "repeat",
            },
            {
                "decision_type": "act",
                "action": "finish",
                "evidence_query": None,
                "expected_progress": "complete",
            },
        ]
    )
    harness = ScienceWorldExperimentHarness(
        [make_skill("1-1", "boil", "finish the task")], adapter=adapter
    )

    result = harness.run_episode(
        AlternatingEnvironment(),
        method="b4_sequential_planner",
        task_id="1-1",
        task_name="boil",
        split="dev",
        variation=1,
        seed=11,
        budget=EpisodeBudget(max_environment_steps=4, max_model_calls=5),
    )

    assert result.status == "terminal_success"
    assert [step.action for step in result.trajectory] == [
        "look at drawing",
        "look around",
        "finish",
    ]
    assert result.invalid_decisions == 1


def test_real_skill_artifacts_load_when_present() -> None:
    skill_dir = Path("artifacts/benchmarks/scienceworld/train_skills")
    if not skill_dir.is_dir():
        return
    from lanternquest.scienceworld_experiment import load_scienceworld_skills

    skills = load_scienceworld_skills(skill_dir)
    assert len(skills) == 30
    assert all(skill.gold_actions for skill in skills)
