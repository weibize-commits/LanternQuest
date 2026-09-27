from copy import deepcopy

from lanternquest.benchmark import LanternQuestBenchmarkCase
from lanternquest.e21_methods import prepare_e21_method, verify_plan
from lanternquest.planning import PlanCandidate, SearchBudget
from tests.test_benchmark import make_benchmark_case


def test_b2_uses_fixed_graph_links_when_lexical_text_does_not_match() -> None:
    payload = make_benchmark_case()
    payload["evidence_obligations"][0]["query"] = "needle piercing sequence"
    payload["evidence_catalog"][0]["text"] = "completely unrelated vocabulary"
    case = LanternQuestBenchmarkCase.model_validate(payload)

    result = prepare_e21_method(
        case,
        "b2_kg_rag",
        SearchBudget(initial_evidence_limit=5),
    )

    assert result.status == "deterministic_plan_ready"
    assert result.retrieved_evidence_ids == ["evidence_1"]
    assert result.decision is not None
    assert result.decision.method == "b2_kg_rag"
    assert result.decision.plans[0].action_ids == ["action_1"]
    assert result.mechanism_flags["fixed_graph_retrieval"] is True


def test_ensr_runs_obligation_retrieval_and_symbolic_verification() -> None:
    case = LanternQuestBenchmarkCase.model_validate(make_benchmark_case())

    result = prepare_e21_method(case, "ensr", SearchBudget())

    assert result.decision is not None
    assert result.decision.method == "ensr"
    assert result.decision.decision_type == "propose"
    assert result.verifier_reports[0].valid is True
    assert result.mechanism_flags == {
        "fixed_graph_retrieval": False,
        "obligation_conditioned_retrieval": True,
        "symbolic_transition_verifier": True,
        "locked_intent_gate": True,
    }


def test_verifier_rejects_locked_intent_change() -> None:
    payload = deepcopy(make_benchmark_case())
    payload["action_space"][0]["changes_intent_ids"] = ["preserve_style"]
    case = LanternQuestBenchmarkCase.model_validate(payload)
    fixture = case.to_fixture()
    plan = PlanCandidate(
        action_ids=["action_1"],
        retained_intent_ids=[],
        proposed_changes=["preserve_style"],
        required_consent=["preserve_style"],
        evidence_ids=["evidence_1"],
        unresolved_conditions=[],
        intent_deviation_cost=3,
        rework_cost=0,
        retrieval_calls=1,
        action_cost=1.0,
    )

    report = verify_plan(
        fixture.case,
        plan,
        {"evidence_1": {"claim_1"}},
    )

    assert report.valid is False
    assert report.locked_intents_preserved is False
    assert report.changed_locked_intent_ids == ["preserve_style"]
