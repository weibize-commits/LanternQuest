from lanternquest.planning import (
    EvidenceHit,
    PlanningCase,
    ReplanningEngine,
    SearchBudget,
)


class MappingRetriever:
    def __init__(self, responses: dict[str, list[EvidenceHit]]) -> None:
        self.responses = responses
        self.queries: list[str] = []

    def retrieve(self, query: str, limit: int) -> list[EvidenceHit]:
        self.queries.append(query)
        return self.responses.get(query, [])[:limit]


def make_case(*, state_value: str = "true", locked: bool = False) -> PlanningCase:
    return PlanningCase.model_validate(
        {
            "case_id": "synthetic_case",
            "goal_statement": "finish sample",
            "user_question": "how should I continue",
            "state_version": "state_v1",
            "intent_version": "intent_v1",
            "state": [
                {
                    "predicate": "material_ready",
                    "value": state_value,
                    "origin": "test fixture",
                },
                {
                    "predicate": "sample_finished",
                    "value": "false",
                    "origin": "test fixture",
                },
            ],
            "goal": [{"predicate": "sample_finished", "value": "true"}],
            "intents": [
                {
                    "intent_id": "keep_pattern",
                    "statement": "keep the chosen pattern",
                    "priority": 3,
                    "locked": locked,
                }
            ],
            "completed_work_ids": ["draft_pattern"],
            "actions": [
                {
                    "action_id": "finish_without_change",
                    "preconditions": [
                        {"predicate": "material_ready", "value": "true"}
                    ],
                    "effects": [
                        {"predicate": "sample_finished", "value": "true"}
                    ],
                    "evidence_obligations": [
                        {"claim_id": "claim_finish", "query": "evidence for finish"}
                    ],
                    "action_cost": 2,
                },
                {
                    "action_id": "replace_pattern",
                    "preconditions": [
                        {"predicate": "material_ready", "value": "true"}
                    ],
                    "effects": [
                        {"predicate": "sample_finished", "value": "true"}
                    ],
                    "evidence_obligations": [
                        {"claim_id": "claim_finish", "query": "evidence for finish"}
                    ],
                    "changes_intent_ids": ["keep_pattern"],
                    "invalidates_work_ids": ["draft_pattern"],
                    "action_cost": 1,
                },
            ],
        }
    )


def test_iper_retrieves_for_path_obligation_while_sequential_baseline_stops() -> None:
    hit = EvidenceHit("ev_finish", ("claim_finish",))
    responses = {"evidence for finish": [hit]}
    budget = SearchBudget(max_retrieval_calls=3)

    baseline_retriever = MappingRetriever(responses)
    baseline = ReplanningEngine(baseline_retriever).run(
        make_case(), budget, method="b4_sequential"
    )
    assert baseline.decision_type == "evidence_insufficient"
    assert baseline.unresolved_conditions == ["claim:claim_finish"]

    iper_retriever = MappingRetriever(responses)
    proposed = ReplanningEngine(iper_retriever).run(
        make_case(), budget, method="iper_rag"
    )
    assert proposed.decision_type == "propose"
    assert proposed.plans[0].action_ids == ["finish_without_change"]
    assert proposed.plans[0].evidence_ids == ["ev_finish"]
    assert "evidence for finish" in iper_retriever.queries


def test_unknown_precondition_requests_clarification() -> None:
    decision = ReplanningEngine(MappingRetriever({})).run(
        make_case(state_value="unknown"), SearchBudget(), method="iper_rag"
    )

    assert decision.decision_type == "clarify"
    assert decision.unresolved_conditions == ["state:material_ready"]


def test_locked_intent_cannot_be_changed_without_consent() -> None:
    hit = EvidenceHit("ev_finish", ("claim_finish",))
    case = make_case(locked=True)
    case.actions = [action for action in case.actions if action.action_id == "replace_pattern"]

    decision = ReplanningEngine(
        MappingRetriever({"evidence for finish": [hit]})
    ).run(case, SearchBudget(), method="iper_rag")

    assert decision.decision_type == "conflict"
    assert decision.unresolved_conditions == ["locked_intent:keep_pattern"]


def test_lower_intent_and_rework_cost_wins_before_action_cost() -> None:
    hit = EvidenceHit("ev_finish", ("claim_finish",))
    decision = ReplanningEngine(
        MappingRetriever({"evidence for finish": [hit]})
    ).run(make_case(), SearchBudget(), method="iper_rag")

    assert [plan.action_ids for plan in decision.plans] == [
        ["finish_without_change"],
        ["replace_pattern"],
    ]
    assert decision.plans[0].intent_deviation_cost == 0
    assert decision.plans[1].intent_deviation_cost == 3
    assert decision.plans[1].rework_cost == 1


def test_dominated_path_to_same_label_is_pruned() -> None:
    hit = EvidenceHit("ev_finish", ("claim_finish",))
    case = make_case()
    cheaper = case.actions[0].model_copy(
        update={"action_id": "finish_cheaper", "action_cost": 1.0}
    )
    expensive = case.actions[0].model_copy(
        update={"action_id": "finish_expensive", "action_cost": 5.0}
    )
    case.actions = [expensive, cheaper]

    decision = ReplanningEngine(
        MappingRetriever({"evidence for finish": [hit]})
    ).run(case, SearchBudget(), method="iper_rag")

    assert [plan.action_ids for plan in decision.plans] == [["finish_cheaper"]]
