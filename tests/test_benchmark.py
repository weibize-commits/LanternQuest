import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from lanternquest.benchmark import (
    LanternQuestBenchmarkCase,
    benchmark_fingerprint,
    load_benchmark_jsonl,
)


def make_benchmark_case(index: int = 1) -> dict:
    source_id = f"src_{index:016x}"
    obligation_id = f"claim_{index}"
    evidence_id = f"evidence_{index}"
    action_id = f"action_{index}"
    return {
        "schema_version": "1.0",
        "case_id": f"lq_case_{index}",
        "scenario_family": f"family_{index}",
        "split": "test",
        "task_id": f"task_{index}",
        "title": "Prepare the artifact",
        "goal": "Complete the evidence-grounded action",
        "user_question": "What should happen next?",
        "state_version": "state_v1",
        "intent_version": "intent_v1",
        "initial_state": [
            {
                "fact_id": f"fact_{index}",
                "predicate": "ready",
                "value": "true",
                "origin": "expert review",
                "source_ids": [source_id],
            }
        ],
        "locked_intents": [
            {
                "intent_id": "preserve_style",
                "description": "Preserve the selected visual style",
                "priority": 3,
                "locked": True,
            }
        ],
        "completed_work_ids": ["painted_background"],
        "decision_nodes": [],
        "action_space": [
            {
                "action_id": action_id,
                "label": "Apply the grounded action",
                "preconditions": [{"predicate": "ready", "value": "true"}],
                "effects": [{"predicate": "done", "value": "true"}],
                "evidence_obligation_ids": [obligation_id],
                "changes_intent_ids": [],
                "invalidates_work_ids": [],
                "action_cost": 1.0,
            }
        ],
        "evidence_obligations": [
            {
                "obligation_id": obligation_id,
                "query": "Find support for the action",
                "satisfied_by_evidence_ids": [evidence_id],
            }
        ],
        "evidence_catalog": [
            {
                "evidence_id": evidence_id,
                "source_id": source_id,
                "text": "The reviewed source supports this action.",
                "supported_claim_ids": [obligation_id],
                "review_status": "domain_approved",
            }
        ],
        "expected_decision_type": "propose",
        "expected_unresolved_conditions": [],
        "gold_plan": [action_id],
        "terminal_conditions": {
            "success": [{"predicate": "done", "value": "true"}],
            "failure": [{"predicate": "damaged", "value": "true"}],
        },
        "source_ids": [source_id],
        "canonical_content_sha256": [f"{index:064x}"],
        "rights": {
            "research_use_allowed": True,
            "external_processing_allowed": False,
            "review_status": "frozen",
        },
        "review": {
            "domain_status": "frozen",
            "annotation_status": "frozen",
            "reviewed_at": "2026-09-23T12:00:00Z",
            "reviewer_ids": ["domain_1", "annotator_1", "annotator_2"],
        },
    }


def test_benchmark_case_converts_to_existing_experiment_fixture() -> None:
    case = LanternQuestBenchmarkCase.model_validate(make_benchmark_case())

    fixture = case.to_fixture()

    assert fixture.purpose == "frozen_evaluation"
    assert fixture.case.goal[0].predicate == "done"
    assert fixture.case.actions[0].evidence_obligations[0].claim_id == "claim_1"
    assert fixture.evidence_catalog[0].supported_claim_ids == ["claim_1"]


def test_benchmark_case_rejects_unknown_evidence_reference() -> None:
    payload = make_benchmark_case()
    payload["evidence_obligations"][0]["satisfied_by_evidence_ids"] = ["unknown"]

    with pytest.raises(ValidationError, match="unknown evidence"):
        LanternQuestBenchmarkCase.model_validate(payload)


def test_loader_and_fingerprint_are_order_invariant(tmp_path: Path) -> None:
    payloads = [make_benchmark_case(2), make_benchmark_case(1)]
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
        encoding="utf-8",
    )

    cases = load_benchmark_jsonl(path)
    reversed_cases = [
        LanternQuestBenchmarkCase.model_validate(item)
        for item in reversed(deepcopy(payloads))
    ]

    assert benchmark_fingerprint(cases) == benchmark_fingerprint(reversed_cases)


def test_loader_rejects_duplicate_case_identifiers(tmp_path: Path) -> None:
    payload = make_benchmark_case()
    path = tmp_path / "duplicates.jsonl"
    line = json.dumps(payload, ensure_ascii=False)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case"):
        load_benchmark_jsonl(path)
