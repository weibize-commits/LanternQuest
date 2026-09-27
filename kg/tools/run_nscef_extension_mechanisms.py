from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kg.tools.kg_evidence_retriever import KGEvidenceRetriever, load_graph
from lanternquest.planning import PlanningCase, ReplanningEngine, SearchBudget

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "kg" / "data" / "dev_multimodal_neurosymbolic_graph.json"
DEFAULT_CASES = ROOT / "annotations" / "eval" / "nscef_extension_mechanisms_v0.json"
DEFAULT_OUTPUT = ROOT / "kg" / "data" / "nscef_extension_mechanisms_v0_results.json"


def run_cases(graph: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    rows = []
    all_expected = True
    for case_payload in fixture["cases"]:
        expected = case_payload["expected"]
        case = PlanningCase.model_validate(
            {key: value for key, value in case_payload.items() if key != "expected"}
        )
        method_results = {}
        for method in ("b4_sequential", "iper_rag"):
            retriever = KGEvidenceRetriever(graph)
            decision = ReplanningEngine(retriever).run(
                case,
                SearchBudget(
                    max_depth=4,
                    beam_width=8,
                    max_retrieval_calls=4,
                    initial_evidence_limit=3,
                    evidence_per_obligation=2,
                    max_plans=3,
                ),
                method=method,
            )
            passed = decision.decision_type == expected[method]
            all_expected = all_expected and passed
            method_results[method] = {
                "decision_type": decision.decision_type,
                "expected_decision_type": expected[method],
                "passed": passed,
                "retrieval_calls": decision.retrieval_calls,
                "queries": retriever.queries,
                "plans": [plan.model_dump(mode="json") for plan in decision.plans],
                "unresolved_conditions": decision.unresolved_conditions,
                "excluded_neural_candidates": retriever.excluded_neural_candidates,
            }
        rows.append({"case_id": case.case_id, "methods": method_results})
    return {
        "schema_version": "0.1",
        "dataset_id": fixture["dataset_id"],
        "reporting_boundary": fixture["reporting_boundary"],
        "all_expected_mechanisms_passed": all_expected,
        "case_count": len(rows),
        "results": rows,
        "interpretation": (
            "This is a deterministic mechanism check, not a performance evaluation. "
            "It verifies that reviewed KG evidence can satisfy path-created obligations "
            "while pending neural candidates, unknown state, and locked intent remain gated."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NSCEF-extension mechanism checks")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = json.loads(args.cases.read_text(encoding="utf-8"))
    result = run_cases(load_graph(args.graph), fixture)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case_count": result["case_count"],
                "all_expected_mechanisms_passed": result[
                    "all_expected_mechanisms_passed"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["all_expected_mechanisms_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
