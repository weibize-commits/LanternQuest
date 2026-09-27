from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "kg" / "rules" / "neurosymbolic_policy_v0.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_candidate(
    candidate: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    checks = candidate.get("rule_checks", [])
    hard_failures = [
        check["rule_id"]
        for check in checks
        if check.get("hard") and check.get("result") == "fail"
    ]
    hard_unknowns = [
        check["rule_id"]
        for check in checks
        if check.get("hard") and check.get("result") == "unknown"
    ]
    passed = sum(check.get("result") == "pass" for check in checks)
    rule_pass_rate = passed / len(checks) if checks else 0.0
    weights = policy["weights"]
    scores = candidate["scores"]
    fused_score = (
        weights["cross_modal_similarity"] * scores["cross_modal_similarity"]
        + weights["extractor_confidence"] * scores["extractor_confidence"]
        + weights["symbolic_rule_pass_rate"] * rule_pass_rate
    )

    source_policy = candidate["source_policy"]
    processing_scope = source_policy["processing_scope"]
    rights_status = source_policy["rights_status"]
    allowed_external = set(policy["external_processing_allowed_rights_statuses"])
    rights_blocked = (
        processing_scope in {"external_api", "public_release"}
        and rights_status not in allowed_external
    )
    review_status = source_policy["review_status"]
    auto_review_statuses = set(policy["auto_accept_review_statuses"])
    thresholds = policy["thresholds"]

    if rights_blocked:
        decision = "blocked"
        reasons = ["rights_not_cleared_for_requested_processing_scope"]
    elif hard_failures:
        decision = "rejected"
        reasons = [f"hard_rule_failed:{rule_id}" for rule_id in hard_failures]
    elif hard_unknowns:
        decision = "human_review"
        reasons = [f"hard_rule_unknown:{rule_id}" for rule_id in hard_unknowns]
    elif (
        fused_score >= thresholds["auto_accept"]
        and review_status in auto_review_statuses
    ):
        decision = "accepted"
        reasons = ["neural_score_and_symbolic_gates_passed"]
    elif fused_score >= thresholds["human_review"]:
        decision = "human_review"
        reasons = ["score_or_review_status_requires_human_review"]
    else:
        decision = "rejected"
        reasons = ["fused_score_below_review_threshold"]

    return {
        "schema_version": "0.1",
        "candidate_id": candidate["candidate_id"],
        "policy_id": policy["policy_id"],
        "fused_score": round(fused_score, 6),
        "symbolic_rule_pass_rate": round(rule_pass_rate, 6),
        "decision": decision,
        "reasons": reasons,
        "hard_failures": hard_failures,
        "hard_unknowns": hard_unknowns,
        "model": candidate["model"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply symbolic gates to a neural candidate")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_candidate(load_json(args.candidate), load_json(args.policy))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
