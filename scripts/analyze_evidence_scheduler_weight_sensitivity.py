"""Audit evidence-debt weight sensitivity on frozen scheduler traces.

The analysis perturbs one heuristic priority weight at a time by plus or minus
10% and 20%, renormalizes the five weights to sum to one, and replays the
anchored scheduler at the frozen threshold. It estimates retrieval decisions
only and makes no task-performance claim.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from audit_scienceworld_evidence_scheduler_replay import audit
from lanternquest.evidence_scheduler import DEFAULT_PRIORITY_WEIGHTS


SOURCE_ROOT_DEFAULT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "benchmarks"
    / "scienceworld"
    / "ensr_mechanism_ablation_v1_runs"
    / "scienceworld_ensr_mechanism_ablation_v1"
    / "bit_qwen3_32b_clean"
    / "dev"
    / "full_control"
    / "ensr_v2"
)


def normalized_perturbation(name: str, change: float) -> dict[str, float]:
    weights = dict(DEFAULT_PRIORITY_WEIGHTS)
    weights[name] *= 1.0 + change
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def decision_vector(report: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        decision["action"]
        for episode in report["episodes"]
        for decision in episode["decisions"]
    )


def build_report(root: Path) -> dict[str, Any]:
    baseline = audit(
        root,
        threshold=0.58,
        anchor_first_debt_per_subgoal=True,
        priority_weights=dict(DEFAULT_PRIORITY_WEIGHTS),
    )
    baseline_vector = decision_vector(baseline)
    rows: list[dict[str, Any]] = []
    for name in DEFAULT_PRIORITY_WEIGHTS:
        for change in (-0.20, -0.10, 0.10, 0.20):
            weights = normalized_perturbation(name, change)
            replay = audit(
                root,
                threshold=0.58,
                anchor_first_debt_per_subgoal=True,
                priority_weights=weights,
            )
            vector = decision_vector(replay)
            changed = sum(a != b for a, b in zip(vector, baseline_vector))
            rows.append(
                {
                    "perturbed_weight": name,
                    "relative_change": change,
                    "normalized_weights": weights,
                    "scheduled_obligation_retrieval_calls": replay[
                        "scheduled_obligation_retrieval_calls"
                    ],
                    "estimated_suppression_rate": replay[
                        "estimated_suppression_rate"
                    ],
                    "first_debt_anchor_count": replay["reason_counts"].get(
                        "first_subgoal_evidence_anchor", 0
                    ),
                    "changed_decisions_vs_baseline": changed,
                    "decision_agreement_vs_baseline": (
                        1.0 - changed / len(baseline_vector)
                        if baseline_vector
                        else 1.0
                    ),
                    "parse_failures": replay["parse_failures"],
                }
            )

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": "evidence_scheduler_weight_sensitivity_v1",
        "reporting_boundary": (
            "deterministic_frozen_trace_replay_only_no_task_performance_claim"
        ),
        "source_root": str(root.resolve()),
        "threshold": 0.58,
        "anchor_first_debt_per_subgoal": True,
        "baseline_weights": dict(DEFAULT_PRIORITY_WEIGHTS),
        "baseline": {
            "episode_count": baseline["episode_count"],
            "obligation_decisions": len(baseline_vector),
            "original_obligation_retrieval_calls": baseline[
                "original_obligation_retrieval_calls"
            ],
            "scheduled_obligation_retrieval_calls": baseline[
                "scheduled_obligation_retrieval_calls"
            ],
            "estimated_suppression_rate": baseline["estimated_suppression_rate"],
            "first_debt_anchor_count": baseline["reason_counts"].get(
                "first_subgoal_evidence_anchor", 0
            ),
            "parse_failures": baseline["parse_failures"],
        },
        "one_at_a_time_perturbations": rows,
        "summary": {
            "minimum_scheduled_retrievals": min(
                row["scheduled_obligation_retrieval_calls"] for row in rows
            ),
            "maximum_scheduled_retrievals": max(
                row["scheduled_obligation_retrieval_calls"] for row in rows
            ),
            "minimum_suppression_rate": min(
                row["estimated_suppression_rate"] for row in rows
            ),
            "maximum_suppression_rate": max(
                row["estimated_suppression_rate"] for row in rows
            ),
            "minimum_decision_agreement": min(
                row["decision_agreement_vs_baseline"] for row in rows
            ),
            "maximum_changed_decisions": max(
                row["changed_decisions_vs_baseline"] for row in rows
            ),
            "first_debt_anchor_counts": sorted(
                {row["first_debt_anchor_count"] for row in rows}
            ),
        },
        "interpretation": (
            "The weights are heuristic development controls rather than learned "
            "parameters. The one-at-a-time replay tests local decision stability. "
            "It cannot estimate task score, success, latency, or token effects."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    base = report["baseline"]
    lines = [
        "# LanternQuest evidence scheduler weight sensitivity audit",
        "",
        "The five heuristic weights were perturbed one at a time by plus or minus "
        "10% and 20%, renormalized to sum to one, and replayed at the frozen "
        "threshold of 0.58 with the first-debt anchor enabled.",
        "",
        "| Weight | Change | Scheduled retrievals | Suppression | Agreement with baseline | First-debt anchors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["one_at_a_time_perturbations"]:
        lines.append(
            "| {name} | {change:+.0f}% | {calls} | {supp:.1f}% | {agree:.1f}% | {anchors} |".format(
                name=row["perturbed_weight"],
                change=100 * row["relative_change"],
                calls=row["scheduled_obligation_retrieval_calls"],
                supp=100 * row["estimated_suppression_rate"],
                agree=100 * row["decision_agreement_vs_baseline"],
                anchors=row["first_debt_anchor_count"],
            )
        )
    summary = report["summary"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The baseline scheduled {base['scheduled_obligation_retrieval_calls']} "
            f"of {base['original_obligation_retrieval_calls']} obligation retrievals. "
            f"Across 20 perturbations, scheduled retrievals ranged from "
            f"{summary['minimum_scheduled_retrievals']} to "
            f"{summary['maximum_scheduled_retrievals']}. Decision agreement with the "
            f"baseline remained at least {100 * summary['minimum_decision_agreement']:.1f}%. "
            f"Every replay retained {summary['first_debt_anchor_counts'][0]} first-debt anchors.",
            "",
            "This is a frozen-trace decision audit. It does not estimate score, "
            "success, latency, or token effects.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT_DEFAULT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts"
        / "eval"
        / "LanternQuest_evidence_scheduler_weight_sensitivity_v1.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts"
        / "research"
        / "LanternQuest_证据调度权重敏感性审计_v1.md",
    )
    args = parser.parse_args()
    report = build_report(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
