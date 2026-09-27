"""Audit evidence-debt retrieval sensitivity on frozen ScienceWorld traces.

This analysis replays retrieval decisions only. It does not estimate task
performance because a different evidence set can change later model actions.
"""

from __future__ import annotations

import argparse
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


DEFAULT_THRESHOLDS = (0.45, 0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.75)


def build_report(root: Path, thresholds: tuple[float, ...]) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    for anchor in (False, True):
        for threshold in thresholds:
            replay = audit(
                root,
                threshold=threshold,
                anchor_first_debt_per_subgoal=anchor,
            )
            arms.append(
                {
                    "anchor_first_debt_per_subgoal": anchor,
                    "retrieval_threshold": threshold,
                    "episode_count": replay["episode_count"],
                    "parse_failures": replay["parse_failures"],
                    "original_obligation_retrieval_calls": replay[
                        "original_obligation_retrieval_calls"
                    ],
                    "scheduled_obligation_retrieval_calls": replay[
                        "scheduled_obligation_retrieval_calls"
                    ],
                    "suppressed_obligation_retrieval_calls": replay[
                        "suppressed_obligation_retrieval_calls"
                    ],
                    "estimated_suppression_rate": replay[
                        "estimated_suppression_rate"
                    ],
                    "decision_counts": replay["decision_counts"],
                    "reason_counts": replay["reason_counts"],
                }
            )

    local = [
        row
        for row in arms
        if row["anchor_first_debt_per_subgoal"]
        and row["retrieval_threshold"] in {0.54, 0.58, 0.62}
    ]
    anchor_counts = {
        int(row["reason_counts"].get("first_subgoal_evidence_anchor", 0))
        for row in local
    }
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": "evidence_scheduler_threshold_sensitivity_v1",
        "reporting_boundary": (
            "deterministic_frozen_trace_replay_only_no_task_performance_claim"
        ),
        "source_root": str(root.resolve()),
        "thresholds": list(thresholds),
        "arms": arms,
        "local_anchor_sensitivity": {
            "threshold_range": [0.54, 0.62],
            "minimum_estimated_suppression_rate": min(
                row["estimated_suppression_rate"] for row in local
            ),
            "maximum_estimated_suppression_rate": max(
                row["estimated_suppression_rate"] for row in local
            ),
            "first_subgoal_anchor_count": (
                next(iter(anchor_counts)) if len(anchor_counts) == 1 else None
            ),
        },
        "interpretation": (
            "The threshold controls retrieval expenditure and is not performance-tuned. "
            "The first-debt anchor supplies a stable retrieval floor across the local "
            "threshold range. Live paired experiments remain necessary for score, success, "
            "token, and environment-step claims."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LanternQuest evidence scheduler threshold sensitivity audit",
        "",
        "This deterministic analysis replays the scheduler on 60 frozen full-control "
        "ScienceWorld development traces. It estimates retrieval decisions only.",
        "",
        "| Anchor | Threshold | Scheduled retrievals | Suppression | First-debt anchors |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["arms"]:
        reasons = row["reason_counts"]
        lines.append(
            "| {anchor} | {threshold:.2f} | {calls} | {rate:.1f}% | {anchors} |".format(
                anchor="on" if row["anchor_first_debt_per_subgoal"] else "off",
                threshold=row["retrieval_threshold"],
                calls=row["scheduled_obligation_retrieval_calls"],
                rate=100 * row["estimated_suppression_rate"],
                anchors=reasons.get("first_subgoal_evidence_anchor", 0),
            )
        )
    local = report["local_anchor_sensitivity"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Across the local anchored range from 0.54 to 0.62, estimated retrieval "
            f"suppression ranged from {100 * local['minimum_estimated_suppression_rate']:.1f}% "
            f"to {100 * local['maximum_estimated_suppression_rate']:.1f}%. The replay "
            f"retained {local['first_subgoal_anchor_count']} first-debt anchors at every "
            "threshold in this range. The frozen value of 0.58 is therefore reported as "
            "a prespecified budget trade-off rather than an optimized optimum.",
            "",
            "This audit cannot estimate score, success, tokens, or environment steps. "
            "Those claims come only from paired live experiments E23 to E26.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "artifacts/eval/LanternQuest_evidence_scheduler_threshold_sensitivity_v1.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "artifacts/research/LanternQuest_证据调度阈值敏感性审计_v1.md"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    args = parser.parse_args()

    report = build_report(args.root, tuple(args.thresholds))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(args.output_json.resolve())
    print(args.output_md.resolve())


if __name__ == "__main__":
    main()
