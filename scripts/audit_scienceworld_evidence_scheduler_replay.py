"""Replay the evidence-debt scheduler against frozen ENSR retrieval traces.

This audit estimates retrieval decisions only.  It cannot estimate task scores
because changing retrieved memory can change later model actions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lanternquest.evidence_scheduler import EvidenceDebtLedger, EvidenceDebtSignal


def _fact_key(raw: dict[str, Any]) -> tuple[str, str, str, bool]:
    return (
        str(raw["subject"]),
        str(raw["relation"]),
        str(raw["object"]),
        bool(raw.get("polarity", True)),
    )


def audit_episode(
    row: dict[str, Any],
    *,
    threshold: float,
    anchor_first_debt_per_subgoal: bool,
    priority_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    ledger = EvidenceDebtLedger(
        retrieval_threshold=threshold,
        anchor_first_debt_per_subgoal=anchor_first_debt_per_subgoal,
        priority_weights=priority_weights,
    )
    obligations = {
        str(item["obligation_id"]): item for item in row.get("obligations", [])
    }
    events = list(row.get("retrieval_events", []))
    initial = next((item for item in events if item["trigger"] == "initial"), None)
    active = list(initial["returned_fragment_ids"]) if initial else []
    max_calls = max(1, int(row.get("retrieval_calls", len(events))))
    scheduled_calls = 1 if initial else 0
    decisions: list[dict[str, Any]] = []
    missing_obligation_ids: list[str] = []

    for event in (item for item in events if item["trigger"] == "obligation"):
        obligation_id = str(event.get("obligation_id") or "")
        obligation = obligations.get(obligation_id)
        if obligation is None:
            missing_obligation_ids.append(obligation_id)
            continue
        candidates = list(event.get("returned_fragment_ids", []))
        signal = EvidenceDebtSignal(
            obligation_id=obligation_id,
            subgoal_id=str(obligation["subgoal_id"]),
            source=str(obligation["source"]),
            action=str(obligation["action"]),
            step=int(obligation["step"]),
            fact_keys=tuple(
                _fact_key(item)
                for item in obligation.get("missing_or_conflicting_facts", [])
            ),
        )
        decision = ledger.schedule(
            signal,
            candidate_fragment_ids=candidates,
            active_fragment_ids=active,
            remaining_retrieval_calls=max(0, max_calls - scheduled_calls),
            max_retrieval_calls=max_calls,
            max_retrieval_k=max(1, len(candidates)),
        )
        selected = candidates[: decision.requested_k] if decision.action == "retrieve" else []
        if selected:
            ledger.commit_retrieval(decision.debt_signature, selected)
            scheduled_calls += 1
            active = selected
        decisions.append(
            {
                **decision.__dict__,
                "candidate_fragment_ids": candidates,
                "selected_fragment_ids": selected,
            }
        )

    original_obligation_calls = sum(item["trigger"] == "obligation" for item in events)
    scheduled_obligation_calls = sum(
        item["action"] == "retrieve" for item in decisions
    )
    return {
        "task_id": row.get("task_id"),
        "variation": row.get("variation"),
        "seed": row.get("seed"),
        "status": row.get("status"),
        "final_score": row.get("final_score"),
        "original_obligation_retrieval_calls": original_obligation_calls,
        "scheduled_obligation_retrieval_calls": scheduled_obligation_calls,
        "suppressed_obligation_retrieval_calls": (
            original_obligation_calls - scheduled_obligation_calls
        ),
        "decision_counts": dict(Counter(item["action"] for item in decisions)),
        "reason_counts": dict(Counter(item["reason"] for item in decisions)),
        "missing_obligation_ids": missing_obligation_ids,
        "decisions": decisions,
        "ledger": ledger.snapshot(),
    }


def audit(
    root: Path,
    *,
    threshold: float,
    anchor_first_debt_per_subgoal: bool = False,
    priority_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            episodes.append(
                audit_episode(
                    json.loads(path.read_text(encoding="utf-8")),
                    threshold=threshold,
                    anchor_first_debt_per_subgoal=anchor_first_debt_per_subgoal,
                    priority_weights=priority_weights,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_failures.append(f"{path.name}: {type(exc).__name__}: {exc}")

    original = sum(item["original_obligation_retrieval_calls"] for item in episodes)
    scheduled = sum(item["scheduled_obligation_retrieval_calls"] for item in episodes)
    suppressed = original - scheduled
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "E23-OFFLINE",
        "reporting_boundary": (
            "deterministic_replay_audit_only_no_task_performance_claim"
        ),
        "source_root": str(root.resolve()),
        "retrieval_threshold": threshold,
        "anchor_first_debt_per_subgoal": anchor_first_debt_per_subgoal,
        "episode_count": len(episodes),
        "parse_failures": parse_failures,
        "original_obligation_retrieval_calls": original,
        "scheduled_obligation_retrieval_calls": scheduled,
        "suppressed_obligation_retrieval_calls": suppressed,
        "estimated_suppression_rate": suppressed / original if original else 0.0,
        "decision_counts": dict(
            Counter(
                action
                for episode in episodes
                for action, count in episode["decision_counts"].items()
                for _ in range(count)
            )
        ),
        "reason_counts": dict(
            Counter(
                reason
                for episode in episodes
                for reason, count in episode["reason_counts"].items()
                for _ in range(count)
            )
        ),
        "episodes": episodes,
        "interpretation": (
            "The replay quantifies deterministic retrieval suppression on frozen traces. "
            "A paired live development experiment is required to estimate score, success, "
            "latency, and token effects."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    rate = 100 * float(report["estimated_suppression_rate"])
    lines = [
        "# LanternQuest E23 evidence-debt scheduler replay audit",
        "",
        "This deterministic audit replays the new scheduler over frozen E16 full-control "
        "retrieval traces. It does not rerun a model and cannot support a task-performance claim.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Episodes | {report['episode_count']} |",
        f"| Original obligation retrievals | {report['original_obligation_retrieval_calls']} |",
        f"| Scheduled obligation retrievals | {report['scheduled_obligation_retrieval_calls']} |",
        f"| Suppressed obligation retrievals | {report['suppressed_obligation_retrieval_calls']} |",
        f"| Estimated suppression rate | {rate:.1f}% |",
        f"| Parse failures | {len(report['parse_failures'])} |",
        "",
        "## Decision distribution",
        "",
        "| Decision | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {count} |"
        for name, count in sorted(report["decision_counts"].items())
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(report["interpretation"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--anchor-first-debt-per-subgoal", action="store_true")
    args = parser.parse_args()

    report = audit(
        args.root,
        threshold=args.threshold,
        anchor_first_debt_per_subgoal=args.anchor_first_debt_per_subgoal,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "episode_count",
        "original_obligation_retrieval_calls",
        "scheduled_obligation_retrieval_calls",
        "suppressed_obligation_retrieval_calls",
        "estimated_suppression_rate",
    )}, indent=2))


if __name__ == "__main__":
    main()
