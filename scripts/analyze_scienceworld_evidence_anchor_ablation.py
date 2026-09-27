"""Analyze E26 against the frozen E24 control and anchored scheduler arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_scienceworld_evidence_anchor_dev import (
    _load_results,
    _load_traces,
    _paired_bootstrap,
    _summary,
)


def _verify_replay(
    expected: set[tuple[str, int, int]],
    control_traces: dict[tuple[str, int, int], dict[str, Any]],
    ablation_traces: dict[tuple[str, int, int], dict[str, Any]],
) -> list[str]:
    problems: list[str] = []
    for key in sorted(expected & set(control_traces) & set(ablation_traces)):
        source = control_traces[key]
        replay = ablation_traces[key]
        if source.get("mode") != "record":
            problems.append(f"control_mode:{key}")
        if replay.get("mode") != "replay_until_divergence":
            problems.append(f"ablation_mode:{key}")
        if replay.get("source_trace_sha256") != source.get("__document_sha256"):
            problems.append(f"source_hash:{key}")
        source_entries = list(source.get("entries", []))
        replay_entries = list(replay.get("entries", []))
        consumed = int(replay.get("source_consumed_count", -1))
        if int(replay.get("source_call_count", -1)) != len(source_entries):
            problems.append(f"source_call_count:{key}")
        if consumed < 0 or consumed > len(source_entries) or consumed > len(replay_entries):
            problems.append(f"consumed_count:{key}")
            continue
        for index, (recorded, reproduced) in enumerate(
            zip(source_entries[:consumed], replay_entries[:consumed], strict=True),
            start=1,
        ):
            if (
                reproduced.get("source") != "replay"
                or reproduced.get("request_sha256")
                != recorded.get("request_sha256")
                or reproduced.get("response") != recorded.get("response")
            ):
                problems.append(f"shared_prefix:{key}:{index}")
                break
    return problems


def _contrast(
    left: dict[tuple[str, int, int], dict[str, Any]],
    right: dict[tuple[str, int, int], dict[str, Any]],
    paired: list[tuple[str, int, int]],
    *,
    seed_base: int,
) -> dict[str, Any]:
    return {
        "mean_clipped_score": _paired_bootstrap(
            [
                max(0, float(left[key]["final_score"]))
                - max(0, float(right[key]["final_score"]))
                for key in paired
            ],
            samples=20_000,
            seed=seed_base + 1,
        ),
        "success_rate": _paired_bootstrap(
            [
                float(bool(left[key]["task_success"]))
                - float(bool(right[key]["task_success"]))
                for key in paired
            ],
            samples=20_000,
            seed=seed_base + 2,
        ),
    }


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.glob("*.json") if item.is_file()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def analyze(
    protocol: dict[str, Any],
    *,
    e24_root: Path,
    e24_control_traces_root: Path,
    e24_anchor_traces_root: Path,
    e26_root: Path,
    e26_traces_root: Path,
) -> dict[str, Any]:
    expected = {
        (str(item["task_id"]), int(variation), int(protocol["seed"]))
        for item in protocol["selected_variations"]
        for variation in item["dev_variations"]
    }
    control, control_problems = _load_results(e24_root / "full_control" / "ensr_v2")
    anchored, anchor_problems = _load_results(
        e24_root / "evidence_anchor_scheduler" / "ensr_v2"
    )
    unanchored, unanchored_problems = _load_results(
        e26_root / "evidence_scheduler_no_anchor" / "ensr_v2"
    )
    control_traces, control_trace_problems = _load_traces(e24_control_traces_root)
    anchor_traces, anchor_trace_problems = _load_traces(e24_anchor_traces_root)
    unanchored_traces, unanchored_trace_problems = _load_traces(e26_traces_root)
    replay_problems = _verify_replay(expected, control_traces, unanchored_traces)

    source_hash_checks = {
        "control_results": _tree_sha256(e24_root / "full_control" / "ensr_v2")
        == protocol["source_control_results_sha256"],
        "anchor_results": _tree_sha256(
            e24_root / "evidence_anchor_scheduler" / "ensr_v2"
        )
        == protocol["source_anchor_results_sha256"],
        "control_traces": _tree_sha256(e24_control_traces_root)
        == protocol["source_control_traces_sha256"],
        "anchor_traces": _tree_sha256(e24_anchor_traces_root)
        == protocol["source_anchor_traces_sha256"],
    }
    paired = sorted(expected & set(control) & set(anchored) & set(unanchored))
    complete = (
        set(control) == expected
        and set(anchored) == expected
        and set(unanchored) == expected
        and set(control_traces) == expected
        and set(anchor_traces) == expected
        and set(unanchored_traces) == expected
        and not control_problems
        and not anchor_problems
        and not unanchored_problems
        and not control_trace_problems
        and not anchor_trace_problems
        and not unanchored_trace_problems
        and not replay_problems
        and all(source_hash_checks.values())
    )
    control_summary = _summary([control[key] for key in paired])
    anchored_summary = _summary([anchored[key] for key in paired])
    unanchored_summary = _summary([unanchored[key] for key in paired])
    anchored_minus_unanchored = _contrast(
        anchored, unanchored, paired, seed_base=26_000
    )
    unanchored_minus_control = _contrast(
        unanchored, control, paired, seed_base=26_100
    )
    anchored_minus_control = _contrast(anchored, control, paired, seed_base=26_200)
    unanchored_retrieval_reduction = 1.0 - (
        unanchored_summary["obligation_retrieval_calls"]
        / max(1, control_summary["obligation_retrieval_calls"])
    )
    anchor_retrieval_increment = (
        anchored_summary["obligation_retrieval_calls"]
        - unanchored_summary["obligation_retrieval_calls"]
    )
    minimum_prefix = min(
        (
            int(unanchored_traces[key].get("source_consumed_count", 0))
            for key in expected & set(unanchored_traces)
        ),
        default=0,
    )
    mean_prefix = sum(
        int(unanchored_traces[key].get("source_consumed_count", 0))
        for key in expected & set(unanchored_traces)
    ) / max(1, len(expected & set(unanchored_traces)))
    mismatch_count = sum(
        int(row.get("mismatch_count", 0)) for row in unanchored_traces.values()
    )
    checks = dict(protocol["ablation_checks"])
    gate_checks = {
        "complete_pairing": complete,
        "trace_replay_integrity": (
            mismatch_count <= int(checks["trace_mismatch_count"])
            and minimum_prefix
            >= int(checks["minimum_shared_prefix_calls_per_episode"])
        ),
        "anchored_mean_score_not_lower": (
            anchored_minus_unanchored["mean_clipped_score"]["point_estimate"] >= 0
        ),
        "anchored_success_rate_not_lower": (
            anchored_minus_unanchored["success_rate"]["point_estimate"] >= 0
        ),
        "unanchored_retrieval_reduction": (
            unanchored_retrieval_reduction
            >= float(checks["unanchored_retrieval_reduction_vs_control_at_least"])
        ),
        "zero_uncontrolled_failures": (
            control_summary["uncontrolled_failures"]
            + anchored_summary["uncontrolled_failures"]
            + unanchored_summary["uncontrolled_failures"]
        )
        <= int(checks["uncontrolled_failure_count"]),
    }
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "E26",
        "reporting_boundary": protocol["reporting_boundary"],
        "complete_pairing": complete,
        "paired_episode_count": len(paired),
        "source_hash_checks": source_hash_checks,
        "result_problems": {
            "control": control_problems,
            "anchored": anchor_problems,
            "unanchored": unanchored_problems,
            "control_traces": control_trace_problems,
            "anchor_traces": anchor_trace_problems,
            "unanchored_traces": unanchored_trace_problems,
            "trace_integrity": replay_problems,
        },
        "summaries": {
            "full_control": control_summary,
            "evidence_anchor_scheduler": anchored_summary,
            "evidence_scheduler_no_anchor": unanchored_summary,
        },
        "contrasts": {
            "anchored_minus_unanchored": anchored_minus_unanchored,
            "unanchored_minus_control": unanchored_minus_control,
            "anchored_minus_control": anchored_minus_control,
            "unanchored_retrieval_reduction_vs_control": (
                unanchored_retrieval_reduction
            ),
            "anchor_obligation_retrieval_increment": anchor_retrieval_increment,
        },
        "trace_audit": {
            "mismatch_count": mismatch_count,
            "minimum_shared_prefix_calls": minimum_prefix,
            "mean_shared_prefix_calls": mean_prefix,
            "planned_divergence_count": sum(
                row.get("divergence_purpose") is not None
                for row in unanchored_traces.values()
            ),
            "prefix_complete_count": sum(
                bool(row.get("prefix_complete"))
                for row in unanchored_traces.values()
            ),
        },
        "ablation_gate_checks": gate_checks,
        "ablation_gate_passed": all(gate_checks.values()),
        "interpretation": protocol["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--e24-root", type=Path, required=True)
    parser.add_argument("--e24-control-traces", type=Path, required=True)
    parser.add_argument("--e24-anchor-traces", type=Path, required=True)
    parser.add_argument("--e26-root", type=Path, required=True)
    parser.add_argument("--e26-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        json.loads(args.protocol.read_text(encoding="utf-8")),
        e24_root=args.e24_root,
        e24_control_traces_root=args.e24_control_traces,
        e24_anchor_traces_root=args.e24_anchor_traces,
        e26_root=args.e26_root,
        e26_traces_root=args.e26_traces,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "paired_episode_count": report["paired_episode_count"],
                "ablation_gate_passed": report["ablation_gate_passed"],
                "ablation_gate_checks": report["ablation_gate_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
