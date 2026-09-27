from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_scienceworld_ensr_results import paired_bootstrap
except ModuleNotFoundError:
    from analyze_scienceworld_ensr_results import paired_bootstrap


METHODS = ("iper_rag", "ensr_v2")
FAILURE_STATUSES = {"adapter_error", "environment_error"}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for method in METHODS:
        for path in sorted((root / method).glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("method") != method:
                    raise ValueError("method mismatch")
                rows.append(row)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"{path}: {type(error).__name__}: {error}")
    return rows, failures


def _analyze_profile(
    protocol: dict[str, Any], profile_id: str, profile_root: Path
) -> dict[str, Any]:
    rows, parse_failures = _load_rows(profile_root)
    seed = int(protocol["runs"]["formal_seeds"][0])
    cases = [
        (str(item["task_id"]), int(item["test"]))
        for item in protocol["selected_variations"]
    ]
    expected = {
        (method, task_id, variation, seed)
        for method in METHODS
        for task_id, variation in cases
    }
    observed = [
        (
            str(row.get("method")),
            str(row.get("task_id")),
            int(row.get("variation", -1)),
            int(row.get("seed", -1)),
        )
        for row in rows
    ]
    duplicates = sorted(
        key for key, count in Counter(observed).items() if count > 1
    )
    observed_set = set(observed)
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)
    summaries: dict[str, Any] = {}
    task_values: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        selected = [row for row in rows if row.get("method") == method]
        summaries[method] = {
            "episodes": len(selected),
            "status_counts": dict(
                sorted(Counter(str(row.get("status")) for row in selected).items())
            ),
            "mean_clipped_score": _mean(
                [max(0.0, float(row.get("final_score", 0))) for row in selected]
            ),
            "task_success_rate": _mean(
                [float(bool(row.get("task_success"))) for row in selected]
            ),
            "environment_steps": sum(
                int(row.get("environment_steps", 0)) for row in selected
            ),
            "model_calls": sum(int(row.get("model_calls", 0)) for row in selected),
            "input_tokens": sum(int(row.get("input_tokens", 0)) for row in selected),
            "output_tokens": sum(
                int(row.get("output_tokens", 0)) for row in selected
            ),
            "provider_or_environment_failures": sum(
                str(row.get("status")) in FAILURE_STATUSES for row in selected
            ),
        }
        task_values[method] = {
            str(row["task_id"]): {
                "clipped_score": max(0.0, float(row.get("final_score", 0))),
                "success": float(bool(row.get("task_success"))),
            }
            for row in selected
        }
    score = paired_bootstrap(
        {
            task: value["clipped_score"]
            for task, value in task_values["ensr_v2"].items()
        },
        {
            task: value["clipped_score"]
            for task, value in task_values["iper_rag"].items()
        },
    )
    success = paired_bootstrap(
        {
            task: value["success"]
            for task, value in task_values["ensr_v2"].items()
        },
        {
            task: value["success"]
            for task, value in task_values["iper_rag"].items()
        },
        seed=12027,
    )
    complete = not (missing or unexpected or duplicates or parse_failures)
    return {
        "profile_id": profile_id,
        "source_root": str(profile_root.resolve()),
        "expected_episode_count": len(expected),
        "observed_episode_count": len(rows),
        "complete_matrix": complete,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "duplicate_keys": duplicates,
        "parse_failures": parse_failures,
        "summaries": summaries,
        "ensr_minus_iper": {
            "clipped_score": score,
            "task_success_rate": success,
        },
        "directional_pass": (
            complete
            and float(score["point_estimate"]) > 0
            and float(success["point_estimate"]) >= 0
        ),
        "strong_success": (
            complete
            and float(score["ci95_low"]) > 0
            and float(success["point_estimate"]) >= 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/scienceworld_protocol_external_provider_v1_frozen.json"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/external_provider_v1_runs/"
            "scienceworld_ensr_v3"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/external_provider_v1_audit.json"
        ),
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    profile_ids = list(protocol["external_provider_design"]["profiles"])
    profiles = {
        profile_id: _analyze_profile(
            protocol, profile_id, args.run_root / profile_id / "test"
        )
        for profile_id in profile_ids
    }
    passing = [
        profile_id
        for profile_id, result in profiles.items()
        if result["directional_pass"]
    ]
    complete = all(result["complete_matrix"] for result in profiles.values())
    design = protocol["external_provider_design"]
    required = int(
        design.get(
            "required_directional_pass_profiles",
            max(1, (3 * len(profile_ids) + 3) // 4),
        )
    )
    expected_episode_count = sum(
        item["expected_episode_count"] for item in profiles.values()
    )
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": protocol["experiment_id"],
        "protocol_id": protocol["protocol_id"],
        "complete_all_profiles": complete,
        "expected_episode_count": expected_episode_count,
        "observed_episode_count": sum(
            item["observed_episode_count"] for item in profiles.values()
        ),
        "profiles": profiles,
        "robustness_summary": {
            "directional_pass_profiles": passing,
            "directional_pass_count": len(passing),
            "required_count": required,
            "external_provider_robustness_pass": complete
            and len(passing) >= required,
            "strong_success_profiles": [
                profile_id
                for profile_id, result in profiles.items()
                if result["strong_success"]
            ],
        },
    }
    _write_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
