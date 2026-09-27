import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXTERNAL_FAILURES = {"adapter_error", "environment_error"}
DEFAULT_SETTINGS: dict[str, bool | str] = {
    "enable_hierarchical_subgoals": True,
    "enable_symbolic_transition_verifier": True,
    "enable_obligation_conditioned_retrieval": True,
    "fragment_memory_policy": "replace",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_keys(protocol: dict[str, Any]) -> set[tuple[str, int, int]]:
    seed = int(protocol["seed"])
    return {
        (str(item["task_id"]), int(variation), seed)
        for item in protocol["selected_variations"]
        for variation in item["dev_variations"]
    }


def _load_arm(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    for path in sorted((root / "ensr_v2").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["_source_path"] = str(path.resolve())
            rows.append(row)
        except (OSError, json.JSONDecodeError) as exc:
            parse_failures.append(f"{path}: {type(exc).__name__}: {exc}")
    return rows, parse_failures


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    differences: list[float], *, samples: int, seed: int
) -> dict[str, float | int]:
    rng = random.Random(seed)
    size = len(differences)
    estimates = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    return {
        "paired_task_count": size,
        "point_estimate": sum(differences) / size,
        "ci95_low": _percentile(estimates, 0.025),
        "ci95_high": _percentile(estimates, 0.975),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _permutation_p_value(
    differences: list[float], *, samples: int, seed: int
) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    observed = abs(sum(differences) / len(differences))
    extreme = 0
    for _ in range(samples):
        permuted = sum(value if rng.random() < 0.5 else -value for value in differences)
        if abs(permuted / len(differences)) >= observed - 1e-12:
            extreme += 1
    return {
        "method": "paired_random_sign_flip_two_sided_monte_carlo",
        "samples": samples,
        "seed": seed,
        "p_value": (extreme + 1) / (samples + 1),
    }


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _task_aggregates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    return {
        task_id: {
            "mean_clipped_score": sum(max(0, float(row["final_score"])) for row in task_rows)
            / len(task_rows),
            "task_success_rate": sum(bool(row["task_success"]) for row in task_rows)
            / len(task_rows),
            "variation_count": float(len(task_rows)),
        }
        for task_id, task_rows in by_task.items()
    }


def _manipulation_failures(arm_id: str, rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    flag_expectations = {
        "full_control": (1, 1, 1, 0),
        "without_hierarchical_subgoals": (0, 1, 1, 0),
        "without_symbolic_transition_verifier": (1, 0, 1, 0),
        "without_obligation_conditioned_retrieval": (1, 1, 0, 0),
        "accumulating_fragment_memory": (1, 1, 1, 1),
    }
    keys = (
        "hierarchical_subgoals_enabled",
        "symbolic_transition_verifier_enabled",
        "obligation_conditioned_retrieval_enabled",
        "fragment_memory_accumulate_enabled",
    )
    expected = flag_expectations[arm_id]
    for key, value in zip(keys, expected, strict=True):
        bad = sum(row["mechanism_metrics"].get(key) != value for row in rows)
        if bad:
            failures.append(f"{bad} episodes have incorrect {key}")
    if arm_id == "without_hierarchical_subgoals":
        bad = sum(len(row["plan"]["subgoals"]) != 1 for row in rows)
        if bad:
            failures.append(f"{bad} episodes retained multiple subgoals")
    if arm_id == "without_symbolic_transition_verifier":
        verified = sum(row["mechanism_metrics"]["verified_transition_count"] for row in rows)
        obligations = sum(row["mechanism_metrics"]["obligation_count"] for row in rows)
        if verified:
            failures.append(f"verified_transition_count total is {verified}, expected zero")
        if obligations:
            failures.append(f"obligation_count total is {obligations}, expected zero")
    obligation_events = [
        event
        for row in rows
        for event in row["retrieval_events"]
        if event["trigger"] == "obligation"
    ]
    if arm_id == "full_control":
        if not obligation_events:
            failures.append("full control never exercised obligation retrieval")
        if not any(len(row["plan"]["subgoals"]) > 1 for row in rows):
            failures.append("full control never exercised a multi-subgoal plan")
        replacements = sum(
            len(event["replaced_fragment_ids"]) for event in obligation_events
        )
        if not replacements:
            failures.append("full control never exercised fragment replacement")
    if arm_id == "without_obligation_conditioned_retrieval" and obligation_events:
        failures.append(f"found {len(obligation_events)} disabled obligation retrievals")
    if arm_id == "accumulating_fragment_memory":
        if not obligation_events:
            failures.append("accumulating memory was never exercised")
        replacements = sum(len(event["replaced_fragment_ids"]) for event in obligation_events)
        if replacements:
            failures.append(f"accumulating memory replaced {replacements} fragments")
    return failures


def analyze(
    protocol: dict[str, Any],
    root: Path,
    *,
    protocol_sha256: str,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 20_000,
) -> dict[str, Any]:
    expected = _expected_keys(protocol)
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_reports: dict[str, Any] = {}
    all_complete = True
    all_operational = True
    implementation_hashes: set[str] = set()
    for arm in protocol["arms"]:
        arm_id = str(arm["arm_id"])
        rows, parse_failures = _load_arm(root / arm_id)
        arm_rows[arm_id] = rows
        observed_keys = [
            (str(row["task_id"]), int(row["variation"]), int(row["seed"]))
            for row in rows
        ]
        counts = Counter(observed_keys)
        observed = set(observed_keys)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        external_failures = sum(row.get("status") in EXTERNAL_FAILURES for row in rows)
        manipulation_failures = _manipulation_failures(arm_id, rows) if rows else []
        summary_path = root / arm_id / "summary_ensr_v2.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )
        summary_integrity_failures: list[str] = []
        if summary is not None:
            expected_settings = dict(DEFAULT_SETTINGS)
            expected_settings.update(arm["settings"])
            expected_summary_values = {
                "protocol_sha256": protocol_sha256,
                "experiment_id": "E16",
                "protocol_id": protocol["protocol_id"],
                "split": "dev",
                "model_profile_id": "bit_qwen3_32b_clean",
                "ablation_arm": arm_id,
                "ablation_settings": expected_settings,
            }
            for key, expected_value in expected_summary_values.items():
                if summary.get(key) != expected_value:
                    summary_integrity_failures.append(
                        f"summary {key}={summary.get(key)!r}, expected {expected_value!r}"
                    )
        if summary:
            implementation_hashes.add(str(summary.get("implementation_sha256")))
        complete = (
            len(rows) == len(expected)
            and not missing
            and not unexpected
            and not duplicates
            and not parse_failures
            and summary is not None
        )
        operational = (
            complete
            and external_failures <= 3
            and not manipulation_failures
            and not summary_integrity_failures
        )
        all_complete &= complete
        all_operational &= operational
        obligation_events = [
            event
            for row in rows
            for event in row["retrieval_events"]
            if event["trigger"] == "obligation"
        ]
        arm_reports[arm_id] = {
            "episodes": len(rows),
            "complete": complete,
            "operational_pass": operational,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "duplicate_keys": duplicates,
            "parse_failures": parse_failures,
            "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "external_failures": external_failures,
            "mean_clipped_score": (
                sum(max(0, float(row["final_score"])) for row in rows) / len(rows)
                if rows
                else None
            ),
            "successes": sum(bool(row["task_success"]) for row in rows),
            "task_success_rate": (
                sum(bool(row["task_success"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "environment_steps": sum(int(row["environment_steps"]) for row in rows),
            "model_calls": sum(int(row["model_calls"]) for row in rows),
            "retrieval_calls": sum(int(row["retrieval_calls"]) for row in rows),
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
            "successes_per_100_model_calls": (
                100 * sum(bool(row["task_success"]) for row in rows)
                / sum(int(row["model_calls"]) for row in rows)
                if sum(int(row["model_calls"]) for row in rows)
                else 0.0
            ),
            "obligation_count": sum(
                int(row["mechanism_metrics"]["obligation_count"]) for row in rows
            ),
            "obligation_retrieval_count": len(obligation_events),
            "replaced_fragment_count": sum(
                len(event["replaced_fragment_ids"]) for event in obligation_events
            ),
            "manipulation_pass": not manipulation_failures,
            "manipulation_failures": manipulation_failures,
            "summary_integrity_pass": not summary_integrity_failures,
            "summary_integrity_failures": summary_integrity_failures,
        }

    comparisons: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    if all_complete:
        task_values = {arm: _task_aggregates(rows) for arm, rows in arm_rows.items()}
        control = task_values["full_control"]
        for index, arm in enumerate(protocol["arms"][1:], start=1):
            arm_id = str(arm["arm_id"])
            ablated = task_values[arm_id]
            task_ids = sorted(control)
            score_differences = [
                control[task]["mean_clipped_score"]
                - ablated[task]["mean_clipped_score"]
                for task in task_ids
            ]
            success_differences = [
                control[task]["task_success_rate"] - ablated[task]["task_success_rate"]
                for task in task_ids
            ]
            score = _paired_bootstrap(
                score_differences, samples=bootstrap_samples, seed=16_100 + index
            )
            success = _paired_bootstrap(
                success_differences, samples=bootstrap_samples, seed=16_200 + index
            )
            permutation = _permutation_p_value(
                score_differences, samples=permutation_samples, seed=16_300 + index
            )
            comparison_id = f"full_control_minus_{arm_id}"
            raw_p_values[comparison_id] = float(permutation["p_value"])
            comparisons[comparison_id] = {
                "clipped_score": score,
                "task_success_rate": success,
                "score_permutation": permutation,
                "direction_positive": float(score["point_estimate"]) > 0,
                "score_ci_excludes_zero": float(score["ci95_low"]) > 0
                or float(score["ci95_high"]) < 0,
            }
        adjusted = _holm_adjust(raw_p_values)
        for name, value in adjusted.items():
            comparisons[name]["score_permutation"]["holm_adjusted_p_value"] = value

    frozen_implementation = protocol.get("freeze", {}).get("hashes", {}).get(
        "implementation_sha256"
    )
    implementation_match = (
        len(implementation_hashes) == 1
        and next(iter(implementation_hashes)) == frozen_implementation
    )
    operational_pass = all_operational and implementation_match
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "E16",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "expected_episode_count": protocol["total_episode_count"],
        "observed_episode_count": sum(len(rows) for rows in arm_rows.values()),
        "complete_matrix": all_complete,
        "implementation_hashes": sorted(implementation_hashes),
        "frozen_implementation_sha256": frozen_implementation,
        "implementation_match": implementation_match,
        "arms": arm_reports,
        "comparisons": comparisons,
        "operational_pass": operational_pass,
        "interpretation_boundary": protocol["interpretation_boundary"],
    }


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for arm, values in payload["arms"].items():
        rows.append(
            {
                "row_type": "arm",
                "name": arm,
                "episodes": values["episodes"],
                "mean_clipped_score": values["mean_clipped_score"],
                "task_success_rate": values["task_success_rate"],
                "score_difference": "",
                "score_ci95_low": "",
                "score_ci95_high": "",
                "success_difference": "",
                "success_ci95_low": "",
                "success_ci95_high": "",
                "holm_adjusted_p_value": "",
                "operational_pass": values["operational_pass"],
            }
        )
    for name, values in payload["comparisons"].items():
        score = values["clipped_score"]
        success = values["task_success_rate"]
        rows.append(
            {
                "row_type": "comparison",
                "name": name,
                "episodes": "",
                "mean_clipped_score": "",
                "task_success_rate": "",
                "score_difference": score["point_estimate"],
                "score_ci95_low": score["ci95_low"],
                "score_ci95_high": score["ci95_high"],
                "success_difference": success["point_estimate"],
                "success_ci95_low": success["ci95_low"],
                "success_ci95_high": success["ci95_high"],
                "holm_adjusted_p_value": values["score_permutation"][
                    "holm_adjusted_p_value"
                ],
                "operational_pass": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and analyze E16 ablations.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/scienceworld_protocol_ensr_mechanism_ablation_v1_frozen.json"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_mechanism_ablation_v1_runs/"
            "scienceworld_ensr_mechanism_ablation_v1/bit_qwen3_32b_clean/dev"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_mechanism_ablation_v1_audit.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_mechanism_ablation_v1_results.csv"
        ),
    )
    args = parser.parse_args()
    payload = analyze(
        json.loads(args.protocol.read_text(encoding="utf-8")),
        args.root,
        protocol_sha256=_sha256(args.protocol),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_csv, payload)
    print(
        json.dumps(
            {
                "complete_matrix": payload["complete_matrix"],
                "observed_episode_count": payload["observed_episode_count"],
                "operational_pass": payload["operational_pass"],
                "output_json": str(args.output_json.resolve()),
                "output_csv": str(args.output_csv.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if payload["complete_matrix"] and payload["operational_pass"] else 1)


if __name__ == "__main__":
    main()
