import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

METHODS = ("b4_sequential_planner", "iper_rag", "ensr_v2")
BASELINES = METHODS[:2]
FAILURE_STATUSES = {"adapter_error", "environment_error"}


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _selected_cases(protocol: dict[str, Any], split: str) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for selection in protocol["selected_variations"]:
        variations = (
            selection["dev_variations"] if split == "dev" else [selection["test"]]
        )
        cases.extend((selection["task_id"], int(item)) for item in variations)
    return cases


def _load_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    for method in METHODS:
        for path in sorted((root / method).glob("*.json")):
            if path.name.startswith("summary_"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("method") != method:
                    raise ValueError(
                        f"method mismatch: expected {method}, got {payload.get('method')}"
                    )
                payload["_path"] = str(path.resolve())
                rows.append(payload)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                parse_failures.append(f"{path}: {type(exc).__name__}: {exc}")
    return rows, parse_failures


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    ensr_by_task: dict[str, float],
    baseline_by_task: dict[str, float],
    *,
    samples: int = 20_000,
    seed: int = 12026,
) -> dict[str, float | int]:
    task_ids = sorted(set(ensr_by_task) & set(baseline_by_task))
    if not task_ids:
        return {
            "paired_task_count": 0,
            "point_estimate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        }
    differences = [ensr_by_task[item] - baseline_by_task[item] for item in task_ids]
    rng = random.Random(seed)
    replicates = [
        _mean([differences[rng.randrange(len(differences))] for _ in differences])
        for _ in range(samples)
    ]
    return {
        "paired_task_count": len(task_ids),
        "point_estimate": _mean(differences),
        "ci95_low": _quantile(replicates, 0.025),
        "ci95_high": _quantile(replicates, 0.975),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def analyze(
    *,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    split: str,
    seeds: list[int],
    parse_failures: list[str] | None = None,
) -> dict[str, Any]:
    cases = _selected_cases(protocol, split)
    expected_keys = {
        (method, task_id, variation, seed)
        for method in METHODS
        for task_id, variation in cases
        for seed in seeds
    }
    observed_keys: list[tuple[str, str, int, int]] = [
        (
            str(row["method"]),
            str(row["task_id"]),
            int(row["variation"]),
            int(row["seed"]),
        )
        for row in rows
    ]
    duplicate_keys = sorted(
        key for key, count in Counter(observed_keys).items() if count > 1
    )
    observed_set = set(observed_keys)
    missing_keys = sorted(expected_keys - observed_set)
    unexpected_keys = sorted(observed_set - expected_keys)

    summaries: dict[str, Any] = {}
    task_values: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summaries[method] = {
            "episodes": len(method_rows),
            "status_counts": dict(
                sorted(Counter(str(row["status"]) for row in method_rows).items())
            ),
            "mean_clipped_score": _mean(
                [max(0, float(row["final_score"])) for row in method_rows]
            ),
            "task_success_rate": _mean(
                [float(bool(row["task_success"])) for row in method_rows]
            ),
            "environment_steps": sum(
                int(row["environment_steps"]) for row in method_rows
            ),
            "model_calls": sum(int(row["model_calls"]) for row in method_rows),
            "retrieval_calls": sum(
                int(row["retrieval_calls"]) for row in method_rows
            ),
            "input_tokens": sum(int(row["input_tokens"]) for row in method_rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in method_rows),
            "provider_or_environment_failures": sum(
                str(row["status"]) in FAILURE_STATUSES for row in method_rows
            ),
        }
        by_task: dict[str, list[dict[str, Any]]] = {}
        for row in method_rows:
            by_task.setdefault(str(row["task_id"]), []).append(row)
        task_values[method] = {
            task_id: {
                "clipped_score": _mean(
                    [max(0, float(row["final_score"])) for row in task_rows]
                ),
                "success": _mean(
                    [float(bool(row["task_success"])) for row in task_rows]
                ),
            }
            for task_id, task_rows in by_task.items()
        }

    if split == "test":
        baseline = protocol.get("primary_baseline")
        if baseline not in BASELINES:
            raise ValueError(
                "Frozen test analysis requires primary_baseline selected on dev"
            )
    else:
        baseline = max(
            BASELINES,
            key=lambda item: (
                summaries[item]["mean_clipped_score"],
                summaries[item]["task_success_rate"],
                item,
            ),
        )

    score_bootstrap = paired_bootstrap(
        {
            task: values["clipped_score"]
            for task, values in task_values["ensr_v2"].items()
        },
        {
            task: values["clipped_score"]
            for task, values in task_values[baseline].items()
        },
    )
    success_bootstrap = paired_bootstrap(
        {
            task: values["success"]
            for task, values in task_values["ensr_v2"].items()
        },
        {
            task: values["success"]
            for task, values in task_values[baseline].items()
        },
        seed=12027,
    )
    ensr_rows = [row for row in rows if row["method"] == "ensr_v2"]
    mechanism_totals = {
        metric: sum(
            float(row.get("mechanism_metrics", {}).get(metric, 0))
            for row in ensr_rows
        )
        for metric in (
            "verified_transition_count",
            "obligation_count",
            "local_replan_count",
            "local_replan_recovery_count",
            "adaptive_continuation_plan_count",
            "adaptive_continuation_recovery_count",
            "adaptive_slow_action_count",
            "knowledge_graph_action_count",
            "semantic_focus_action_count",
            "semantic_acquisition_action_count",
            "semantic_delivery_action_count",
            "contradictory_fact_rejection_count",
            "no_progress_action_count",
        )
    }
    parse_failures = parse_failures or []
    complete_matrix = not (
        missing_keys or unexpected_keys or duplicate_keys or parse_failures
    )
    ensr_error_rate = (
        summaries["ensr_v2"]["provider_or_environment_failures"]
        / summaries["ensr_v2"]["episodes"]
        if summaries["ensr_v2"]["episodes"]
        else 1.0
    )
    dev_gates = {
        "complete_matrix": complete_matrix,
        "verified_transition_nonzero": mechanism_totals[
            "verified_transition_count"
        ]
        > 0,
        "obligation_nonzero": mechanism_totals["obligation_count"] > 0,
        "local_replan_nonzero": mechanism_totals["local_replan_count"] > 0,
        "adapter_or_environment_error_rate_below_5_percent": ensr_error_rate
        < 0.05,
        "ensr_score_above_strongest_baseline": score_bootstrap["point_estimate"]
        > 0,
    }
    formal_unlock_recommended = split == "dev" and all(dev_gates.values())
    formal_full_success = (
        split == "test"
        and complete_matrix
        and score_bootstrap["ci95_low"] > 0
        and success_bootstrap["point_estimate"] >= 0
    )
    return {
        "schema_version": "2.0",
        "experiment_id": protocol["experiment_id"],
        "protocol_id": protocol["protocol_id"],
        "split": split,
        "expected_episode_count": len(expected_keys),
        "observed_episode_count": len(rows),
        "complete_matrix": complete_matrix,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "duplicate_keys": duplicate_keys,
        "parse_failures": parse_failures,
        "summaries": summaries,
        "selected_baseline": baseline,
        "ensr_minus_baseline": {
            "clipped_score": score_bootstrap,
            "task_success_rate": success_bootstrap,
        },
        "ensr_mechanism_totals": mechanism_totals,
        "ensr_provider_or_environment_error_rate": ensr_error_rate,
        "development_gates": dev_gates if split == "dev" else None,
        "formal_unlock_recommended": formal_unlock_recommended,
        "formal_full_success": formal_full_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit E12 completeness, mechanisms, paired gains, and unlock gates."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/scienceworld_protocol_ensr_v2_draft.json"),
    )
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    formal_seeds = [int(seed) for seed in protocol["runs"]["formal_seeds"]]
    seeds = (
        [int(seed) for seed in args.seeds.split(",") if seed.strip()]
        if args.seeds
        else formal_seeds
        if args.split == "test"
        else formal_seeds[:1]
    )
    rows, parse_failures = _load_rows(args.root)
    payload = analyze(
        rows=rows,
        protocol=protocol,
        split=args.split,
        seeds=seeds,
        parse_failures=parse_failures,
    )
    _write_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
