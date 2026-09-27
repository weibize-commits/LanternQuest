import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_scienceworld_ensr_results import paired_bootstrap
except ModuleNotFoundError:  # Direct script execution uses the scripts directory.
    from analyze_scienceworld_ensr_results import paired_bootstrap


PROFILES = (
    "bit_ceep_70b_clean",
    "bit_qwen3_32b_clean",
    "bit_qwen3_8b_clean",
)
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


def _load_method_rows(root: Path, method: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
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
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    return rows, failures


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = sum(int(row["model_calls"]) for row in rows)
    successes = sum(bool(row["task_success"]) for row in rows)
    normalizations: Counter[str] = Counter()
    for row in rows:
        normalizations.update(
            {
                str(name): int(count)
                for name, count in row.get("response_normalizations", {}).items()
            }
        )
    return {
        "episodes": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "mean_clipped_score": _mean(
            [max(0.0, float(row["final_score"])) for row in rows]
        ),
        "task_successes": successes,
        "task_success_rate": successes / len(rows) if rows else 0.0,
        "environment_steps": sum(int(row["environment_steps"]) for row in rows),
        "model_calls": calls,
        "retrieval_calls": sum(int(row["retrieval_calls"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "external_failures": sum(row["status"] in FAILURE_STATUSES for row in rows),
        "successes_per_100_model_calls": 100 * successes / calls if calls else 0.0,
        "response_normalizations": dict(sorted(normalizations.items())),
    }


def analyze(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    seed = int(protocol["clean_adapter_design"]["seed"])
    selected = {
        (str(item["task_id"]), int(item["test"]))
        for item in protocol["selected_variations"]
    }
    models: dict[str, Any] = {}
    pooled_iper_score: dict[str, float] = {}
    pooled_ensr_score: dict[str, float] = {}
    pooled_iper_success: dict[str, float] = {}
    pooled_ensr_success: dict[str, float] = {}
    for profile in PROFILES:
        profile_root = run_root / profile / "test"
        rows_by_method: dict[str, list[dict[str, Any]]] = {}
        parse_failures: list[str] = []
        for method in METHODS:
            rows, failures = _load_method_rows(profile_root, method)
            rows_by_method[method] = [
                row for row in rows if int(row.get("seed", -1)) == seed
            ]
            parse_failures.extend(failures)

        expected = {
            (method, task_id, variation, seed)
            for method in METHODS
            for task_id, variation in selected
        }
        observed_list = [
            (
                method,
                str(row["task_id"]),
                int(row["variation"]),
                int(row["seed"]),
            )
            for method, rows in rows_by_method.items()
            for row in rows
        ]
        observed = set(observed_list)
        duplicate_keys = sorted(
            key for key, count in Counter(observed_list).items() if count > 1
        )
        missing_keys = sorted(expected - observed)
        unexpected_keys = sorted(observed - expected)
        complete = not (
            missing_keys or unexpected_keys or duplicate_keys or parse_failures
        )
        summaries = {
            method: _summarize(rows_by_method[method]) for method in METHODS
        }

        task_values: dict[str, dict[str, dict[str, float]]] = {}
        for method in METHODS:
            task_values[method] = {
                str(row["task_id"]): {
                    "score": max(0.0, float(row["final_score"])),
                    "success": float(bool(row["task_success"])),
                }
                for row in rows_by_method[method]
            }
        score_effect = paired_bootstrap(
            {task: values["score"] for task, values in task_values["ensr_v2"].items()},
            {task: values["score"] for task, values in task_values["iper_rag"].items()},
        )
        success_effect = paired_bootstrap(
            {
                task: values["success"]
                for task, values in task_values["ensr_v2"].items()
            },
            {
                task: values["success"]
                for task, values in task_values["iper_rag"].items()
            },
            seed=15027,
        )
        operational_cells = {
            method: summaries[method]["external_failures"] <= 1 for method in METHODS
        }
        operationally_valid = all(operational_cells.values())
        direction_pass = (
            score_effect["point_estimate"] > 0
            and success_effect["point_estimate"] >= 0
        )
        models[profile] = {
            "complete_matrix": complete,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "duplicate_keys": duplicate_keys,
            "parse_failures": parse_failures,
            "summaries": summaries,
            "ensr_minus_iper": {
                "clipped_score": score_effect,
                "task_success_rate": success_effect,
            },
            "operational_cells": operational_cells,
            "operationally_valid": operationally_valid,
            "direction_pass": direction_pass,
            "profile_pass": complete and operationally_valid and direction_pass,
        }
        for task_id in task_values["iper_rag"]:
            pooled_key = f"{profile}:{task_id}"
            pooled_iper_score[pooled_key] = task_values["iper_rag"][task_id]["score"]
            pooled_ensr_score[pooled_key] = task_values["ensr_v2"][task_id]["score"]
            pooled_iper_success[pooled_key] = task_values["iper_rag"][task_id][
                "success"
            ]
            pooled_ensr_success[pooled_key] = task_values["ensr_v2"][task_id][
                "success"
            ]

    complete = all(model["complete_matrix"] for model in models.values())
    passed = [profile for profile, model in models.items() if model["profile_pass"]]
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "E15",
        "protocol_id": protocol["protocol_id"],
        "profile_count": len(PROFILES),
        "expected_episode_count": len(PROFILES) * len(METHODS) * len(selected),
        "observed_episode_count": sum(
            model["summaries"][method]["episodes"]
            for model in models.values()
            for method in METHODS
        ),
        "complete_all_profiles": complete,
        "models": models,
        "pooled_ensr_minus_iper": {
            "clipped_score": paired_bootstrap(
                pooled_ensr_score, pooled_iper_score, seed=15028
            ),
            "task_success_rate": paired_bootstrap(
                pooled_ensr_success, pooled_iper_success, seed=15029
            ),
        },
        "clean_adapter_summary": {
            "passing_profiles": passed,
            "passing_profile_count": len(passed),
            "required_profile_count": len(PROFILES),
            "clean_adapter_pass": complete and len(passed) == len(PROFILES),
        },
        "reporting_boundary": protocol["reporting_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen E15 result matrix.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/scienceworld_protocol_ensr_clean_adapter_v1_frozen.json"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/"
            "ensr_clean_adapter_v1_runs/scienceworld_ensr_v3"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_clean_adapter_v1_audit.json"
        ),
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    payload = analyze(protocol, args.run_root)
    _write_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
