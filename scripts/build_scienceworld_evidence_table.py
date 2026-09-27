import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METHODS = ("iper_rag", "ensr_v2")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _method_row(
    *,
    experiment: str,
    profile: str,
    analysis: str,
    method: str,
    summary: dict[str, Any],
    effect: dict[str, Any],
    experiment_pass: bool,
) -> dict[str, Any]:
    calls = int(summary.get("model_calls", 0))
    successes = round(
        float(summary.get("task_success_rate", 0.0))
        * int(summary.get("episodes", 0))
    )
    external_failures = int(
        summary.get(
            "external_failures",
            summary.get("provider_or_environment_failures", 0),
        )
    )
    score_effect = effect["clipped_score"]
    success_effect = effect["task_success_rate"]
    return {
        "experiment": experiment,
        "profile": profile,
        "analysis": analysis,
        "method": method,
        "episodes": int(summary.get("episodes", 0)),
        "mean_clipped_score": float(summary.get("mean_clipped_score", 0.0)),
        "task_successes": successes,
        "task_success_rate": float(summary.get("task_success_rate", 0.0)),
        "external_failures": external_failures,
        "environment_steps": int(summary.get("environment_steps", 0)),
        "model_calls": calls,
        "retrieval_calls": int(summary.get("retrieval_calls", 0)),
        "input_tokens": int(summary.get("input_tokens", 0)),
        "output_tokens": int(summary.get("output_tokens", 0)),
        "total_tokens": int(summary.get("input_tokens", 0))
        + int(summary.get("output_tokens", 0)),
        "wall_seconds": summary.get("wall_seconds", ""),
        "successes_per_100_model_calls": (
            100 * successes / calls if calls else 0.0
        ),
        "ensr_minus_iper_score": float(score_effect["point_estimate"]),
        "score_ci95_low": float(score_effect["ci95_low"]),
        "score_ci95_high": float(score_effect["ci95_high"]),
        "ensr_minus_iper_success_rate": float(success_effect["point_estimate"]),
        "success_ci95_low": float(success_effect["ci95_low"]),
        "success_ci95_high": float(success_effect["ci95_high"]),
        "experiment_pass": bool(experiment_pass),
    }


def _effect(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "clipped_score": payload["clipped_score"],
        "task_success_rate": payload["task_success_rate"],
    }


def build(
    e13: dict[str, Any],
    e14: dict[str, Any],
    e14_sensitivity: dict[str, Any],
    e15: dict[str, Any] | None,
    e15r: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    e13_effect = _effect(e13["ensr_minus_baseline"])
    for method in METHODS:
        rows.append(
            _method_row(
                experiment="E13",
                profile="bit_qwen3_235b",
                analysis="confirmatory_all_pairs",
                method=method,
                summary=e13["summaries"][method],
                effect=e13_effect,
                experiment_pass=bool(e13["formal_full_success"]),
            )
        )

    sensitivity_models = e14_sensitivity["models"]
    for profile, model in e14["models"].items():
        effect = _effect(model["ensr_minus_baseline"])
        for method in METHODS:
            rows.append(
                _method_row(
                    experiment="E14",
                    profile=profile,
                    analysis="cross_model_all_pairs",
                    method=method,
                    summary=model["summaries"][method],
                    effect=effect,
                    experiment_pass=bool(model["formal_full_success"]),
                )
            )
        sensitivity = sensitivity_models[profile]
        clean = sensitivity["clean_complete_case_pairs"]
        rows.append(
            {
                "experiment": "E14-SENS-1",
                "profile": profile,
                "analysis": "clean_complete_case_effect",
                "method": "ensr_v2_minus_iper_rag",
                "episodes": int(clean["pair_count"]),
                "mean_clipped_score": "",
                "task_successes": "",
                "task_success_rate": "",
                "external_failures": int(sensitivity["excluded_pair_count"]),
                "environment_steps": "",
                "model_calls": "",
                "retrieval_calls": "",
                "input_tokens": "",
                "output_tokens": "",
                "total_tokens": "",
                "wall_seconds": "",
                "successes_per_100_model_calls": "",
                "ensr_minus_iper_score": float(
                    clean["ensr_minus_iper_score"]["point_estimate"]
                ),
                "score_ci95_low": float(
                    clean["ensr_minus_iper_score"]["ci95_low"]
                ),
                "score_ci95_high": float(
                    clean["ensr_minus_iper_score"]["ci95_high"]
                ),
                "ensr_minus_iper_success_rate": float(
                    clean["ensr_minus_iper_success_rate"]["point_estimate"]
                ),
                "success_ci95_low": float(
                    clean["ensr_minus_iper_success_rate"]["ci95_low"]
                ),
                "success_ci95_high": float(
                    clean["ensr_minus_iper_success_rate"]["ci95_high"]
                ),
                "experiment_pass": bool(sensitivity["clean_direction_preserved"]),
            }
        )

    if e15 is not None:
        for profile, model in e15["models"].items():
            effect = _effect(model["ensr_minus_iper"])
            for method in METHODS:
                rows.append(
                    _method_row(
                        experiment="E15",
                        profile=profile,
                        analysis="clean_adapter_all_pairs",
                        method=method,
                        summary=model["summaries"][method],
                        effect=effect,
                        experiment_pass=bool(model["profile_pass"]),
                    )
                )

    if e15r is not None:
        effect = _effect(e15r["ensr_minus_iper"])
        for method in METHODS:
            rows.append(
                _method_row(
                    experiment="E15R",
                    profile="bit_qwen3_8b_recovery_v1",
                    analysis="prospective_operational_recovery",
                    method=method,
                    summary=e15r["summaries"][method],
                    effect=effect,
                    experiment_pass=bool(e15r["recovery_pass"]),
                )
            )

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "e15_status": "complete" if e15 is not None else "pending",
        "e15r_status": "complete" if e15r is not None else "pending",
        "rows": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the E13-E15 evidence table.")
    parser.add_argument(
        "--e13",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_v3_test_audit.json"),
    )
    parser.add_argument(
        "--e14",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_cross_model_v1_audit.json"),
    )
    parser.add_argument(
        "--e14-sensitivity",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_cross_model_v1_sensitivity.json"
        ),
    )
    parser.add_argument(
        "--e15",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_clean_adapter_v1_audit.json"),
    )
    parser.add_argument(
        "--e15r",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/qwen3_8b_recovery_v1_audit.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_evidence_table_v1.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_evidence_table_v1.csv"),
    )
    args = parser.parse_args()
    e15 = _load(args.e15) if args.e15.is_file() else None
    e15r = _load(args.e15r) if args.e15r.is_file() else None
    payload = build(
        _load(args.e13),
        _load(args.e14),
        _load(args.e14_sensitivity),
        e15,
        e15r,
    )
    payload["source_sha256"] = {
        str(args.e13): _hash(args.e13),
        str(args.e14): _hash(args.e14),
        str(args.e14_sensitivity): _hash(args.e14_sensitivity),
        **({str(args.e15): _hash(args.e15)} if e15 is not None else {}),
        **({str(args.e15r): _hash(args.e15r)} if e15r is not None else {}),
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, payload["rows"])
    print(
        json.dumps(
            {
                "e15_status": payload["e15_status"],
                "e15r_status": payload["e15r_status"],
                "row_count": len(payload["rows"]),
                "output_json": str(args.output_json.resolve()),
                "output_csv": str(args.output_csv.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
