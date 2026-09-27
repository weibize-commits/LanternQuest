import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _efficiency_row(row: dict[str, Any]) -> dict[str, Any]:
    episodes = int(row["episodes"])
    successes = int(row["task_successes"])
    model_calls = int(row["model_calls"])
    input_tokens = int(row["input_tokens"])
    output_tokens = int(row["output_tokens"])
    total_tokens = input_tokens + output_tokens
    total_clipped_score = float(row["mean_clipped_score"]) * episodes
    return {
        "experiment": row["experiment"],
        "profile": row["profile"],
        "analysis": row["analysis"],
        "method": row["method"],
        "episodes": episodes,
        "successes": successes,
        "mean_clipped_score": float(row["mean_clipped_score"]),
        "environment_steps": int(row.get("environment_steps", 0)),
        "model_calls": model_calls,
        "retrieval_calls": int(row.get("retrieval_calls", 0)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "successes_per_100_model_calls": (
            100 * successes / model_calls if model_calls else 0.0
        ),
        "successes_per_million_tokens": (
            1_000_000 * successes / total_tokens if total_tokens else 0.0
        ),
        "clipped_score_per_100_model_calls": (
            100 * total_clipped_score / model_calls if model_calls else 0.0
        ),
        "currency_cost": None,
        "cost_note": (
            "Not estimated: the school MaaS endpoint does not expose a frozen, "
            "model-specific tariff in the experiment artifacts."
        ),
    }


def build(evidence: dict[str, Any], e16: dict[str, Any] | None) -> dict[str, Any]:
    rows = [
        _efficiency_row(row)
        for row in evidence["rows"]
        if row["method"] in {"iper_rag", "ensr_v2"}
    ]
    if e16 is not None and e16.get("complete_matrix"):
        for arm, values in e16["arms"].items():
            rows.append(
                _efficiency_row(
                    {
                        "experiment": "E16",
                        "profile": "bit_qwen3_32b_clean",
                        "analysis": "development_mechanism_ablation",
                        "method": f"ensr_v2:{arm}",
                        "episodes": values["episodes"],
                        "task_successes": values["successes"],
                        "mean_clipped_score": values["mean_clipped_score"],
                        "environment_steps": values["environment_steps"],
                        "model_calls": values["model_calls"],
                        "retrieval_calls": values["retrieval_calls"],
                        "input_tokens": values["input_tokens"],
                        "output_tokens": values["output_tokens"],
                    }
                )
            )

    comparisons: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["method"] not in {"iper_rag", "ensr_v2"}:
            continue
        key = (row["experiment"], row["profile"], row["analysis"])
        groups.setdefault(key, {})[row["method"]] = row
    for (experiment, profile, analysis), methods in sorted(groups.items()):
        if set(methods) != {"iper_rag", "ensr_v2"}:
            continue
        baseline = methods["iper_rag"]
        proposed = methods["ensr_v2"]
        success_gain = proposed["successes"] - baseline["successes"]
        comparisons.append(
            {
                "experiment": experiment,
                "profile": profile,
                "analysis": analysis,
                "ensr_minus_iper_mean_clipped_score": (
                    proposed["mean_clipped_score"] - baseline["mean_clipped_score"]
                ),
                "ensr_minus_iper_successes": success_gain,
                "ensr_minus_iper_model_calls": (
                    proposed["model_calls"] - baseline["model_calls"]
                ),
                "ensr_minus_iper_retrieval_calls": (
                    proposed["retrieval_calls"] - baseline["retrieval_calls"]
                ),
                "ensr_minus_iper_total_tokens": (
                    proposed["total_tokens"] - baseline["total_tokens"]
                ),
                "additional_model_calls_per_additional_success": (
                    (proposed["model_calls"] - baseline["model_calls"]) / success_gain
                    if success_gain > 0
                    else None
                ),
                "additional_tokens_per_additional_success": (
                    (proposed["total_tokens"] - baseline["total_tokens"]) / success_gain
                    if success_gain > 0
                    else None
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "descriptive_efficiency_accounting",
        "e16_status": (
            "complete" if e16 is not None and e16.get("complete_matrix") else "pending"
        ),
        "currency_cost_status": "not_estimable_from_frozen_artifacts",
        "rows": rows,
        "paired_system_comparisons": comparisons,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ScienceWorld efficiency tables.")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_evidence_table_v1.json"),
    )
    parser.add_argument(
        "--e16",
        type=Path,
        default=Path(
            "artifacts/benchmarks/scienceworld/ensr_mechanism_ablation_v1_audit.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_efficiency_summary_v1.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/benchmarks/scienceworld/ensr_efficiency_summary_v1.csv"),
    )
    args = parser.parse_args()
    e16 = json.loads(args.e16.read_text(encoding="utf-8")) if args.e16.is_file() else None
    payload = build(json.loads(args.evidence.read_text(encoding="utf-8")), e16)
    payload["source_sha256"] = {
        str(args.evidence): _sha256(args.evidence),
        **({str(args.e16): _sha256(args.e16)} if e16 is not None else {}),
    }
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_csv, payload["rows"])
    print(
        json.dumps(
            {
                "row_count": len(payload["rows"]),
                "comparison_count": len(payload["paired_system_comparisons"]),
                "e16_status": payload["e16_status"],
                "output_json": str(args.output_json.resolve()),
                "output_csv": str(args.output_csv.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
