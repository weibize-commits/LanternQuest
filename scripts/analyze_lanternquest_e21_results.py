from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from lanternquest.benchmark import LanternQuestBenchmarkCase, load_benchmark_jsonl
from lanternquest.e21 import E21CaseScore, aggregate_scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "lanternquest_e21_protocol_v1_draft.json"
DEFAULT_DATASET = ROOT / "annotations" / "benchmark" / "lanternquest_frozen_v1.jsonl"
DEFAULT_SCORES = ROOT / "artifacts" / "eval" / "lanternquest_e21_scores.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "eval" / "lanternquest_e21_analysis.json"
DEFAULT_CSV = ROOT / "artifacts" / "eval" / "lanternquest_e21_results.csv"
EXTERNAL_FAILURES = {"provider_error", "parser_error", "timeout"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    differences: list[float], *, samples: int, seed: int
) -> dict[str, float | int]:
    if not differences:
        raise ValueError("paired bootstrap requires at least one difference")
    rng = random.Random(seed)
    size = len(differences)
    estimates = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    return {
        "paired_case_count": size,
        "point_estimate": sum(differences) / size,
        "ci95_low": _percentile(estimates, 0.025),
        "ci95_high": _percentile(estimates, 0.975),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def permutation_p_value(
    differences: list[float], *, samples: int, seed: int
) -> dict[str, float | int | str]:
    if not differences:
        raise ValueError("paired permutation requires at least one difference")
    rng = random.Random(seed)
    observed = abs(sum(differences) / len(differences))
    extreme = 0
    for _ in range(samples):
        permuted = sum(
            value if rng.random() < 0.5 else -value for value in differences
        )
        if abs(permuted / len(differences)) >= observed - 1e-12:
            extreme += 1
    return {
        "method": "paired_random_sign_flip_two_sided_monte_carlo",
        "samples": samples,
        "seed": seed,
        "p_value": (extreme + 1) / (samples + 1),
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def read_scores(path: Path) -> tuple[list[E21CaseScore], list[str]]:
    if not path.is_file():
        return [], ["scores_missing"]
    scores = []
    errors = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                scores.append(E21CaseScore.model_validate_json(line))
            except ValidationError as error:
                errors.append(f"line {line_number}: {error}")
    return scores, errors


def analyze(
    protocol: dict[str, Any],
    cases: list[LanternQuestBenchmarkCase],
    scores: list[E21CaseScore],
    *,
    score_parse_errors: list[str] | None = None,
    bootstrap_samples: int | None = None,
    permutation_samples: int | None = None,
) -> dict[str, Any]:
    score_parse_errors = score_parse_errors or []
    statistics = protocol["statistics"]
    bootstrap_samples = bootstrap_samples or int(statistics["bootstrap_samples"])
    permutation_samples = permutation_samples or int(statistics["permutation_samples"])
    formal_split = str(protocol["dataset"]["formal_split"])
    formal_cases = {item.case_id: item for item in cases if item.split == formal_split}
    method_ids = [str(item["method_id"]) for item in protocol["methods"]]
    expected = {
        (case_id, method_id)
        for case_id in formal_cases
        for method_id in method_ids
    }
    formal_scores = [item for item in scores if item.split == formal_split]
    observed_keys = [(item.case_id, item.method) for item in formal_scores]
    counts = Counter(observed_keys)
    observed = set(observed_keys)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    split_mismatches = sorted(
        (item.case_id, item.method, item.split)
        for item in scores
        if item.case_id in formal_cases and item.split != formal_split
    )
    complete_matrix = bool(expected) and not any(
        (score_parse_errors, missing, unexpected, duplicates, split_mismatches)
    )
    by_key = {(item.case_id, item.method): item for item in formal_scores}
    aggregates = [
        item.model_dump(mode="json") for item in aggregate_scores(formal_scores)
    ]

    comparison_plan = protocol["comparison_plan"]
    proposed = str(comparison_plan["proposed_method_id"])
    selected_baseline = comparison_plan.get("selected_strongest_baseline_method_id")
    eligible = [str(item) for item in comparison_plan["eligible_baseline_method_ids"]]
    comparisons: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    if complete_matrix:
        for index, baseline in enumerate(eligible, start=1):
            success_differences = [
                float(by_key[(case_id, proposed)].grounded_task_success)
                - float(by_key[(case_id, baseline)].grounded_task_success)
                for case_id in sorted(formal_cases)
            ]
            evidence_differences = [
                by_key[(case_id, proposed)].evidence_coverage
                - by_key[(case_id, baseline)].evidence_coverage
                for case_id in sorted(formal_cases)
            ]
            intent_differences = [
                float(by_key[(case_id, proposed)].locked_intent_preserved)
                - float(by_key[(case_id, baseline)].locked_intent_preserved)
                for case_id in sorted(formal_cases)
            ]
            rework_reduction = [
                float(by_key[(case_id, baseline)].avoidable_rework_count)
                - float(by_key[(case_id, proposed)].avoidable_rework_count)
                for case_id in sorted(formal_cases)
            ]
            comparison_id = f"{proposed}_minus_{baseline}"
            permutation = permutation_p_value(
                success_differences,
                samples=permutation_samples,
                seed=int(statistics["permutation_seed"]) + index,
            )
            raw_p_values[comparison_id] = float(permutation["p_value"])
            comparisons[comparison_id] = {
                "is_preregistered_primary": baseline == selected_baseline,
                "grounded_task_success": paired_bootstrap(
                    success_differences,
                    samples=bootstrap_samples,
                    seed=int(statistics["bootstrap_seed"]) + index,
                ),
                "evidence_coverage": paired_bootstrap(
                    evidence_differences,
                    samples=bootstrap_samples,
                    seed=int(statistics["bootstrap_seed"]) + 100 + index,
                ),
                "locked_intent_preservation": paired_bootstrap(
                    intent_differences,
                    samples=bootstrap_samples,
                    seed=int(statistics["bootstrap_seed"]) + 200 + index,
                ),
                "avoidable_rework_reduction": paired_bootstrap(
                    rework_reduction,
                    samples=bootstrap_samples,
                    seed=int(statistics["bootstrap_seed"]) + 300 + index,
                ),
                "success_permutation": permutation,
            }
        for name, adjusted in holm_adjust(raw_p_values).items():
            comparisons[name]["success_permutation"][
                "holm_adjusted_p_value"
            ] = adjusted

    max_failure_rate = float(
        protocol["operational_gates"]["maximum_external_failure_rate_per_method"]
    )
    failure_rates = {}
    for method_id in method_ids:
        subset = [item for item in formal_scores if item.method == method_id]
        failures = sum(item.run_status in EXTERNAL_FAILURES for item in subset)
        failure_rates[method_id] = failures / len(subset) if subset else None
    missing_model_metadata = sorted(
        (item.case_id, item.method)
        for item in formal_scores
        if item.model_calls > 0
        and (not item.returned_model or not item.system_fingerprint)
    )
    metadata_required = bool(
        protocol["operational_gates"].get(
            "require_returned_model_and_system_fingerprint"
        )
    )
    model_metadata_pass = not metadata_required or not missing_model_metadata
    operational_pass = complete_matrix and all(
        rate is not None and rate <= max_failure_rate
        for rate in failure_rates.values()
    ) and model_metadata_pass
    primary_id = (
        f"{proposed}_minus_{selected_baseline}" if selected_baseline else None
    )
    primary_available = primary_id in comparisons if primary_id else False
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": protocol["experiment_id"],
        "protocol_id": protocol["protocol_id"],
        "status": "complete" if operational_pass and primary_available else "blocked",
        "complete_matrix": complete_matrix,
        "operational_pass": operational_pass,
        "formal_case_count": len(formal_cases),
        "method_count": len(method_ids),
        "expected_score_count": len(expected),
        "observed_score_count": len(formal_scores),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "duplicate_keys": duplicates,
        "split_mismatches": split_mismatches,
        "score_parse_errors": score_parse_errors,
        "external_failure_rates": failure_rates,
        "model_metadata_pass": model_metadata_pass,
        "missing_model_metadata": missing_model_metadata,
        "aggregates": aggregates,
        "primary_comparison_id": primary_id,
        "primary_comparison_available": primary_available,
        "comparisons": comparisons,
        "reporting_boundary": (
            "Only a complete, operationally valid matrix under a frozen protocol "
            "supports formal E21 performance claims."
        ),
    }


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for item in payload["aggregates"]:
        rows.append(
            {
                "row_type": "method",
                "name": item["method"],
                "episodes": item["episodes"],
                "grounded_task_success_rate": item[
                    "grounded_task_success_rate"
                ],
                "mean_evidence_coverage": item["mean_evidence_coverage"],
                "effect": "",
                "ci95_low": "",
                "ci95_high": "",
                "holm_adjusted_p_value": "",
            }
        )
    for name, item in payload["comparisons"].items():
        effect = item["grounded_task_success"]
        rows.append(
            {
                "row_type": "comparison",
                "name": name,
                "episodes": effect["paired_case_count"],
                "grounded_task_success_rate": "",
                "mean_evidence_coverage": "",
                "effect": effect["point_estimate"],
                "ci95_low": effect["ci95_low"],
                "ci95_high": effect["ci95_high"],
                "holm_adjusted_p_value": item["success_permutation"][
                    "holm_adjusted_p_value"
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "row_type",
            "name",
            "episodes",
            "grounded_task_success_rate",
            "mean_evidence_coverage",
            "effect",
            "ci95_low",
            "ci95_high",
            "holm_adjusted_p_value",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and analyze E21 scores")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    cases = []
    dataset_error = None
    try:
        cases = load_benchmark_jsonl(args.dataset)
    except (OSError, ValueError) as error:
        dataset_error = str(error)
    scores, score_errors = read_scores(args.scores)
    if dataset_error:
        score_errors.insert(0, f"dataset: {dataset_error}")
    payload = analyze(
        protocol,
        cases,
        scores,
        score_parse_errors=score_errors,
    )
    payload["source_sha256"] = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in (args.protocol, args.dataset, args.scores)
        if path.is_file()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, payload)
    print(
        f"wrote {args.output}: status={payload['status']}, "
        f"scores={payload['observed_score_count']}/{payload['expected_score_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
