from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from lanternquest.dynamic_eval import DynamicCaseScore

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("full_ENSR", "ENSR_base", "B4")
BINARY_METRICS = (
    "terminal_success",
    "persistent_obligation_resolved_correctly",
    "unsupported_action",
    "repair_locality_success",
    "evidence_trace_complete",
)
COUNT_METRICS = ("avoidable_rework_count", "retrieval_calls")


def load_scores(path: Path) -> list[DynamicCaseScore]:
    with path.open(encoding="utf-8-sig") as handle:
        return [DynamicCaseScore.model_validate_json(line) for line in handle if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of empty data")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    differences: list[float], *, seed: int, replicates: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    boot = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates)
    ]
    return percentile(boot, 0.025), percentile(boot, 0.975)


def exact_two_sided_discordant_p(better: int, worse: int) -> float:
    total = better + worse
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(0, min(better, worse) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def metric_summary(rows: list[DynamicCaseScore], metric: str) -> dict[str, Any]:
    by_method = {
        method: [float(getattr(row, metric)) for row in rows if row.method == method]
        for method in METHODS
    }
    return {
        method: {
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "n": len(values),
        }
        for method, values in by_method.items()
    }


def paired_comparison(
    rows: list[DynamicCaseScore],
    metric: str,
    comparator: str,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    index = {(row.case_id, row.method): row for row in rows}
    case_ids = sorted({row.case_id for row in rows})
    differences = [
        float(getattr(index[(case_id, "full_ENSR")], metric))
        - float(getattr(index[(case_id, comparator)], metric))
        for case_id in case_ids
    ]
    low, high = paired_bootstrap(differences, seed=seed, replicates=replicates)
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    tied = sum(value == 0 for value in differences)
    return {
        "comparison": f"full_ENSR-minus-{comparator}",
        "absolute_paired_difference": sum(differences) / len(differences),
        "paired_case_bootstrap_95_ci": [low, high],
        "discordant_positive_difference": positive,
        "discordant_negative_difference": negative,
        "ties": tied,
        "exact_two_sided_discordant_p": exact_two_sided_discordant_p(positive, negative),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze dynamic controller test results")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=27103)
    args = parser.parse_args()

    rows = load_scores(args.scores)
    if len(rows) != 30 or {row.split for row in rows} != {"test"}:
        raise ValueError("formal analysis requires 30 test scores")
    keys = {(row.case_id, row.method) for row in rows}
    if len(keys) != 30:
        raise ValueError("duplicate or missing case-method scores")
    if any(row.run_status != "ok" for row in rows):
        raise ValueError("formal result contains failed runs")

    metrics: dict[str, Any] = {}
    for offset, metric in enumerate(BINARY_METRICS + COUNT_METRICS):
        metrics[metric] = {
            "method_summary": metric_summary(rows, metric),
            "paired_comparisons": [
                paired_comparison(
                    rows,
                    metric,
                    comparator,
                    seed=args.seed + 10 * offset + index,
                    replicates=args.bootstrap_replicates,
                )
                for index, comparator in enumerate(("ENSR_base", "B4"), start=1)
            ],
        }

    by_type: dict[str, Any] = {}
    types = sorted({row.perturbation_type for row in rows})
    for perturbation in types:
        subset = [row for row in rows if row.perturbation_type == perturbation]
        by_type[perturbation] = {
            method: {
                "n": sum(row.method == method for row in subset),
                "terminal_success": sum(
                    row.terminal_success for row in subset if row.method == method
                ),
                "persistent_obligation_resolved_correctly": sum(
                    row.persistent_obligation_resolved_correctly
                    for row in subset
                    if row.method == method
                ),
                "unsupported_action": sum(
                    row.unsupported_action for row in subset if row.method == method
                ),
            }
            for method in METHODS
        }

    summary = {
        metric: metrics[metric]["method_summary"] for metric in metrics
    }
    full_obligation = summary["persistent_obligation_resolved_correctly"]["full_ENSR"]["mean"]
    base_obligation = summary["persistent_obligation_resolved_correctly"]["ENSR_base"]["mean"]
    full_unsupported = summary["unsupported_action"]["full_ENSR"]["mean"]
    base_unsupported = summary["unsupported_action"]["ENSR_base"]["mean"]
    full_terminal = summary["terminal_success"]["full_ENSR"]["mean"]
    base_terminal = summary["terminal_success"]["ENSR_base"]["mean"]
    full_locality = summary["repair_locality_success"]["full_ENSR"]["mean"]
    base_locality = summary["repair_locality_success"]["ENSR_base"]["mean"]
    mechanism_supported = (
        full_obligation > base_obligation
        and full_unsupported <= base_unsupported
        and (full_terminal > base_terminal or full_locality > base_locality)
    )
    claim = (
        "incremental_ENSR_mechanism_supported"
        if mechanism_supported
        else (
            "package_only_claim"
            if full_terminal > summary["terminal_success"]["B4"]["mean"]
            and full_obligation == base_obligation
            else "mechanism_not_supported"
        )
    )
    result = {
        "schema_version": "1.0",
        "analysis_population": "frozen test split",
        "case_count": 10,
        "method_count": 3,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "metrics": metrics,
        "by_perturbation_type": by_type,
        "prespecified_claim_decision": claim,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "formal_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "case_level_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "scenario_family",
                "perturbation_type",
                "method",
                *BINARY_METRICS,
                *COUNT_METRICS,
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.case_id, item.method)):
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "scenario_family": row.scenario_family,
                    "perturbation_type": row.perturbation_type,
                    "method": row.method,
                    **{
                        metric: getattr(row, metric)
                        for metric in BINARY_METRICS + COUNT_METRICS
                    },
                }
            )

    lines = [
        "# LanternQuest dynamic heritage benchmark v2.1",
        "",
        f"Prespecified decision: **{claim}**",
        "",
        "| Method | Terminal success | Correct obligation state | Unsupported action | Repair locality |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            "| {method} | {terminal:.0%} | {obligation:.0%} | {unsupported:.0%} | {locality:.0%} |".format(
                method=method,
                terminal=summary["terminal_success"][method]["mean"],
                obligation=summary["persistent_obligation_resolved_correctly"][method]["mean"],
                unsupported=summary["unsupported_action"][method]["mean"],
                locality=summary["repair_locality_success"][method]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "The neural proposal and pre-event action trace were held fixed. The result therefore isolates controller behavior after a dynamic transition.",
            "",
        ]
    )
    (args.output_dir / "formal_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
