from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

DATA = Path("data/dynamic_benchmark/case_level_metrics_anonymized.csv")
METHODS = ("full_ENSR", "ENSR_base", "B4")
METRICS = (
    "terminal_success",
    "persistent_obligation_resolved_correctly",
    "unsupported_action",
    "repair_locality_success",
    "evidence_trace_complete",
    "avoidable_rework_count",
    "retrieval_calls",
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def exact_two_sided(positive: int, negative: int) -> float:
    total = positive + negative
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2**total))


with DATA.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
index = {(row["public_case_id"], row["method"]): row for row in rows}
cases = sorted({row["public_case_id"] for row in rows})
result = {}
for offset, metric in enumerate(METRICS):
    result[metric] = {"method_summary": {}, "paired_comparisons": []}
    for method in METHODS:
        values = [float(index[(case_id, method)][metric]) for case_id in cases]
        result[metric]["method_summary"][method] = {
            "sum": sum(values), "mean": sum(values) / len(values), "n": len(values)
        }
    for comparator_index, comparator in enumerate(("ENSR_base", "B4"), start=1):
        differences = [
            float(index[(case_id, "full_ENSR")][metric])
            - float(index[(case_id, comparator)][metric])
            for case_id in cases
        ]
        rng = random.Random(27103 + 10 * offset + comparator_index)
        boot = [
            sum(differences[rng.randrange(len(cases))] for _ in cases) / len(cases)
            for _ in range(10000)
        ]
        positive = sum(value > 0 for value in differences)
        negative = sum(value < 0 for value in differences)
        result[metric]["paired_comparisons"].append({
            "comparison": f"full_ENSR-minus-{comparator}",
            "absolute_paired_difference": sum(differences) / len(differences),
            "paired_case_bootstrap_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
            "discordant_positive_difference": positive,
            "discordant_negative_difference": negative,
            "ties": sum(value == 0 for value in differences),
            "exact_two_sided_discordant_p": exact_two_sided(positive, negative),
        })
print(json.dumps(result, indent=2))
