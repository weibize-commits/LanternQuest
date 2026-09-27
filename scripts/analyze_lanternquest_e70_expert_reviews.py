from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RATING_FIELDS = (
    "domain_correctness_1_5",
    "evidence_sufficiency_1_5",
    "locked_intent_preserved",
    "unsupported_or_unsafe_claim_count",
)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def load_key(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    key: dict[str, dict[str, str]] = {}
    for row in rows:
        review_id = row.get("review_id", "").strip()
        if not review_id or review_id in key:
            raise ValueError("private key contains a blank or duplicate review_id")
        if row.get("system_x_method") == row.get("system_y_method"):
            raise ValueError(f"private key repeats a method for {review_id}")
        key[review_id] = row
    if not key:
        raise ValueError("private key is empty")
    return key


def parse_score(row: dict[str, str], side: str, stem: str) -> float | str:
    value = row[f"{side}_{stem}"].strip()
    if stem == "locked_intent_preserved":
        if value not in {"yes", "no", "unclear"}:
            raise ValueError(f"invalid intent-preservation value: {value}")
        return value
    number = float(value)
    if stem in {"domain_correctness_1_5", "evidence_sufficiency_1_5"}:
        if not number.is_integer() or not 1 <= number <= 5:
            raise ValueError(f"rating outside 1..5: {value}")
    elif not number.is_integer() or number < 0:
        raise ValueError(f"claim count must be a nonnegative integer: {value}")
    return number


def load_ratings(
    paths: list[Path], key: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], list[str]]:
    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    reviewer_seen: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for line_number, row in enumerate(rows, start=2):
            try:
                reviewer = row.get("reviewer_code", "").strip()
                review_id = row.get("review_id", "").strip()
                if not reviewer or review_id not in key:
                    raise ValueError("unknown reviewer or review_id")
                if review_id in reviewer_seen[reviewer]:
                    raise ValueError("duplicate reviewer-review_id row")
                reviewer_seen[reviewer].add(review_id)
                if row.get("case_id", "").strip() != key[review_id]["case_id"]:
                    raise ValueError("case_id does not match private key")
                gradable = parse_bool(row.get("gradable", ""))
                if not gradable:
                    if not row.get("unable_reason", "").strip():
                        raise ValueError("ungradable row requires unable_reason")
                    parsed.append(
                        {
                            "reviewer_code": reviewer,
                            "review_id": review_id,
                            "case_id": key[review_id]["case_id"],
                            "gradable": False,
                            "unable_reason": row["unable_reason"].strip(),
                        }
                    )
                    continue
                confidence = int(row.get("confidence_1_5", ""))
                if not 1 <= confidence <= 5:
                    raise ValueError("confidence must be 1..5")
                preference = row.get("preference", "").strip()
                if preference not in {"X", "Y", "tie", "unable"}:
                    raise ValueError("invalid preference")
                values = {
                    side: {
                        stem: parse_score(row, side, stem) for stem in RATING_FIELDS
                    }
                    for side in ("x", "y")
                }
                parsed.append(
                    {
                        "reviewer_code": reviewer,
                        "review_id": review_id,
                        "case_id": key[review_id]["case_id"],
                        "gradable": True,
                        "confidence": confidence,
                        "preference": preference,
                        "values": values,
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{path}:{line_number}: {error}")
    expected = set(key)
    for reviewer, observed in sorted(reviewer_seen.items()):
        missing = sorted(expected - observed)
        if missing:
            errors.append(f"reviewer {reviewer} is missing {len(missing)} review rows")
    return parsed, errors


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_mean_ci(
    values: list[float], *, iterations: int = 20_000, seed: int = 20260923
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    draws = [
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(iterations)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def icc2k(matrix: list[list[float]]) -> float | None:
    if len(matrix) < 2 or not matrix or len(matrix[0]) < 2:
        return None
    k = len(matrix[0])
    if any(len(row) != k for row in matrix):
        return None
    n = len(matrix)
    grand = sum(sum(row) for row in matrix) / (n * k)
    row_means = [sum(row) / k for row in matrix]
    column_means = [sum(row[index] for row in matrix) / n for index in range(k)]
    ms_rows = k * sum((value - grand) ** 2 for value in row_means) / (n - 1)
    ms_columns = n * sum((value - grand) ** 2 for value in column_means) / (k - 1)
    residual = sum(
        (matrix[i][j] - row_means[i] - column_means[j] + grand) ** 2
        for i in range(n)
        for j in range(k)
    )
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = ms_rows + (ms_columns - ms_error) / n
    return None if denominator == 0 else (ms_rows - ms_error) / denominator


def analyze(
    rows: list[dict[str, Any]],
    key: dict[str, dict[str, str]],
    *,
    target_method: str,
    baseline_method: str,
) -> dict[str, Any]:
    by_case_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    preferences: Counter[str] = Counter()
    long_rows = []
    for row in rows:
        if not row["gradable"]:
            continue
        mapping = key[row["review_id"]]
        side_to_method = {
            "x": mapping["system_x_method"],
            "y": mapping["system_y_method"],
        }
        preference = row["preference"]
        preferences[
            side_to_method[preference.lower()]
            if preference in {"X", "Y"}
            else preference
        ] += 1
        for side, method in side_to_method.items():
            values = row["values"][side]
            item = {
                "reviewer_code": row["reviewer_code"],
                "case_id": row["case_id"],
                "method": method,
                **values,
            }
            by_case_method[(row["case_id"], method)].append(item)
            long_rows.append(item)

    case_ids = sorted({row["case_id"] for row in rows})

    def paired_case_differences(stem: str) -> list[float]:
        differences = []
        for case_id in case_ids:
            target = by_case_method.get((case_id, target_method), [])
            baseline = by_case_method.get((case_id, baseline_method), [])
            if not target or not baseline:
                continue

            def score(item: dict[str, Any]) -> float:
                value = item[stem]
                if stem == "locked_intent_preserved":
                    return 1.0 if value == "yes" else 0.0
                return float(value)

            target_mean = sum(score(item) for item in target) / len(target)
            baseline_mean = sum(score(item) for item in baseline) / len(baseline)
            differences.append(target_mean - baseline_mean)
        return differences

    case_differences = paired_case_differences("domain_correctness_1_5")
    method_summaries = {}
    for method in (target_method, baseline_method):
        method_rows = [item for item in long_rows if item["method"] == method]
        intent_counts = Counter(
            str(item["locked_intent_preserved"]) for item in method_rows
        )
        method_summaries[method] = {
            "rating_count": len(method_rows),
            "mean_domain_correctness_1_5": sum(
                float(item["domain_correctness_1_5"]) for item in method_rows
            )
            / len(method_rows),
            "mean_evidence_sufficiency_1_5": sum(
                float(item["evidence_sufficiency_1_5"]) for item in method_rows
            )
            / len(method_rows),
            "locked_intent_counts": dict(intent_counts),
            "locked_intent_yes_rate": intent_counts.get("yes", 0)
            / len(method_rows),
            "unsupported_or_unsafe_claim_total": sum(
                float(item["unsupported_or_unsafe_claim_count"])
                for item in method_rows
            ),
            "unsupported_or_unsafe_claim_mean": sum(
                float(item["unsupported_or_unsafe_claim_count"])
                for item in method_rows
            )
            / len(method_rows),
        }

    paired_secondary = {}
    for output_name, stem in (
        ("evidence_sufficiency_1_5", "evidence_sufficiency_1_5"),
        ("locked_intent_yes_rate", "locked_intent_preserved"),
        (
            "unsupported_or_unsafe_claim_count",
            "unsupported_or_unsafe_claim_count",
        ),
    ):
        differences = paired_case_differences(stem)
        paired_secondary[output_name] = {
            "mean_difference": (
                sum(differences) / len(differences) if differences else None
            ),
            "case_bootstrap_95_ci": bootstrap_mean_ci(differences),
            "direction": (
                "higher_is_better"
                if stem != "unsupported_or_unsafe_claim_count"
                else "lower_is_better"
            ),
        }
    reviewers = sorted({row["reviewer_code"] for row in rows if row["gradable"]})
    agreement = {}
    for method in (target_method, baseline_method):
        matrix = []
        for case_id in case_ids:
            items = by_case_method.get((case_id, method), [])
            by_reviewer = {
                item["reviewer_code"]: item["domain_correctness_1_5"]
                for item in items
            }
            if set(by_reviewer) == set(reviewers):
                matrix.append([by_reviewer[reviewer] for reviewer in reviewers])
        agreement[method] = {"icc_2_k": icc2k(matrix), "complete_cases": len(matrix)}

    return {
        "case_count_with_primary_pair": len(case_differences),
        "reviewer_count": len(reviewers),
        "primary_domain_correctness_target_minus_baseline": {
            "target_method": target_method,
            "baseline_method": baseline_method,
            "mean_difference": (
                sum(case_differences) / len(case_differences)
                if case_differences
                else None
            ),
            "case_bootstrap_95_ci": bootstrap_mean_ci(case_differences),
        },
        "preference_counts": dict(preferences),
        "gradable_count": sum(bool(row["gradable"]) for row in rows),
        "ungradable_count": sum(not row["gradable"] for row in rows),
        "confidence_counts": dict(
            Counter(
                str(row["confidence"])
                for row in rows
                if row["gradable"]
            )
        ),
        "method_summaries": method_summaries,
        "paired_secondary_target_minus_baseline": paired_secondary,
        "domain_correctness_interrater_agreement": agreement,
        "long_rows": long_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and analyze E70 reviews")
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--ratings", nargs="+", type=Path, required=True)
    parser.add_argument("--target-method", default="ensr")
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-long-csv", type=Path, required=True)
    args = parser.parse_args()
    key = load_key(args.private_key)
    rows, errors = load_ratings(args.ratings, key)
    reviewer_count = len({row["reviewer_code"] for row in rows})
    complete = not errors and reviewer_count >= 3
    analysis = (
        analyze(
            rows,
            key,
            target_method=args.target_method,
            baseline_method=args.baseline_method,
        )
        if complete
        else None
    )
    result = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "blocked",
        "reviewer_count": reviewer_count,
        "validation_errors": errors,
        "analysis": (
            {key: value for key, value in analysis.items() if key != "long_rows"}
            if analysis
            else None
        ),
        "reporting_boundary": (
            "Formal results require at least three complete independent reviewers "
            "and zero validation errors."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_long_csv.parent.mkdir(parents=True, exist_ok=True)
    if analysis and analysis["long_rows"]:
        with args.output_long_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(analysis["long_rows"][0]))
            writer.writeheader()
            writer.writerows(analysis["long_rows"])
    print(json.dumps({"status": result["status"], "errors": errors}, ensure_ascii=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
