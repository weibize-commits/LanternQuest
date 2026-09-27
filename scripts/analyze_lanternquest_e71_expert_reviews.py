from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts" / "human_study" / "e71"
KEY_PATH = STUDY / "E71_private_blinding_key.csv"
PROTOCOL_PATH = STUDY / "E71_protocol_frozen.json"
EXPECTED_REVIEWERS = {"R1", "R2", "R3"}
EXPECTED_ROWS_PER_REVIEWER = 10
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 71_071

METRICS = {
    "recovery_appropriateness": "recovery_appropriateness_1_5",
    "evidence_traceability": "evidence_traceability_1_5",
    "intent_preservation": "intent_preservation_1_5",
    "rework_risk": "rework_risk_1_5",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def normalize_bool(value: str) -> bool:
    token = value.strip().lower()
    if token in {"true", "yes", "1", "y"}:
        return True
    if token in {"false", "no", "0", "n"}:
        return False
    raise ValueError(f"gradable must be true or false, received {value!r}")


def rating(value: str, field: str, row_id: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{row_id}: {field} must be an integer from 1 to 5") from exc
    if result not in range(1, 6):
        raise ValueError(f"{row_id}: {field} must be in 1..5")
    return result


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    rows: list[dict[str, Any]], field: str
) -> dict[str, float | int | list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["gradable"]:
            grouped[row["case_id"]].append(float(row[field]))
    if not grouped:
        return {"case_count": 0, "mean": math.nan, "ci95": [math.nan, math.nan]}
    case_means = {case: statistics.fmean(values) for case, values in grouped.items()}
    cases = sorted(case_means)
    estimate = statistics.fmean(case_means.values())
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [case_means[rng.choice(cases)] for _ in cases]
        draws.append(statistics.fmean(sample))
    return {
        "case_count": len(cases),
        "mean": estimate,
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def icc_2(matrix: list[list[float]]) -> dict[str, float | int]:
    """Two-way random, absolute-agreement ICC for single and average ratings."""
    n = len(matrix)
    k = len(matrix[0]) if matrix else 0
    if n < 2 or k < 2 or any(len(row) != k for row in matrix):
        return {"targets": n, "raters": k, "icc_2_1": math.nan, "icc_2_k": math.nan}
    grand = statistics.fmean(value for row in matrix for value in row)
    row_means = [statistics.fmean(row) for row in matrix]
    col_means = [statistics.fmean(row[j] for row in matrix) for j in range(k)]
    ss_rows = k * sum((value - grand) ** 2 for value in row_means)
    ss_cols = n * sum((value - grand) ** 2 for value in col_means)
    ss_error = sum(
        (matrix[i][j] - row_means[i] - col_means[j] + grand) ** 2
        for i in range(n)
        for j in range(k)
    )
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator_single = (
        ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    )
    denominator_average = ms_rows + (ms_cols - ms_error) / n
    return {
        "targets": n,
        "raters": k,
        "icc_2_1": (ms_rows - ms_error) / denominator_single
        if denominator_single
        else math.nan,
        "icc_2_k": (ms_rows - ms_error) / denominator_average
        if denominator_average
        else math.nan,
    }


def load_key() -> dict[str, dict[str, str]]:
    _, rows = read_csv(KEY_PATH)
    key = {row["review_id"]: row for row in rows}
    if len(key) != 30:
        raise RuntimeError(f"Private key contains {len(key)} unique rows, expected 30")
    return key


def validate_and_unblind(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    key = load_key()
    long_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_review_ids: set[str] = set()
    seen_reviewers: set[str] = set()
    source_files: list[str] = []

    for path in paths:
        source_files.append(str(path.resolve()))
        fieldnames, rows = read_csv(path)
        required = {
            "reviewer_code",
            "review_id",
            "vignette_id",
            "case_id",
            "gradable",
            "unable_reason",
            "preference",
            "confidence_1_5",
        }
        for suffix in METRICS.values():
            required.update({f"x_{suffix}", f"y_{suffix}"})
        missing = sorted(required - set(fieldnames))
        if missing:
            errors.append(f"{path}: missing columns {missing}")
            continue
        if len(rows) != EXPECTED_ROWS_PER_REVIEWER:
            errors.append(
                f"{path}: contains {len(rows)} rows, expected {EXPECTED_ROWS_PER_REVIEWER}"
            )
        for source in rows:
            row_id = source.get("review_id", "").strip()
            reviewer = source.get("reviewer_code", "").strip()
            if row_id in seen_review_ids:
                errors.append(f"duplicate review_id across returns: {row_id}")
                continue
            seen_review_ids.add(row_id)
            seen_reviewers.add(reviewer)
            if row_id not in key:
                errors.append(f"{path}: unknown review_id {row_id}")
                continue
            assignment = key[row_id]
            for field in ("reviewer_code", "vignette_id", "case_id"):
                if source.get(field, "").strip() != assignment[field]:
                    errors.append(
                        f"{row_id}: {field} does not match the frozen assignment"
                    )
            if reviewer not in EXPECTED_REVIEWERS:
                errors.append(f"{row_id}: unexpected reviewer code {reviewer!r}")
            try:
                gradable = normalize_bool(source.get("gradable", ""))
            except ValueError as exc:
                errors.append(f"{row_id}: {exc}")
                continue
            output: dict[str, Any] = {
                "reviewer_code": reviewer,
                "review_id": row_id,
                "vignette_id": assignment["vignette_id"],
                "case_id": assignment["case_id"],
                "gradable": gradable,
                "unable_reason": source.get("unable_reason", "").strip(),
                "notes": source.get("notes", "").strip(),
                "system_x_method": assignment["system_x_method"],
                "system_y_method": assignment["system_y_method"],
            }
            if not gradable:
                if not output["unable_reason"]:
                    errors.append(f"{row_id}: unable_reason is required when gradable=false")
                output["preference"] = "unable"
                output["confidence_1_5"] = None
                for name in METRICS:
                    output[f"ensr_{name}"] = None
                    output[f"iper_rag_{name}"] = None
                    output[f"ensr_minus_iper_{name}"] = None
                long_rows.append(output)
                continue
            try:
                for name, suffix in METRICS.items():
                    x_value = rating(source[f"x_{suffix}"].strip(), f"x_{suffix}", row_id)
                    y_value = rating(source[f"y_{suffix}"].strip(), f"y_{suffix}", row_id)
                    if assignment["system_x_method"] == "ensr":
                        ensr_value, iper_value = x_value, y_value
                    else:
                        ensr_value, iper_value = y_value, x_value
                    output[f"ensr_{name}"] = ensr_value
                    output[f"iper_rag_{name}"] = iper_value
                    output[f"ensr_minus_iper_{name}"] = ensr_value - iper_value
                output["confidence_1_5"] = rating(
                    source.get("confidence_1_5", "").strip(), "confidence_1_5", row_id
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            preference = source.get("preference", "").strip().lower()
            if preference not in {"x", "y", "tie", "unable"}:
                errors.append(f"{row_id}: preference must be X, Y, tie or unable")
                continue
            if preference in {"x", "y"}:
                output["preference"] = assignment[f"system_{preference}_method"]
            else:
                output["preference"] = preference
            long_rows.append(output)

    if seen_reviewers != EXPECTED_REVIEWERS:
        errors.append(
            f"returned reviewer set is {sorted(seen_reviewers)}, expected {sorted(EXPECTED_REVIEWERS)}"
        )
    expected_ids = set(key)
    if seen_review_ids != expected_ids:
        errors.append(
            f"returned review IDs differ from frozen key: missing={len(expected_ids - seen_review_ids)}, "
            f"unexpected={len(seen_review_ids - expected_ids)}"
        )
    return long_rows, errors


def agreement_for_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
    by_vignette: dict[str, dict[str, float]] = defaultdict(dict)
    field = f"ensr_minus_iper_{metric}"
    for row in rows:
        if row["gradable"]:
            by_vignette[row["vignette_id"]][row["reviewer_code"]] = float(row[field])
    complete = []
    for vignette in sorted(by_vignette):
        if set(by_vignette[vignette]) == EXPECTED_REVIEWERS:
            complete.append([by_vignette[vignette][r] for r in sorted(EXPECTED_REVIEWERS)])
    return icc_2(complete)


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gradable = [row for row in rows if row["gradable"]]
    metrics: dict[str, Any] = {}
    for name in METRICS:
        raw_field = f"ensr_minus_iper_{name}"
        result = cluster_bootstrap(gradable, raw_field)
        result["direction"] = (
            "negative favors ENSR" if name == "rework_risk" else "positive favors ENSR"
        )
        result["gradable_rating_pairs"] = len(gradable)
        result["agreement_on_paired_difference"] = agreement_for_metric(gradable, name)
        metrics[name] = result

    preferences = Counter(row["preference"] for row in gradable)
    reviewer_means: dict[str, dict[str, float]] = {}
    for reviewer in sorted(EXPECTED_REVIEWERS):
        reviewer_rows = [row for row in gradable if row["reviewer_code"] == reviewer]
        reviewer_means[reviewer] = {
            name: statistics.fmean(
                float(row[f"ensr_minus_iper_{name}"]) for row in reviewer_rows
            )
            for name in METRICS
        }
    return {
        "study_id": "E71",
        "status": "complete_real_expert_ratings_analyzed",
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "reviewer_count": len({row["reviewer_code"] for row in rows}),
        "source_case_count": len({row["case_id"] for row in rows}),
        "vignette_count": len({row["vignette_id"] for row in rows}),
        "rating_pair_count": len(rows),
        "gradable_rating_pair_count": len(gradable),
        "ungradable_rating_pair_count": len(rows) - len(gradable),
        "primary_outcome": "evidence_traceability",
        "metrics": metrics,
        "preferences": dict(sorted(preferences.items())),
        "reviewer_mean_paired_differences": reviewer_means,
        "interpretation_boundary": (
            "The experiment estimates expert ratings of dynamic recovery traces. "
            "It does not estimate player learning and does not convert the ENSR-versus-B4 "
            "participant comparison into an ENSR-versus-IPER-RAG participant comparison."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_interval(result: dict[str, Any]) -> str:
    return f"{result['mean']:.3f} [{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]"


def write_report(path: Path, result: dict[str, Any]) -> None:
    metric = result["metrics"]
    lines = [
        "# LanternQuest E71 expert mechanism comparison",
        "",
        f"Status: {result['status']}",
        "",
        "## Design",
        "",
        (
            f"Three experts rated {result['vignette_count']} paired recovery vignettes "
            f"derived from {result['source_case_count']} frozen heritage cases. Method labels "
            "were independently masked for each reviewer and vignette."
        ),
        "",
        "## Results",
        "",
        "Paired method differences are ENSR minus IPER-RAG. Values are the mean with a "
        "95% case-cluster bootstrap interval.",
        "",
        f"- Evidence traceability, primary: {format_interval(metric['evidence_traceability'])}",
        f"- Recovery appropriateness: {format_interval(metric['recovery_appropriateness'])}",
        f"- Intent preservation: {format_interval(metric['intent_preservation'])}",
        f"- Rework risk: {format_interval(metric['rework_risk'])}. Negative values favor ENSR.",
        f"- Preferences: {json.dumps(result['preferences'], ensure_ascii=False, sort_keys=True)}",
        f"- Gradable pairs: {result['gradable_rating_pair_count']} of {result['rating_pair_count']}",
        "",
        "## Interpretation boundary",
        "",
        result["interpretation_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    rows, errors = validate_and_unblind(paths)
    if errors:
        raise RuntimeError("Return validation failed:\n- " + "\n- ".join(errors))
    result = analyze(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    long_path = output_dir / "E71_unblinded_long.csv"
    analysis_path = output_dir / "E71_analysis.json"
    report_path = output_dir / "E71_results_report.md"
    write_csv(long_path, rows)
    analysis_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_report(report_path, result)
    manifest = {
        "study_id": "E71",
        "status": result["status"],
        "raw_return_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
        ],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "private_key_sha256": sha256(KEY_PATH),
        "outputs": {
            path.name: sha256(path) for path in (long_path, analysis_path, report_path)
        },
    }
    (output_dir / "E71_result_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def self_test() -> None:
    key = load_key()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for assignment in key.values():
        row = {
            "reviewer_code": assignment["reviewer_code"],
            "review_id": assignment["review_id"],
            "vignette_id": assignment["vignette_id"],
            "case_id": assignment["case_id"],
            "gradable": "true",
            "unable_reason": "",
            "preference": "X" if assignment["system_x_method"] == "ensr" else "Y",
            "confidence_1_5": "5",
            "notes": "self-test",
        }
        for suffix in METRICS.values():
            ensr, iper = (2, 4) if suffix.startswith("rework_risk") else (5, 3)
            if assignment["system_x_method"] == "ensr":
                row[f"x_{suffix}"], row[f"y_{suffix}"] = str(ensr), str(iper)
            else:
                row[f"x_{suffix}"], row[f"y_{suffix}"] = str(iper), str(ensr)
        grouped[assignment["reviewer_code"]].append(row)
    fields = [
        "reviewer_code",
        "review_id",
        "vignette_id",
        "case_id",
        "gradable",
        "unable_reason",
    ]
    for suffix in METRICS.values():
        fields.extend([f"x_{suffix}", f"y_{suffix}"])
    fields.extend(["preference", "confidence_1_5", "notes"])
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for reviewer, rows in grouped.items():
            path = Path(directory) / f"expert_review_{reviewer}.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            paths.append(path)
        unblinded, errors = validate_and_unblind(paths)
        if errors:
            raise AssertionError(errors)
        result = analyze(unblinded)
        assert result["metrics"]["evidence_traceability"]["mean"] == 2.0
        assert result["metrics"]["rework_risk"]["mean"] == -2.0
        assert result["preferences"] == {"ensr": 30}
    print("E71 analyzer self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if len(args.ratings) != 3:
        raise SystemExit("Provide exactly three completed expert-review CSV files, one for R1, R2 and R3")
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("return_%Y%m%d_%H%M%S")
        output_dir = STUDY / "returns" / stamp
    result = run_analysis([path.resolve() for path in args.ratings], output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
