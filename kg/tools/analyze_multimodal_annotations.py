from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "annotations" / "eval" / "multimodal_gold_pilot_v0.csv"
DEFAULT_OUTPUT = ROOT / "kg" / "data" / "multimodal_gold_pilot_status.json"


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    count = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / count
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = counts_a.keys() | counts_b.keys()
    expected = sum(
        (counts_a[category] / count) * (counts_b[category] / count)
        for category in categories
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def analyze(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_modality: dict[str, Any] = {}
    all_final = True
    for modality in sorted({row["modality"] for row in rows}):
        subset = [row for row in rows if row["modality"] == modality]
        valid_labels = set(subset[0]["label_set"].split("|"))
        invalid = []
        dual_labeled = []
        final_labels = []
        disagreements = 0
        for row in subset:
            label_a = row["annotator_a_label"].strip()
            label_b = row["annotator_b_label"].strip()
            adjudicated = row["adjudicated_label"].strip()
            for column, label in (
                ("annotator_a_label", label_a),
                ("annotator_b_label", label_b),
                ("adjudicated_label", adjudicated),
            ):
                if label and label not in valid_labels:
                    invalid.append(
                        {
                            "row_id": row["row_id"],
                            "column": column,
                            "label": label,
                        }
                    )
            if label_a and label_b:
                dual_labeled.append((label_a, label_b))
                if label_a != label_b:
                    disagreements += 1
            final = adjudicated or (label_a if label_a and label_a == label_b else "")
            if final:
                final_labels.append(final)
            else:
                all_final = False

        labels_a = [pair[0] for pair in dual_labeled]
        labels_b = [pair[1] for pair in dual_labeled]
        raw_agreement = (
            sum(a == b for a, b in dual_labeled) / len(dual_labeled)
            if dual_labeled
            else None
        )
        kappa = cohens_kappa(labels_a, labels_b)
        by_modality[modality] = {
            "row_count": len(subset),
            "dual_labeled_count": len(dual_labeled),
            "disagreement_count": disagreements,
            "final_label_count": len(final_labels),
            "raw_agreement": (
                round(raw_agreement, 6) if raw_agreement is not None else None
            ),
            "cohens_kappa": round(kappa, 6) if kappa is not None else None,
            "final_label_distribution": dict(sorted(Counter(final_labels).items())),
            "invalid_labels": invalid,
        }

    invalid_count = sum(
        len(statistics["invalid_labels"])
        for statistics in by_modality.values()
    )
    return {
        "schema_version": "0.1",
        "dataset": "annotations/eval/multimodal_gold_pilot_v0.csv",
        "status": "ready" if all_final and invalid_count == 0 else "incomplete",
        "row_count": len(rows),
        "all_rows_have_final_label": all_final,
        "invalid_label_count": invalid_count,
        "by_modality": by_modality,
        "reporting_boundary": (
            "Metrics are reportable only when status is ready; an incomplete file "
            "is an annotation progress report."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze multimodal double annotations")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = analyze(rows)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: status={result['status']}, rows={result['row_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
