"""Render Figure 10 from frozen LanternQuest E21 and E70 results.

Panels separate the six-method held-out task evaluation from the independent
expert comparison. No values or uncertainty intervals are simulated.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
E21_ROOT = REPO_ROOT / "artifacts" / "eval" / "lanternquest_e21_formal_local_qwen3_8b_v1"
E70_ROOT = REPO_ROOT / "artifacts" / "human_study" / "e70"
E70_RETURN = E70_ROOT / "returns" / "return_20260924_v1"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "paper" / "figures" / "figure10"
OUTPUT_STEM = OUTPUT_DIR / "Figure_10_domain_expert_validation_nature_v1"

E21_SUMMARY = E21_ROOT / "formal_summary.json"
E21_SCORES = E21_ROOT / "scores.jsonl"
E21_RESULTS = REPO_ROOT / "artifacts" / "eval" / "lanternquest_e21_results.csv"
E70_ANALYSIS = E70_RETURN / "e70_analysis_v1.json"
E70_LONG = E70_RETURN / "e70_unblinded_long_v1.csv"
E70_PANEL = E70_RETURN / "expert_panel_verification_v1.json"
E70_KEY = E70_ROOT / "private_key.csv"
E70_RAW_ROOT = E70_RETURN / "raw"


COLORS = {
    "text": "#26364A",
    "subtext": "#667085",
    "grid": "#E8EBEF",
    "ensr": "#C95E72",
    "ensr_light": "#F2D6DC",
    "iper": "#6576A5",
    "iper_light": "#B8C3DF",
    "neutral": "#E8EBEF",
    "neutral_dark": "#A9B1BA",
    "white": "#FFFFFF",
    "green": "#4F806B",
    "green_light": "#DDEDE6",
    "gold": "#C7954A",
}

METHOD_ORDER = [
    "b0_full_context",
    "b1_rag",
    "b2_kg_rag",
    "b4_sequential",
    "iper_rag",
    "ensr",
]

METHOD_LABELS = {
    "b0_full_context": "B0 Full context",
    "b1_rag": "B1 RAG",
    "b2_kg_rag": "B2 KG-RAG",
    "b4_sequential": "B4 Sequential",
    "iper_rag": "IPER-RAG",
    "ensr": "ENSR",
}

FAMILY_LABELS = {
    "compositional_choices": "Compositional\nchoices",
    "intent_and_rework": "Intent and\nrework",
    "evidence_boundary": "Evidence\nboundary",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["text"],
    )


def clean_axis(ax: plt.Axes, *, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if left:
        ax.spines["left"].set_color(COLORS["text"])
    if bottom:
        ax.spines["bottom"].set_color(COLORS["text"])
    ax.tick_params(axis="both", colors=COLORS["text"], width=0.7, length=3)


def case_number(case_id: str) -> str:
    return case_id.split("_")[1]


def case_family(case_id: str) -> str:
    for family in FAMILY_LABELS:
        if family in case_id:
            return family
    raise ValueError(f"Unknown case family: {case_id}")


def extract_e21(summary: dict, score_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    if summary.get("status") != "complete" or summary.get("case_count") != 12 or summary.get("score_count") != 72:
        raise RuntimeError("E21 frozen result matrix is incomplete.")
    methods = {item["method"]: item for item in summary["aggregates"]}
    if set(methods) != set(METHOD_ORDER):
        raise RuntimeError("E21 method set does not match the frozen six-method design.")
    aggregate_rows = []
    for method in METHOD_ORDER:
        source = methods[method]
        aggregate_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "episodes": source["episodes"],
                "grounded_task_success_rate": source["grounded_task_success_rate"],
                "decision_type_accuracy": source["decision_type_accuracy"],
                "mean_evidence_coverage": source["mean_evidence_coverage"],
                "locked_intent_preservation_rate": source["locked_intent_preservation_rate"],
                "unsupported_action_count": source["unsupported_action_count"],
                "invalid_action_count": source["invalid_action_count"],
                "avoidable_rework_count": source["avoidable_rework_count"],
                "external_failure_count": source["external_failure_count"],
            }
        )
    if len(score_rows) != 72:
        raise RuntimeError(f"Expected 72 E21 case-method rows, found {len(score_rows)}.")
    if any(row.get("run_status") != "completed" for row in score_rows):
        raise RuntimeError("E21 contains a non-completed case-method run.")
    compact_rows = []
    for row in score_rows:
        compact_rows.append(
            {
                "case_id": row["case_id"],
                "case_number": case_number(row["case_id"]),
                "case_family": case_family(row["case_id"]),
                "method": row["method"],
                "grounded_task_success": bool(row["grounded_task_success"]),
                "decision_type_exact": bool(row["decision_type_exact"]),
                "evidence_coverage": row["evidence_coverage"],
                "locked_intent_preserved": bool(row["locked_intent_preserved"]),
                "unsupported_action_count": row["unsupported_action_count"],
                "invalid_action_count": row["invalid_action_count"],
                "avoidable_rework_count": row["avoidable_rework_count"],
            }
        )
    return aggregate_rows, compact_rows


def extract_preferences() -> list[dict]:
    key_rows = read_csv(E70_KEY)
    key = {row["review_id"]: row for row in key_rows}
    preferences: list[dict] = []
    for path in sorted(E70_RAW_ROOT.glob("R*/expert_review_R*.csv")):
        for row in read_csv(path):
            mapping = key[row["review_id"]]
            raw_preference = row["preference"].strip()
            if raw_preference == "tie":
                preference = "tie"
            elif raw_preference in {"X", "Y"}:
                preference = mapping[f"system_{raw_preference.lower()}_method"]
            else:
                raise RuntimeError(f"Unexpected E70 preference: {raw_preference}")
            preferences.append(
                {
                    "reviewer_code": row["reviewer_code"],
                    "case_id": row["case_id"],
                    "case_number": case_number(row["case_id"]),
                    "case_family": case_family(row["case_id"]),
                    "preference": preference,
                    "confidence": int(row["confidence_1_5"]),
                }
            )
    if len(preferences) != 36:
        raise RuntimeError(f"Expected 36 E70 preferences, found {len(preferences)}.")
    return preferences


def extract_e70(analysis: dict, long_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if analysis.get("status") != "complete" or analysis.get("reviewer_count") != 3 or analysis.get("validation_errors"):
        raise RuntimeError("E70 expert audit is incomplete or invalid.")
    if len(long_rows) != 72:
        raise RuntimeError(f"Expected 72 unblinded ratings, found {len(long_rows)}.")
    a = analysis["analysis"]
    primary = a["primary_domain_correctness_target_minus_baseline"]
    secondary = a["paired_secondary_target_minus_baseline"]
    effects = [
        {
            "metric": "Domain correctness",
            "unit": "score points",
            "estimate": primary["mean_difference"],
            "ci95_low": primary["case_bootstrap_95_ci"][0],
            "ci95_high": primary["case_bootstrap_95_ci"][1],
            "direction": "higher_is_better",
        },
        {
            "metric": "Evidence sufficiency",
            "unit": "score points",
            "estimate": secondary["evidence_sufficiency_1_5"]["mean_difference"],
            "ci95_low": secondary["evidence_sufficiency_1_5"]["case_bootstrap_95_ci"][0],
            "ci95_high": secondary["evidence_sufficiency_1_5"]["case_bootstrap_95_ci"][1],
            "direction": "higher_is_better",
        },
        {
            "metric": "Intent preserved",
            "unit": "percentage points",
            "estimate": 100.0 * secondary["locked_intent_yes_rate"]["mean_difference"],
            "ci95_low": 100.0 * secondary["locked_intent_yes_rate"]["case_bootstrap_95_ci"][0],
            "ci95_high": 100.0 * secondary["locked_intent_yes_rate"]["case_bootstrap_95_ci"][1],
            "direction": "higher_is_better",
        },
        {
            "metric": "Unsupported/unsafe claims",
            "unit": "claims per rating",
            "estimate": secondary["unsupported_or_unsafe_claim_count"]["mean_difference"],
            "ci95_low": secondary["unsupported_or_unsafe_claim_count"]["case_bootstrap_95_ci"][0],
            "ci95_high": secondary["unsupported_or_unsafe_claim_count"]["case_bootstrap_95_ci"][1],
            "direction": "lower_is_better",
        },
    ]

    normalized_long: list[dict] = []
    for row in long_rows:
        normalized_long.append(
            {
                "reviewer_code": row["reviewer_code"],
                "case_id": row["case_id"],
                "case_number": case_number(row["case_id"]),
                "case_family": case_family(row["case_id"]),
                "method": row["method"],
                "domain_correctness_1_5": float(row["domain_correctness_1_5"]),
                "evidence_sufficiency_1_5": float(row["evidence_sufficiency_1_5"]),
                "locked_intent_preserved": row["locked_intent_preserved"],
                "unsupported_or_unsafe_claim_count": float(row["unsupported_or_unsafe_claim_count"]),
            }
        )

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in normalized_long:
        grouped.setdefault((row["case_id"], row["method"]), []).append(row)
    case_means: list[dict] = []
    for (case_id, method), records in sorted(grouped.items()):
        case_means.append(
            {
                "case_id": case_id,
                "case_number": case_number(case_id),
                "case_family": case_family(case_id),
                "method": method,
                "reviewer_count": len(records),
                "mean_domain_correctness": np.mean([row["domain_correctness_1_5"] for row in records]),
                "mean_evidence_sufficiency": np.mean([row["evidence_sufficiency_1_5"] for row in records]),
                "intent_yes_rate": np.mean([row["locked_intent_preserved"] == "yes" for row in records]),
                "mean_unsupported_or_unsafe_claims": np.mean([row["unsupported_or_unsafe_claim_count"] for row in records]),
            }
        )
    return effects, normalized_long, case_means


def draw_panel_a(ax: plt.Axes, rows: list[dict]) -> None:
    ax.set_title("E21 held-out method profile (12 cases)", loc="left", pad=6)
    panel_label(ax, "a", x=-0.24)
    metrics = [
        ("grounded_task_success_rate", "Grounded\nsuccess"),
        ("decision_type_accuracy", "Decision\naccuracy"),
        ("mean_evidence_coverage", "Evidence\ncoverage"),
        ("locked_intent_preservation_rate", "Intent\npreserved"),
    ]
    matrix = 100.0 * np.array([[row[key] for key, _ in metrics] for row in rows])
    cmap = mcolors.LinearSegmentedColormap.from_list("neutral_to_rose", ["#F4F5F7", COLORS["ensr_light"], COLORS["ensr"]])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.1, color="white" if value >= 80 else COLORS["text"], fontweight="semibold" if value == 100 else "normal")
    ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
    ax.set_yticks(np.arange(len(rows)), [row["method_label"] for row in rows])
    ax.tick_params(axis="both", length=0)
    for label, row in zip(ax.get_yticklabels(), rows):
        if row["method"] == "ensr":
            label.set_color(COLORS["ensr"]); label.set_fontweight("bold")
        elif row["method"] == "iper_rag":
            label.set_color(COLORS["iper"]); label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(3.67, 4.5, "tie", color=COLORS["subtext"], fontsize=5.4, va="center", ha="left", clip_on=False)


def draw_panel_b(ax: plt.Axes, aggregate_rows: list[dict], case_rows: list[dict]) -> None:
    ax.set_title("Case-level success and action burden", loc="left", pad=6)
    panel_label(ax, "b", x=-0.12)
    cases = sorted({row["case_id"] for row in case_rows}, key=lambda value: int(case_number(value)))
    lookup = {(row["method"], row["case_id"]): row for row in case_rows}
    success = np.array([[int(lookup[(method, case)]["grounded_task_success"]) for case in cases] for method in METHOD_ORDER])
    cmap = mcolors.ListedColormap(["#E4E8F1", COLORS["ensr"]])
    ax.imshow(success, cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(-0.5, 11.5, 5.5, -0.5), interpolation="nearest")
    for i, method in enumerate(METHOD_ORDER):
        for j, case in enumerate(cases):
            if success[i, j]:
                ax.scatter(j, i, s=7, facecolor="white", edgecolor="none", zorder=4)
    ax.set_xlim(-0.5, 15.7)
    ax.set_ylim(5.5, -0.8)
    ax.set_xticks(np.arange(12), [case_number(case) for case in cases])
    ax.set_yticks(np.arange(6), [METHOD_LABELS[method] for method in METHOD_ORDER])
    ax.tick_params(axis="both", length=0)
    for boundary in [3.5, 7.5]:
        ax.axvline(boundary, color="white", linewidth=2.2, zorder=3)
    for start, end, family in [(0, 3, "compositional_choices"), (4, 7, "intent_and_rework"), (8, 11, "evidence_boundary")]:
        ax.text((start + end) / 2, -0.72, FAMILY_LABELS[family].replace("\n", " "), fontsize=5.1, color=COLORS["subtext"], ha="center", va="bottom")
    burdens = {row["method"]: row for row in aggregate_rows}
    burden_keys = [("unsupported_action_count", "U"), ("invalid_action_count", "I"), ("avoidable_rework_count", "R")]
    for col, (key, short) in enumerate(burden_keys, start=13):
        ax.text(col, -0.64, short, fontsize=5.7, fontweight="bold", color=COLORS["subtext"], ha="center")
        for i, method in enumerate(METHOD_ORDER):
            value = int(burdens[method][key])
            ax.scatter(col, i, s=19 + 8 * value, facecolor=COLORS["iper"] if value else COLORS["white"], edgecolor=COLORS["iper"], linewidth=0.8, zorder=5)
            ax.text(col, i, str(value), fontsize=4.8, ha="center", va="center", color="white" if value else COLORS["subtext"], zorder=6)
    ax.text(14, 5.55, "U unsupported · I invalid · R rework", fontsize=4.9, color=COLORS["subtext"], ha="center", va="top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_effect_axis(ax: plt.Axes, row: dict, xlim: tuple[float, float], xlabel: str) -> None:
    estimate = row["estimate"]
    low, high = row["ci95_low"], row["ci95_high"]
    ax.axvline(0, color=COLORS["subtext"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax.errorbar(
        estimate,
        0,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt="o",
        markersize=5.6,
        markerfacecolor=COLORS["ensr"],
        markeredgecolor="white",
        markeredgewidth=0.6,
        ecolor=COLORS["ensr"],
        elinewidth=1.25,
        capsize=2.5,
        zorder=3,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=5.5)
    ax.set_title(row["metric"], loc="left", fontsize=6.5, pad=3)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    clean_axis(ax, left=False, bottom=True)
    fmt = ".1f" if row["unit"] == "percentage points" else ".2f"
    ax.text(0.98, 0.86, f"Δ {estimate:{fmt}}", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=COLORS["ensr"], fontweight="semibold")


def draw_panel_c(fig: plt.Figure, spec, effects: list[dict]) -> None:
    sub = spec.subgridspec(2, 2, wspace=0.38, hspace=0.62)
    axes = [fig.add_subplot(sub[i, j]) for i in range(2) for j in range(2)]
    panel_label(axes[0], "c", x=-0.38, y=1.35)
    axes[0].text(-0.02, 1.35, "E70 paired expert effects", transform=axes[0].transAxes, fontsize=8.5, fontweight="semibold", ha="left", va="bottom")
    draw_effect_axis(axes[0], effects[0], (-0.2, 1.1), "Score difference")
    draw_effect_axis(axes[1], effects[1], (-0.2, 1.1), "Score difference")
    draw_effect_axis(axes[2], effects[2], (-5, 30), "Difference (pp)")
    draw_effect_axis(axes[3], effects[3], (-0.30, 0.10), "Claims per rating")
    axes[3].text(0.02, 0.86, "lower is better", transform=axes[3].transAxes, fontsize=5.0, color=COLORS["subtext"], va="top")


def draw_panel_d(ax: plt.Axes, preferences: list[dict]) -> None:
    ax.set_title("Blinded preferences (36 judgments)", loc="left", pad=6)
    panel_label(ax, "d")
    reviewers = ["R1", "R2", "R3"]
    cases = sorted({row["case_id"] for row in preferences}, key=lambda value: int(case_number(value)))
    lookup = {(row["reviewer_code"], row["case_id"]): row["preference"] for row in preferences}
    code = {"b4_sequential": -1, "tie": 0, "ensr": 1}
    matrix = np.array([[code[lookup[(reviewer, case)]] for case in cases] for reviewer in reviewers])
    cmap = mcolors.ListedColormap([COLORS["iper"], COLORS["neutral"], COLORS["ensr"]])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", extent=(-0.5, 11.5, 2.5, -0.5), interpolation="nearest")
    for i, reviewer in enumerate(reviewers):
        for j, case in enumerate(cases):
            value = lookup[(reviewer, case)]
            ax.text(j, i, "E" if value == "ensr" else ("B" if value == "b4_sequential" else "="), ha="center", va="center", fontsize=6.2, color="white" if value != "tie" else COLORS["subtext"], fontweight="bold")
        counts = Counter(row["preference"] for row in preferences if row["reviewer_code"] == reviewer)
        ax.text(12.1, i, f"E {counts.get('ensr', 0)}  = {counts.get('tie', 0)}", fontsize=5.4, va="center", color=COLORS["text"])
    ax.set_xlim(-0.5, 13.8)
    ax.set_ylim(2.5, -0.85)
    ax.set_xticks(np.arange(12), [case_number(case) for case in cases])
    ax.set_yticks(np.arange(3), reviewers)
    ax.tick_params(axis="both", length=0)
    for boundary in [3.5, 7.5]:
        ax.axvline(boundary, color="white", linewidth=2.2)
    for start, end, family in [(0, 3, "compositional_choices"), (4, 7, "intent_and_rework"), (8, 11, "evidence_boundary")]:
        ax.text((start + end) / 2, -0.72, FAMILY_LABELS[family].replace("\n", " "), fontsize=5.0, color=COLORS["subtext"], ha="center", va="bottom")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="none", markersize=6, markerfacecolor=COLORS["ensr"], markeredgecolor="none", label="ENSR"),
            Line2D([], [], marker="s", linestyle="none", markersize=6, markerfacecolor=COLORS["neutral"], markeredgecolor="none", label="Tie"),
            Line2D([], [], marker="s", linestyle="none", markersize=6, markerfacecolor=COLORS["iper"], markeredgecolor="none", label="B4"),
        ],
        loc="lower right",
        ncol=3,
        fontsize=5.3,
        handletextpad=0.25,
        columnspacing=0.8,
    )


def draw_panel_e(fig: plt.Figure, spec, analysis: dict) -> list[dict]:
    sub = spec.subgridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.45)
    ax_icc = fig.add_subplot(sub[0, 0])
    ax_conf = fig.add_subplot(sub[0, 1])
    panel_label(ax_icc, "e", x=-0.35, y=1.10)
    ax_icc.text(-0.02, 1.10, "Agreement and review confidence", transform=ax_icc.transAxes, fontsize=8.5, fontweight="semibold", ha="left", va="bottom")
    a = analysis["analysis"]
    agreement = a["domain_correctness_interrater_agreement"]
    icc_rows = [
        {"method": "ensr", "label": "ENSR", "icc_2_k": agreement["ensr"]["icc_2_k"], "complete_cases": agreement["ensr"]["complete_cases"]},
        {"method": "b4_sequential", "label": "B4 Sequential", "icc_2_k": agreement["b4_sequential"]["icc_2_k"], "complete_cases": agreement["b4_sequential"]["complete_cases"]},
    ]
    ax_icc.axvspan(0.75, 1.0, color=COLORS["green_light"], alpha=0.65, zorder=0)
    ax_icc.axvline(0.75, color=COLORS["green"], linewidth=0.8, linestyle=(0, (3, 2)))
    for i, row in enumerate(icc_rows):
        color = COLORS["ensr"] if row["method"] == "ensr" else COLORS["iper"]
        face = color if row["method"] == "ensr" else "white"
        ax_icc.plot([0.7, row["icc_2_k"]], [i, i], color=COLORS["neutral_dark"], linewidth=1.0)
        ax_icc.scatter(row["icc_2_k"], i, s=42, facecolor=face, edgecolor=color, linewidth=1.0, zorder=3)
        ax_icc.text(row["icc_2_k"] - 0.008, i - 0.18, f"{row['icc_2_k']:.3f}", ha="right", va="top", fontsize=5.5, color=color, fontweight="semibold")
    ax_icc.set_xlim(0.7, 1.01)
    ax_icc.set_ylim(1.5, -0.5)
    ax_icc.set_yticks([0, 1], [row["label"] for row in icc_rows])
    ax_icc.set_xlabel("ICC(2,k)")
    ax_icc.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    clean_axis(ax_icc)
    ax_icc.text(0.752, 1.42, "0.75 threshold", fontsize=4.9, color=COLORS["green"], ha="left", va="bottom")

    confidence = {int(key): value for key, value in a["confidence_counts"].items()}
    count4, count5 = confidence.get(4, 0), confidence.get(5, 0)
    ax_conf.barh(0, count4, height=0.35, color=COLORS["iper_light"], edgecolor="white", linewidth=0.5)
    ax_conf.barh(0, count5, left=count4, height=0.35, color=COLORS["ensr"], edgecolor="white", linewidth=0.5)
    ax_conf.text(count4 / 2, 0, str(count4), ha="center", va="center", fontsize=5.6, color=COLORS["text"])
    ax_conf.text(count4 + count5 / 2, 0, str(count5), ha="center", va="center", fontsize=5.6, color="white")
    ax_conf.set_xlim(0, 36)
    ax_conf.set_ylim(-0.7, 0.7)
    ax_conf.set_yticks([])
    ax_conf.set_xticks([0, 12, 24, 36])
    ax_conf.set_xlabel("Judgments")
    ax_conf.set_title("Confidence", loc="left", fontsize=6.5, pad=3)
    clean_axis(ax_conf, left=False, bottom=True)
    ax_conf.legend(
        handles=[
            Line2D([], [], color=COLORS["iper_light"], linewidth=5, label="4/5"),
            Line2D([], [], color=COLORS["ensr"], linewidth=5, label="5/5"),
        ],
        loc="upper center",
        ncol=2,
        fontsize=5.2,
        handlelength=1.3,
        columnspacing=0.7,
    )
    ax_conf.text(0.5, 0.12, "36/36 gradable", transform=ax_conf.transAxes, ha="center", fontsize=5.3, color=COLORS["subtext"])
    return icc_rows + [
        {"method": "all", "label": "confidence_4", "icc_2_k": "", "complete_cases": count4},
        {"method": "all", "label": "confidence_5", "icc_2_k": "", "complete_cases": count5},
    ]


def draw_panel_f(ax: plt.Axes, case_means: list[dict]) -> list[dict]:
    ax.set_title("Where expert-score gains occurred", loc="left", pad=6)
    panel_label(ax, "f")
    lookup = {(row["case_id"], row["method"]): row for row in case_means}
    cases = sorted({row["case_id"] for row in case_means}, key=lambda value: int(case_number(value)))
    delta_rows = []
    for case in cases:
        ensr = lookup[(case, "ensr")]
        b4 = lookup[(case, "b4_sequential")]
        delta_rows.append(
            {
                "case_id": case,
                "case_number": case_number(case),
                "case_family": case_family(case),
                "correctness_delta_ensr_minus_b4": ensr["mean_domain_correctness"] - b4["mean_domain_correctness"],
                "evidence_delta_ensr_minus_b4": ensr["mean_evidence_sufficiency"] - b4["mean_evidence_sufficiency"],
                "intent_delta_ensr_minus_b4": ensr["intent_yes_rate"] - b4["intent_yes_rate"],
                "unsafe_claim_delta_ensr_minus_b4": ensr["mean_unsupported_or_unsafe_claims"] - b4["mean_unsupported_or_unsafe_claims"],
            }
        )
    families = list(FAMILY_LABELS)
    jitter = np.array([-0.18, -0.06, 0.06, 0.18])
    for x, family in enumerate(families):
        rows = [row for row in delta_rows if row["case_family"] == family]
        rows.sort(key=lambda row: int(row["case_number"]))
        values = np.array([row["correctness_delta_ensr_minus_b4"] for row in rows])
        ax.scatter(x + jitter, values, s=30, facecolor="white", edgecolor=COLORS["ensr"], linewidth=0.9, zorder=3)
        mean = float(np.mean(values))
        ax.scatter(x, mean, s=48, marker="D", facecolor=COLORS["ensr"], edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(x, mean + 0.13, f"mean {mean:.2f}", ha="center", va="bottom", fontsize=5.4, color=COLORS["ensr"], fontweight="semibold")
        for xi, row, value in zip(x + jitter, rows, values):
            if value > 0:
                ax.text(xi, value + 0.08, row["case_number"], ha="center", va="bottom", fontsize=4.8, color=COLORS["subtext"])
    ax.axhline(0, color=COLORS["subtext"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.12, 2.05)
    ax.set_xticks(np.arange(3), [FAMILY_LABELS[family] for family in families])
    ax.set_ylabel("Correctness difference\nENSR − B4 (1–5)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    clean_axis(ax)
    ax.text(0.02, 0.96, "four cases per family", transform=ax.transAxes, fontsize=5.2, color=COLORS["subtext"], va="top")
    ax.text(0.98, 0.47, "open points: cases\nfilled diamonds: family means", transform=ax.transAxes, fontsize=5.0, color=COLORS["subtext"], ha="right", va="center")
    return delta_rows


def render() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    e21_summary = read_json(E21_SUMMARY)
    e21_score_rows = read_jsonl(E21_SCORES)
    e70_analysis = read_json(E70_ANALYSIS)
    e70_long_rows = read_csv(E70_LONG)
    panel = read_json(E70_PANEL)
    if not str(panel.get("status", "")).startswith("panel_verified"):
        raise RuntimeError("E70 panel verification is not complete.")

    e21_aggregates, e21_cases = extract_e21(e21_summary, e21_score_rows)
    effects, e70_long, case_means = extract_e70(e70_analysis, e70_long_rows)
    preferences = extract_preferences()

    fig = plt.figure(figsize=(12.2, 9.5))
    outer = fig.add_gridspec(
        3,
        2,
        width_ratios=[0.95, 1.35],
        height_ratios=[1.0, 1.0, 0.83],
        left=0.09,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.31,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_d = fig.add_subplot(outer[1, 1])
    ax_f = fig.add_subplot(outer[2, 1])

    draw_panel_a(ax_a, e21_aggregates)
    draw_panel_b(ax_b, e21_aggregates, e21_cases)
    draw_panel_c(fig, outer[1, 0], effects)
    draw_panel_d(ax_d, preferences)
    reliability_rows = draw_panel_e(fig, outer[2, 0], e70_analysis)
    family_delta_rows = draw_panel_f(ax_f, case_means)

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, facecolor="white")
    plt.close(fig)

    write_csv(OUTPUT_DIR / "Figure_10a_E21_method_profile_source.csv", e21_aggregates)
    write_csv(OUTPUT_DIR / "Figure_10b_E21_case_outcomes_source.csv", e21_cases)
    write_csv(OUTPUT_DIR / "Figure_10c_E70_effects_source.csv", effects)
    write_csv(OUTPUT_DIR / "Figure_10d_E70_preferences_source.csv", preferences)
    write_csv(OUTPUT_DIR / "Figure_10e_E70_reliability_source.csv", reliability_rows)
    write_csv(OUTPUT_DIR / "Figure_10f_E70_family_effects_source.csv", family_delta_rows)
    write_csv(OUTPUT_DIR / "Figure_10_E70_case_means_source.csv", case_means)

    raw_paths = sorted(E70_RAW_ROOT.glob("R*/expert_review_R*.csv"))
    source_paths = [
        E21_SUMMARY,
        E21_SCORES,
        E21_RESULTS,
        E70_ANALYSIS,
        E70_LONG,
        E70_PANEL,
        E70_KEY,
        *raw_paths,
    ]
    manifest = {
        "figure": "Figure 10",
        "output_stem": str(OUTPUT_STEM.relative_to(REPO_ROOT)),
        "sources": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_paths},
        "audit_counts": {
            "e21_cases": 12,
            "e21_methods": 6,
            "e21_case_method_rows": len(e21_cases),
            "e70_cases": 12,
            "e70_reviewers": 3,
            "e70_long_ratings": len(e70_long),
            "e70_preferences": len(preferences),
            "e70_gradable": e70_analysis["analysis"]["gradable_count"],
        },
        "claim_boundary": "E21 found ENSR tied with IPER-RAG. E70 compares ENSR with B4 Sequential only and does not establish superiority over IPER-RAG or SOTA.",
    }
    (OUTPUT_DIR / "Figure_10_source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    render()
