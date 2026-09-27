"""Render Figure 7 from frozen ENSR mechanism, semantic-KG, and scheduler audits.

The figure uses four complementary evidence forms:
  a) component-effect intervals with multiplicity-adjusted inference,
  b) a ranked task-impact strip for the semantic KG,
  c) evidence-debt scheduling development tests and the first-debt anchor ablation,
  d) two paired, action-level trajectories showing benefit and failure boundary.

No observations or uncertainty intervals are simulated. All values come from
the frozen E16, semantic-KG, E23-E26 audits, or their episode logs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENCEWORLD_DIR = REPO_ROOT / "artifacts" / "benchmarks" / "scienceworld"
E16_AUDIT = SCIENCEWORLD_DIR / "ensr_mechanism_ablation_v1_audit.json"
KG_AUDIT = SCIENCEWORLD_DIR / "ensr_v3_ablation_no_semantic_kg_audit.json"
E23_AUDIT = SCIENCEWORLD_DIR / "e23_evidence_scheduler_development_audit_v1.json"
E24_AUDIT = SCIENCEWORLD_DIR / "e24_evidence_anchor_development_audit_v1.json"
E25_AUDIT = SCIENCEWORLD_DIR / "e25_evidence_anchor_cross_model_development_audit_v1.json"
E26_AUDIT = SCIENCEWORLD_DIR / "e26_evidence_anchor_ablation_audit_v1.json"
KG_WITHOUT_ROOT = (
    SCIENCEWORLD_DIR
    / "ensr_v3_ablation_no_semantic_kg_runs"
    / "scienceworld_ensr_v3"
    / "bit_qwen3_235b"
    / "dev"
    / "ensr_v2"
)
KG_WITH_ROOT = (
    SCIENCEWORLD_DIR
    / "ensr_v2_runs_amended_v23r_full"
    / "scienceworld_ensr_v2"
    / "bit_qwen3_235b"
    / "dev"
    / "ensr_v2"
)
OUTPUT_DIR = REPO_ROOT / "artifacts" / "paper" / "figures" / "figure7"
OUTPUT_STEM = OUTPUT_DIR / "Figure_7_ENSR_mechanisms_nature_v2"


COLORS = {
    "text": "#26364A",
    "subtext": "#667085",
    "grid": "#E8EBEF",
    "control": "#C95E72",
    "control_light": "#F2D6DC",
    "without": "#A9B1BA",
    "without_dark": "#6576A5",
    "positive": "#C95E72",
    "negative": "#6576A5",
    "teal": "#5F8F82",
    "teal_light": "#E5F0ED",
    "ochre": "#B78336",
    "violet": "#9B93B9",
    "gold": "#C95E72",
    "purple": "#6576A5",
    "neutral": "#E8EBEF",
    "white": "#FFFFFF",
}


E16_COMPONENTS = [
    (
        "full_control_minus_without_hierarchical_subgoals",
        "Hierarchical subgoals",
    ),
    (
        "full_control_minus_without_symbolic_transition_verifier",
        "Symbolic transition verifier",
    ),
    (
        "full_control_minus_without_obligation_conditioned_retrieval",
        "Obligation-conditioned retrieval",
    ),
    (
        "full_control_minus_accumulating_fragment_memory",
        "Fragment replacement",
    ),
]


MECHANISM_METRICS = [
    ("knowledge_graph_action_count", "KG-guided actions"),
    ("symbolic_goal_action_count", "Symbolic-goal actions"),
    ("exploration_action_count", "Exploration actions"),
    ("measurement_poll_action_count", "Measurement polling"),
    ("focus_stage_advance_count", "Focus-stage advances"),
    ("contradictory_fact_rejection_count", "Contradiction rejections"),
    ("local_replan_count", "Local replans"),
    ("no_progress_action_count", "No-progress events"),
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clipped_score(value: float | int | None) -> float:
    return float(np.clip(0 if value is None else value, 0, 100))


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
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
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


def extract_component_effects(e16: dict) -> list[dict]:
    if not e16.get("complete_matrix") or int(e16.get("observed_episode_count", 0)) != 300:
        raise RuntimeError("E16 frozen matrix is incomplete.")
    rows: list[dict] = []
    for key, label in E16_COMPONENTS:
        comparison = e16["comparisons"][key]
        score = comparison["clipped_score"]
        success = comparison["task_success_rate"]
        permutation = comparison["score_permutation"]
        rows.append(
            {
                "comparison": key,
                "component": label,
                "paired_tasks": int(score["paired_task_count"]),
                "score_delta": float(score["point_estimate"]),
                "score_low": float(score["ci95_low"]),
                "score_high": float(score["ci95_high"]),
                "success_delta_pp": 100.0 * float(success["point_estimate"]),
                "success_low_pp": 100.0 * float(success["ci95_low"]),
                "success_high_pp": 100.0 * float(success["ci95_high"]),
                "p_raw": float(permutation["p_value"]),
                "p_holm": float(permutation["holm_adjusted_p_value"]),
                "holm_significant": float(permutation["holm_adjusted_p_value"]) < 0.05,
            }
        )
    return rows


def extract_kg_task_effects(kg: dict) -> list[dict]:
    if not kg.get("complete_pairing") or int(kg.get("paired_episode_count", 0)) != 60:
        raise RuntimeError("Semantic-KG paired ablation is incomplete.")
    rows = []
    for record in kg["task_comparisons"]:
        rows.append(
            {
                "task_id": str(record["task_id"]),
                "paired_episodes": int(record["paired_episodes"]),
                "without_kg_mean_score": float(record["old_mean_score"]),
                "with_kg_mean_score": float(record["new_mean_score"]),
                "score_delta": float(record["score_delta"]),
                "without_kg_success_rate": float(record["old_success_rate"]),
                "with_kg_success_rate": float(record["new_success_rate"]),
                "success_delta": float(record["success_delta"]),
            }
        )
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 KG task comparisons, found {len(rows)}.")
    rows.sort(key=lambda row: (row["score_delta"], row["success_delta"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def extract_mechanism_rates(kg: dict) -> list[dict]:
    without = kg["old_summary"]
    with_kg = kg["new_summary"]
    without_steps = float(without["environment_steps"])
    with_steps = float(with_kg["environment_steps"])
    rows: list[dict] = []
    for key, label in MECHANISM_METRICS:
        without_rate = 100.0 * float(without["mechanism_totals"].get(key, 0.0)) / without_steps
        with_rate = 100.0 * float(with_kg["mechanism_totals"].get(key, 0.0)) / with_steps
        rows.append(
            {
                "metric_key": key,
                "metric": label,
                "without_kg_per_100_steps": without_rate,
                "with_kg_per_100_steps": with_rate,
                "delta_per_100_steps": with_rate - without_rate,
            }
        )
    return rows


def extract_scheduler_trials(e23: dict, e24: dict, e25: dict) -> list[dict]:
    specs = [
        (e23, "E23", "Unanchored", "Gate fail", "X", COLORS["control"]),
        (e24, "E24", "Anchored 32B", "Pass", "o", COLORS["teal"]),
        (e25, "E25", "Anchored 8B", "Trace boundary", "D", COLORS["ochre"]),
    ]
    rows: list[dict] = []
    for audit, experiment, variant, outcome, marker, color in specs:
        if not audit.get("complete_pairing") or int(audit.get("paired_episode_count", 0)) != 20:
            raise RuntimeError(f"{experiment} frozen pairing is incomplete.")
        effect = audit["scheduler_minus_control"]
        rows.append(
            {
                "experiment": experiment,
                "variant": variant,
                "outcome": outcome,
                "paired_episodes": int(audit["paired_episode_count"]),
                "retrieval_reduction_pct": 100.0 * float(effect["obligation_retrieval_reduction"]),
                "success_delta_pp": 100.0 * float(effect["success_rate"]["point_estimate"]),
                "score_delta": float(effect["mean_clipped_score"]["point_estimate"]),
                "score_ci95_low": float(effect["mean_clipped_score"]["ci95_low"]),
                "score_ci95_high": float(effect["mean_clipped_score"]["ci95_high"]),
                "trace_replay_integrity": bool(audit["development_gate_checks"].get("trace_replay_integrity", True)),
                "development_gate_passed": bool(audit["development_gate_passed"]),
                "marker": marker,
                "color": color,
            }
        )
    return rows


def extract_anchor_ablation(e26: dict) -> tuple[list[dict], dict]:
    if not e26.get("complete_pairing") or int(e26.get("paired_episode_count", 0)) != 20:
        raise RuntimeError("E26 frozen pairing is incomplete.")
    arms = [
        ("No anchor", e26["summaries"]["evidence_scheduler_no_anchor"]),
        ("First-debt anchor", e26["summaries"]["evidence_anchor_scheduler"]),
    ]
    rows = []
    for condition, summary in arms:
        rows.append(
            {
                "condition": condition,
                "paired_episodes": int(summary["episodes"]),
                "mean_clipped_score": float(summary["mean_clipped_score"]),
                "success_rate_pct": 100.0 * float(summary["success_rate"]),
                "obligation_retrieval_calls": int(summary["obligation_retrieval_calls"]),
                "environment_steps": int(summary["environment_steps"]),
                "verified_transition_rate_pct": 100.0 * float(summary["verified_transition_rate"]),
                "uncontrolled_failures": int(summary["uncontrolled_failures"]),
            }
        )
    contrast = e26["contrasts"]["anchored_minus_unanchored"]
    meta = {
        "score_delta": float(contrast["mean_clipped_score"]["point_estimate"]),
        "score_ci95_low": float(contrast["mean_clipped_score"]["ci95_low"]),
        "score_ci95_high": float(contrast["mean_clipped_score"]["ci95_high"]),
        "success_delta_pp": 100.0 * float(contrast["success_rate"]["point_estimate"]),
        "success_ci95_low_pp": 100.0 * float(contrast["success_rate"]["ci95_low"]),
        "success_ci95_high_pp": 100.0 * float(contrast["success_rate"]["ci95_high"]),
        "anchor_retrieval_increment": int(e26["contrasts"]["anchor_obligation_retrieval_increment"]),
        "ablation_gate_passed": bool(e26["ablation_gate_passed"]),
    }
    return rows, meta


def load_episode_map(root: Path) -> dict[tuple[str, int, int], dict]:
    rows: dict[tuple[str, int, int], dict] = {}
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        record = read_json(path)
        key = (str(record["task_id"]), int(record["variation"]), int(record["seed"]))
        rows[key] = record
    if len(rows) != 60:
        raise RuntimeError(f"Expected 60 episodes in {root}, found {len(rows)}.")
    return rows


def select_trajectory_pairs() -> list[dict]:
    without = load_episode_map(KG_WITHOUT_ROOT)
    with_kg = load_episode_map(KG_WITH_ROOT)
    if set(without) != set(with_kg):
        raise RuntimeError("Semantic-KG episode keys do not pair exactly.")

    selections: list[tuple[str, str, tuple[str, int, int]]] = []
    for task_id, kind in [("4-1", "Benefit case"), ("1-2", "Boundary case")]:
        keys = [key for key in without if key[0] == task_id]
        if not keys:
            raise RuntimeError(f"No paired episodes for task {task_id}.")
        if kind == "Benefit case":
            selected = max(
                keys,
                key=lambda key: clipped_score(with_kg[key]["final_score"])
                - clipped_score(without[key]["final_score"]),
            )
        else:
            selected = min(
                keys,
                key=lambda key: clipped_score(with_kg[key]["final_score"])
                - clipped_score(without[key]["final_score"]),
            )
        selections.append((task_id, kind, selected))

    pairs = []
    for task_id, kind, key in selections:
        old = without[key]
        new = with_kg[key]
        pairs.append(
            {
                "task_id": task_id,
                "kind": kind,
                "variation": key[1],
                "seed": key[2],
                "score_delta": clipped_score(new["final_score"])
                - clipped_score(old["final_score"]),
                "without": old,
                "with": new,
            }
        )
    return pairs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty source table: {path}")
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_source_data(
    components: list[dict],
    tasks: list[dict],
    scheduler_trials: list[dict],
    anchor_rows: list[dict],
    anchor_meta: dict,
    pairs: list[dict],
) -> None:
    write_csv(OUTPUT_DIR / "Figure_7a_component_effects_v2.csv", components)
    write_csv(OUTPUT_DIR / "Figure_7b_semantic_kg_task_effects_v2.csv", tasks)
    write_csv(
        OUTPUT_DIR / "Figure_7c_scheduler_development_v2.csv",
        [{key: value for key, value in row.items() if key not in {"marker", "color"}} for row in scheduler_trials],
    )
    anchor_export = []
    for row in anchor_rows:
        merged = dict(row)
        merged.update(anchor_meta)
        anchor_export.append(merged)
    write_csv(OUTPUT_DIR / "Figure_7c_anchor_ablation_v2.csv", anchor_export)

    trajectory_rows: list[dict] = []
    for pair in pairs:
        for condition_key, condition_label in [("without", "Without semantic KG"), ("with", "With semantic KG")]:
            episode = pair[condition_key]
            obligation_steps = {int(item["step"]) for item in episode.get("obligations", [])}
            for step in episode.get("trajectory", []):
                trajectory_rows.append(
                    {
                        "case": pair["kind"],
                        "task_id": pair["task_id"],
                        "variation": pair["variation"],
                        "seed": pair["seed"],
                        "condition": condition_label,
                        "step": int(step["step_index"]),
                        "score": clipped_score(step.get("score", 0)),
                        "reward": float(step.get("reward", 0)),
                        "action_source": str(step.get("action_source", "unknown")),
                        "action": str(step.get("action", "")),
                        "obligation_opened": int(int(step["step_index"]) in obligation_steps),
                        "final_status": str(episode.get("status", "unknown")),
                        "task_success": int(bool(episode.get("task_success", False))),
                    }
                )
    write_csv(OUTPUT_DIR / "Figure_7d_representative_trajectories_v2.csv", trajectory_rows)


def draw_panel_a(ax: plt.Axes, components: list[dict]) -> None:
    y_positions = np.arange(len(components))[::-1]
    ax.axvspan(-4, 0, color="#F6F7F9", zorder=0)
    ax.axvline(0, color=COLORS["subtext"], linewidth=0.9, zorder=1)

    for y, row in zip(y_positions, components):
        color = COLORS["control"]
        alpha = 1.0 if row["holm_significant"] else 0.72
        ax.plot(
            [row["score_low"], row["score_high"]],
            [y, y],
            color=color,
            linewidth=1.8,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot([row["score_low"], row["score_low"]], [y - 0.08, y + 0.08], color=color, linewidth=0.9, alpha=alpha)
        ax.plot([row["score_high"], row["score_high"]], [y - 0.08, y + 0.08], color=color, linewidth=0.9, alpha=alpha)
        ax.scatter(
            row["score_delta"],
            y,
            s=30,
            facecolor=color if row["holm_significant"] else COLORS["white"],
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            16.9,
            y + 0.09,
            f"Δsuccess {row['success_delta_pp']:+.1f} pp",
            ha="right",
            va="bottom",
            fontsize=5.8,
            color=COLORS["subtext"],
        )
        ax.text(
            16.9,
            y - 0.09,
            f"Holm P={row['p_holm']:.3f}",
            ha="right",
            va="top",
            fontsize=5.8,
            color=COLORS["control"] if row["holm_significant"] else COLORS["subtext"],
            fontweight="semibold" if row["holm_significant"] else "normal",
        )

    ax.set_yticks(y_positions, [row["component"] for row in components])
    ax.set_xlim(-4, 18)
    ax.set_ylim(-0.6, len(components) - 0.4)
    ax.set_xlabel("Δ clipped score (full ENSR − ablated), 95% CI")
    ax.set_title("Component contributions are selective", loc="left", pad=9)
    ax.text(
        0,
        1.01,
        "E16 development set · 60 episodes per arm · 30 paired tasks",
        transform=ax.transAxes,
        fontsize=6.1,
        color=COLORS["subtext"],
        va="bottom",
    )
    ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.65)
    clean_axis(ax)
    panel_label(ax, "a", x=-0.18)


def draw_panel_b(ax: plt.Axes, tasks: list[dict], kg: dict) -> None:
    ax.set_xlim(-0.8, len(tasks) - 0.2)
    ax.set_ylim(-1.5, 2.65)
    ax.axis("off")

    max_abs = max(abs(row["score_delta"]) for row in tasks)
    score_norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    score_cmap = mcolors.LinearSegmentedColormap.from_list(
        "kg_effect",
        [COLORS["negative"], "#F5F5F3", COLORS["control"]],
    )
    success_colors = {
        -1.0: COLORS["negative"],
        -0.5: "#B8C3DF",
        0.0: COLORS["neutral"],
        0.5: "#E6A7B3",
        1.0: COLORS["positive"],
    }

    for x, row in enumerate(tasks):
        ax.add_patch(
            Rectangle(
                (x - 0.46, 0.55),
                0.92,
                0.78,
                facecolor=score_cmap(score_norm(row["score_delta"])),
                edgecolor="white",
                linewidth=0.45,
            )
        )
        delta = float(row["success_delta"])
        nearest = min(success_colors, key=lambda key: abs(key - delta))
        ax.add_patch(
            Rectangle(
                (x - 0.46, -0.31),
                0.92,
                0.58,
                facecolor=success_colors[nearest],
                edgecolor="white",
                linewidth=0.45,
            )
        )
        ax.text(
            x,
            -0.48,
            row["task_id"],
            rotation=90,
            ha="center",
            va="top",
            fontsize=4.7,
            color=COLORS["subtext"],
        )

    ax.text(-0.63, 0.94, "Score Δ", ha="right", va="center", fontsize=6.1, color=COLORS["text"])
    ax.text(-0.63, -0.02, "Success Δ", ha="right", va="center", fontsize=6.1, color=COLORS["text"])
    ax.text(
        0,
        1.40,
        "Tasks ranked by score gain →",
        fontsize=5.8,
        color=COLORS["subtext"],
        ha="left",
        va="bottom",
    )

    effect = kg["new_minus_old"]
    score = effect["mean_clipped_score"]
    success = effect["success_rate"]
    ax.text(
        0,
        1.73,
        f"Mean score {score['point_estimate']:+.1f} [{score['ci95_low']:.1f}, {score['ci95_high']:.1f}]  ·  "
        f"success {100*success['point_estimate']:+.1f} pp [{100*success['ci95_low']:.1f}, {100*success['ci95_high']:.1f}]",
        fontsize=5.8,
        color=COLORS["text"],
        ha="left",
        va="bottom",
    )

    outcomes = effect["episode_outcomes"]
    total = sum(int(outcomes[key]) for key in ["improved", "tied", "regressed"])
    x0, total_width, y0, height = 0.0, 18.0, 2.02, 0.28
    cursor = x0
    outcome_spec = [
        ("improved", "Improved", COLORS["control"]),
        ("tied", "Tied", COLORS["neutral"]),
        ("regressed", "Regressed", COLORS["negative"]),
    ]
    for key, label, color in outcome_spec:
        count = int(outcomes[key])
        width = total_width * count / total
        ax.add_patch(
            FancyBboxPatch(
                (cursor, y0),
                width,
                height,
                boxstyle="round,pad=0.015,rounding_size=0.08",
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
            )
        )
        cursor += width
    ax.text(
        total_width / 2,
        y0 + height / 2,
        "16 improved   ·   38 tied   ·   6 regressed",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COLORS["text"],
    )
    ax.text(
        total_width + 0.55,
        y0 + height / 2,
        "60 paired episodes",
        ha="left",
        va="center",
        fontsize=5.6,
        color=COLORS["subtext"],
    )

    cax = ax.inset_axes([0.77, 0.80, 0.20, 0.038])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=score_norm, cmap=score_cmap),
        cax=cax,
        orientation="horizontal",
    )
    cb.ax.tick_params(labelsize=4.8, length=2, pad=1)
    cb.outline.set_visible(False)
    cb.set_label("Task score Δ", fontsize=5.2, labelpad=1)

    ax.set_title("Semantic KG benefits are large but task-dependent", loc="left", pad=9)
    panel_label(ax, "b", x=-0.09, y=1.05)


def draw_panel_c(
    ax: plt.Axes,
    scheduler_trials: list[dict],
    anchor_rows: list[dict],
    anchor_meta: dict,
) -> None:
    ax.axis("off")
    ax.set_title("Evidence-debt scheduling needs an anchor", loc="left", pad=9)
    ax.text(
        0,
        0.982,
        "Development evidence only; E23–E25 are separate tests and are not pooled",
        transform=ax.transAxes,
        fontsize=5.8,
        color=COLORS["subtext"],
        va="bottom",
    )
    panel_label(ax, "c", x=-0.17, y=1.06)

    trade_ax = ax.inset_axes([0.04, 0.57, 0.92, 0.31])
    trade_ax.axhspan(-0.45, 2.25, color=COLORS["teal_light"], alpha=0.72, zorder=0)
    trade_ax.axhline(0, color=COLORS["subtext"], linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    for row in scheduler_trials:
        trade_ax.scatter(
            row["retrieval_reduction_pct"],
            row["success_delta_pp"],
            marker=row["marker"],
            s=42,
            facecolor=row["color"] if row["marker"] != "X" else "none",
            edgecolor=row["color"],
            linewidth=1.35,
            zorder=3,
        )
        if row["experiment"] == "E23":
            offset, ha, va = (-6, 6), "right", "bottom"
        elif row["experiment"] == "E24":
            offset, ha, va = (0, 7), "center", "bottom"
        else:
            offset, ha, va = (-2, -10), "center", "top"
        trade_ax.annotate(
            f"{row['experiment']} {row['variant']}\n{row['outcome'].lower()} · score {row['score_delta']:+.1f}",
            (row["retrieval_reduction_pct"], row["success_delta_pp"]),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=4.9,
            color=COLORS["text"],
        )
    trade_ax.text(
        0.99,
        0.95,
        "Outcome retained",
        transform=trade_ax.transAxes,
        fontsize=5.1,
        color=COLORS["teal"],
        va="top",
        ha="right",
        fontweight="semibold",
    )
    trade_ax.set_xlim(50, 92)
    trade_ax.set_ylim(-6.4, 3.0)
    trade_ax.set_xticks([50, 60, 70, 80, 90])
    trade_ax.set_yticks([-5, 0])
    trade_ax.set_xlabel("Obligation retrieval reduction (%)", labelpad=1)
    trade_ax.set_ylabel("Δ success (pp)", labelpad=1)
    trade_ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.5)
    clean_axis(trade_ax)

    pair_ax = ax.inset_axes([0.02, 0.01, 0.96, 0.47])
    pair_ax.set_xlim(-0.56, 1.25)
    pair_ax.set_ylim(-0.58, 4.05)
    pair_ax.axis("off")
    pair_ax.text(-0.52, 3.87, "E26 · same 20 cases", fontsize=5.8, color=COLORS["text"], fontweight="semibold")
    pair_ax.text(0.00, 3.48, "No anchor", ha="center", fontsize=5.4, color=COLORS["without_dark"], fontweight="semibold")
    pair_ax.text(1.00, 3.48, "First-debt anchor", ha="center", fontsize=5.4, color=COLORS["teal"], fontweight="semibold")

    no_anchor = next(row for row in anchor_rows if row["condition"] == "No anchor")
    anchored = next(row for row in anchor_rows if row["condition"] == "First-debt anchor")
    metric_rows = [
        ("Success", no_anchor["success_rate_pct"], anchored["success_rate_pct"], "{:.0f}%", 3.0, COLORS["teal"],
         f"+{anchor_meta['success_delta_pp']:.0f} pp [{anchor_meta['success_ci95_low_pp']:.0f}, {anchor_meta['success_ci95_high_pp']:.0f}]"),
        ("Environment steps", no_anchor["environment_steps"], anchored["environment_steps"], "{:.0f}", 2.1, COLORS["teal"],
         f"{anchored['environment_steps'] - no_anchor['environment_steps']:+.0f}"),
        ("Retrieval calls", no_anchor["obligation_retrieval_calls"], anchored["obligation_retrieval_calls"], "{:.0f}", 1.2, COLORS["ochre"],
         f"+{anchor_meta['anchor_retrieval_increment']:.0f}"),
        ("Clipped score", no_anchor["mean_clipped_score"], anchored["mean_clipped_score"], "{:.2f}", 0.3, COLORS["control"],
         f"{anchor_meta['score_delta']:+.2f} [{anchor_meta['score_ci95_low']:.2f}, {anchor_meta['score_ci95_high']:.2f}]"),
    ]
    for label, left_value, right_value, value_format, y, delta_color, delta_label in metric_rows:
        pair_ax.plot([0.08, 0.92], [y, y], color=COLORS["grid"], linewidth=1.1, zorder=1)
        pair_ax.scatter(0, y, s=26, facecolor="white", edgecolor=COLORS["without_dark"], linewidth=1.1, zorder=2)
        pair_ax.scatter(1, y, s=30, facecolor=COLORS["teal"], edgecolor="white", linewidth=0.6, zorder=3)
        pair_ax.text(-0.52, y, label, ha="left", va="center", fontsize=5.3, color=COLORS["text"])
        pair_ax.text(0, y - 0.26, value_format.format(left_value), ha="center", va="top", fontsize=5.1, color=COLORS["without_dark"])
        pair_ax.text(1, y - 0.26, value_format.format(right_value), ha="center", va="top", fontsize=5.1, color=COLORS["teal"])
        pair_ax.text(0.5, y + 0.12, delta_label, ha="center", va="bottom", fontsize=4.8, color=delta_color, fontweight="semibold")
    pair_ax.text(
        -0.52,
        -0.47,
        "Anchor improves completion and efficiency; score interval crosses zero.",
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=COLORS["subtext"],
    )


def trajectory_xy(episode: dict) -> tuple[np.ndarray, np.ndarray]:
    trajectory = episode.get("trajectory", [])
    x = np.array([0] + [int(step["step_index"]) for step in trajectory], dtype=float)
    y = np.array([0.0] + [clipped_score(step.get("score", 0)) for step in trajectory], dtype=float)
    return x, y


def score_at_step(episode: dict, step_index: int) -> float:
    trajectory = episode.get("trajectory", [])
    if not trajectory:
        return 0.0
    eligible = [step for step in trajectory if int(step["step_index"]) <= step_index]
    return clipped_score(eligible[-1].get("score", 0)) if eligible else 0.0


def draw_trajectory_axis(ax: plt.Axes, pair: dict, show_legend: bool) -> None:
    styles = [
        ("without", "Without semantic KG", COLORS["without_dark"]),
        ("with", "With semantic KG", COLORS["control"]),
    ]
    for condition, label, color in styles:
        episode = pair[condition]
        x, y = trajectory_xy(episode)
        ax.plot(x, y, color=color, linewidth=1.35, drawstyle="steps-post", label=label, zorder=2)
        obligation_steps = sorted({int(item["step"]) for item in episode.get("obligations", [])})
        if obligation_steps:
            obligation_y = [score_at_step(episode, step) for step in obligation_steps]
            ax.scatter(
                obligation_steps,
                obligation_y,
                marker="v",
                s=13,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )
        endpoint_marker = "*" if episode.get("task_success") else "X"
        ax.scatter(x[-1], y[-1], marker=endpoint_marker, s=35 if endpoint_marker == "*" else 22, color=color, zorder=4)

    old = pair["without"]
    new = pair["with"]
    ax.set_xlim(0, max(int(old["environment_steps"]), int(new["environment_steps"]), 20) + 3)
    ax.set_ylim(-4, 106)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_ylabel("Score")
    ax.set_xlabel("Environment action")
    clean_axis(ax)
    ax.set_title(
        f"{pair['kind']}: task {pair['task_id']} · variation {pair['variation']} · Δscore {pair['score_delta']:+.0f}",
        loc="left",
        fontsize=6.6,
        pad=4,
        color=COLORS["positive"] if pair["score_delta"] > 0 else COLORS["negative"],
    )
    old_oblig = len(old.get("obligations", []))
    new_oblig = len(new.get("obligations", []))
    new_kg = int(new.get("mechanism_metrics", {}).get("knowledge_graph_action_count", 0))
    ax.text(
        0.995,
        0.06,
        f"actions {old['environment_steps']}→{new['environment_steps']}  ·  obligations {old_oblig}→{new_oblig}  ·  KG actions {new_kg}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=COLORS["subtext"],
    )
    if show_legend:
        obligation_handle = Line2D([], [], marker="v", linestyle="None", markersize=4.5, color=COLORS["subtext"], label="Evidence obligation")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [obligation_handle], labels + ["Evidence obligation"], loc="upper right", fontsize=5.4, ncol=3, handlelength=1.6, columnspacing=0.8)


def draw_figure(
    components: list[dict],
    tasks: list[dict],
    scheduler_trials: list[dict],
    anchor_rows: list[dict],
    anchor_meta: dict,
    pairs: list[dict],
    kg: dict,
) -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(11.4, 7.55), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.975,
        bottom=0.075,
        top=0.955,
        width_ratios=[0.93, 1.32],
        height_ratios=[0.86, 1.14],
        wspace=0.25,
        hspace=0.38,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    trace_grid = outer[1, 1].subgridspec(2, 1, hspace=0.42)
    ax_d1 = fig.add_subplot(trace_grid[0, 0])
    ax_d2 = fig.add_subplot(trace_grid[1, 0])

    draw_panel_a(ax_a, components)
    draw_panel_b(ax_b, tasks, kg)
    draw_panel_c(ax_c, scheduler_trials, anchor_rows, anchor_meta)
    draw_trajectory_axis(ax_d1, pairs[0], show_legend=True)
    draw_trajectory_axis(ax_d2, pairs[1], show_legend=False)
    ax_d1.text(
        0,
        1.22,
        "Representative paired trajectories reveal benefit and boundary",
        transform=ax_d1.transAxes,
        fontsize=8.5,
        fontweight="semibold",
        ha="left",
        va="bottom",
        color=COLORS["text"],
    )
    ax_d1.text(
        0,
        1.09,
        "Same task, variation and seed; triangles mark evidence obligations",
        transform=ax_d1.transAxes,
        fontsize=6.0,
        color=COLORS["subtext"],
        ha="left",
        va="bottom",
    )
    panel_label(ax_d1, "d", x=-0.09, y=1.20)
    return fig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_caption(
    e16: dict,
    kg: dict,
    scheduler_trials: list[dict],
    anchor_rows: list[dict],
    anchor_meta: dict,
    pairs: list[dict],
) -> None:
    score = kg["new_minus_old"]["mean_clipped_score"]
    success = kg["new_minus_old"]["success_rate"]
    outcomes = kg["new_minus_old"]["episode_outcomes"]
    e23, e24, e25 = scheduler_trials
    no_anchor = next(row for row in anchor_rows if row["condition"] == "No anchor")
    anchored = next(row for row in anchor_rows if row["condition"] == "First-debt anchor")
    caption = f"""# Figure 7 | ENSR mechanism contributions, semantic grounding and evidence-debt scheduling

**a,** Frozen E16 development-set ablations. Points show the paired mean clipped-score contribution of the full ENSR system relative to each ablated arm across 30 tasks; horizontal lines show percentile-bootstrap 95% confidence intervals based on 20,000 resamples. Filled points indicate a Holm-adjusted paired sign-flip permutation *P* < 0.05. Success-rate differences are shown in percentage points. Each arm contains 60 episodes. **b,** Thirty tasks from the paired semantic-KG ablation, ranked by score change after adding the semantic KG. The upper strip encodes task-level mean score change and the lower strip encodes success-rate change. Across 60 paired episodes, the semantic KG changed mean clipped score by {score['point_estimate']:.1f} points (95% CI {score['ci95_low']:.1f} to {score['ci95_high']:.1f}) and success by {100*success['point_estimate']:.1f} percentage points (95% CI {100*success['ci95_low']:.1f} to {100*success['ci95_high']:.1f}); {outcomes['improved']} episode pairs improved, {outcomes['tied']} tied and {outcomes['regressed']} regressed. **c,** Evidence-debt scheduling results from separate 20-pair development tests. The upper plot shows the reduction in obligation-retrieval calls against the success-rate difference from the corresponding control. E23 reduced retrieval by {e23['retrieval_reduction_pct']:.1f}% but reduced success by {e23['success_delta_pp']:.1f} percentage points and failed the development gate. The anchored E24 scheduler reduced retrieval by {e24['retrieval_reduction_pct']:.1f}% without changing score or success and passed all frozen gates. E25 transferred the outcome and retrieval result to the 8B model profile but failed the strict trace-integrity gate because one pair had no shared model-call prefix. The lower slope display isolates the first-debt anchor on the same 20 E26 cases. Anchoring increased success from {no_anchor['success_rate_pct']:.0f}% to {anchored['success_rate_pct']:.0f}%, reduced environment steps from {no_anchor['environment_steps']} to {anchored['environment_steps']}, and increased retrieval calls from {no_anchor['obligation_retrieval_calls']} to {anchored['obligation_retrieval_calls']}. Mean clipped score changed from {no_anchor['mean_clipped_score']:.2f} to {anchored['mean_clipped_score']:.2f}; the paired difference was {anchor_meta['score_delta']:.2f} points (95% CI {anchor_meta['score_ci95_low']:.2f} to {anchor_meta['score_ci95_high']:.2f}). **d,** Action-level cumulative-score trajectories for a benefit case (task {pairs[0]['task_id']}, variation {pairs[0]['variation']}) and a boundary case (task {pairs[1]['task_id']}, variation {pairs[1]['variation']}) under identical task, variation and seed. Triangles denote evidence obligations, stars terminal success and crosses non-successful termination. All panels are mechanism or development analyses and do not replace the held-out E13 result or establish global state of the art.
"""
    (OUTPUT_DIR / "Figure_7_caption_nature_v2.md").write_text(caption, encoding="utf-8")


def write_qa_report(
    paths: list[Path],
    components: list[dict],
    tasks: list[dict],
    scheduler_trials: list[dict],
    anchor_rows: list[dict],
    pairs: list[dict],
) -> None:
    report = [
        "# Figure 7 QA report",
        "",
        "- Source-only figure: no simulated observations.",
        f"- E16 component rows: {len(components)}.",
        f"- Semantic-KG paired tasks: {len(tasks)}.",
        f"- Scheduler development tests: {len(scheduler_trials)}, each with 20 paired episodes.",
        f"- E26 anchor arms: {len(anchor_rows)}, each with 20 paired episodes.",
        f"- Representative paired cases: {len(pairs)}.",
        "- E16 boundary: development-set supporting evidence.",
        "- Semantic-KG boundary: development-set paired ablation.",
        "- Scheduler boundary: E23-E26 are development analyses, are not pooled, and do not support a global SOTA claim.",
        "- E25 is labelled as a trace boundary because its strict trace-integrity gate failed.",
        "- E26 shows both the completion benefit and the clipped-score trade-off of the first-debt anchor.",
        "- SVG text remains editable; PDF uses TrueType font embedding.",
        "- Raster exports: 600 dpi PNG and LZW-compressed TIFF.",
        "",
        "## File hashes",
        "",
    ]
    for path in paths:
        report.append(f"- `{path.name}`: `{sha256(path)}`")
    report.extend(
        [
            "",
            "## Source hashes",
            "",
            f"- `{E16_AUDIT.name}`: `{sha256(E16_AUDIT)}`",
            f"- `{KG_AUDIT.name}`: `{sha256(KG_AUDIT)}`",
            f"- `{E23_AUDIT.name}`: `{sha256(E23_AUDIT)}`",
            f"- `{E24_AUDIT.name}`: `{sha256(E24_AUDIT)}`",
            f"- `{E25_AUDIT.name}`: `{sha256(E25_AUDIT)}`",
            f"- `{E26_AUDIT.name}`: `{sha256(E26_AUDIT)}`",
        ]
    )
    (OUTPUT_DIR / "Figure_7_QA_report_v2.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    e16 = read_json(E16_AUDIT)
    kg = read_json(KG_AUDIT)
    e23 = read_json(E23_AUDIT)
    e24 = read_json(E24_AUDIT)
    e25 = read_json(E25_AUDIT)
    e26 = read_json(E26_AUDIT)
    components = extract_component_effects(e16)
    tasks = extract_kg_task_effects(kg)
    scheduler_trials = extract_scheduler_trials(e23, e24, e25)
    anchor_rows, anchor_meta = extract_anchor_ablation(e26)
    pairs = select_trajectory_pairs()
    export_source_data(components, tasks, scheduler_trials, anchor_rows, anchor_meta, pairs)

    fig = draw_figure(components, tasks, scheduler_trials, anchor_rows, anchor_meta, pairs, kg)
    png_path = OUTPUT_STEM.with_suffix(".png")
    svg_path = OUTPUT_STEM.with_suffix(".svg")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    tiff_path = OUTPUT_STEM.with_suffix(".tiff")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    write_caption(e16, kg, scheduler_trials, anchor_rows, anchor_meta, pairs)
    write_qa_report(
        [png_path, svg_path, pdf_path, tiff_path],
        components,
        tasks,
        scheduler_trials,
        anchor_rows,
        pairs,
    )
    for path in [png_path, svg_path, pdf_path, tiff_path]:
        print(path)


if __name__ == "__main__":
    main()
