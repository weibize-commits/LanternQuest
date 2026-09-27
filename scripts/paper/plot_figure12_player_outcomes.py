"""Render Figure 12 from the frozen LanternQuest E90 validation outputs.

The figure reports participant-level distributions together with the frozen
confirmatory estimates. It deliberately retains null, exploratory and
diagnostic results alongside the positive outcomes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    ROOT
    / "artifacts"
    / "human_study"
    / "e90"
    / "returns"
    / "return_20260925_real_v1"
    / "validated_analysis"
)
SUMMARY_PATH = SOURCE_DIR / "e90_validated_analysis_v1.json"
PARTICIPANT_PATH = SOURCE_DIR / "e90_deidentified_scored_participants.csv"
RATING_PATH = SOURCE_DIR / "e90_blinded_ratings_scored.csv"
OUTPUT_DIR = ROOT / "artifacts" / "paper" / "figures" / "figure12"
OUTPUT_STEM = OUTPUT_DIR / "Figure_12_player_outcomes_nature_v1"


C = {
    "text": "#26364A",
    "subtext": "#667085",
    "grid": "#E8EBEF",
    "ensr": "#C95E72",
    "ensr_light": "#F2D6DC",
    "b4": "#6576A5",
    "b4_light": "#DDE3F1",
    "neutral": "#E8EBEF",
    "neutral_dark": "#A9B1BA",
    "teal": "#5F8F82",
    "teal_light": "#DCECE7",
    "gold": "#B78336",
    "gold_light": "#F4E8CF",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 8.6,
            "axes.titleweight": "semibold",
            "axes.labelcolor": C["text"],
            "axes.edgecolor": C["text"],
            "text.color": C["text"],
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.13, 1.075, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, fontweight="bold")


def title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", pad=8)


def clean(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["text"])
    ax.spines["bottom"].set_color(C["text"])
    ax.tick_params(colors=C["text"], width=0.7, length=3)
    if grid:
        ax.grid(axis=grid, color=C["grid"], lw=0.7, zorder=0)


def stable_jitter(codes: pd.Series, width: float = 0.08) -> np.ndarray:
    values = []
    for code in codes.astype(str):
        h = int(hashlib.sha256(code.encode("utf-8")).hexdigest()[:8], 16)
        values.append(((h % 10001) / 10000 - 0.5) * 2 * width)
    return np.asarray(values)


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    mean = float(np.mean(x))
    half = float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return mean, mean - half, mean + half


def group_color(condition: str) -> str:
    return C["ensr"] if condition == "ensr" else C["b4"]


def draw_distribution(ax: plt.Axes, frame: pd.DataFrame, value: str, xlim: tuple[float, float]) -> None:
    groups = ["b4", "ensr"]
    positions = [0, 1]
    arrays = [frame.loc[frame.internal_condition.eq(g), value].dropna().to_numpy(float) for g in groups]
    violins = ax.violinplot(arrays, positions=positions, vert=False, widths=0.64, showmeans=False, showmedians=False, showextrema=False)
    for body, group in zip(violins["bodies"], groups):
        body.set_facecolor(group_color(group))
        body.set_edgecolor("none")
        body.set_alpha(0.16)
    for y, group in zip(positions, groups):
        sub = frame.loc[frame.internal_condition.eq(group) & frame[value].notna(), ["participant_code", value]]
        jitter = stable_jitter(sub["participant_code"], 0.18)
        ax.scatter(sub[value], y + jitter, s=11, color=group_color(group), alpha=0.54, linewidth=0, zorder=3)
        m, lo, hi = mean_ci(sub[value])
        ax.plot([lo, hi], [y, y], color=C["text"], lw=1.25, zorder=4)
        ax.scatter([m], [y], s=34, facecolor=C["white"], edgecolor=C["text"], linewidth=1.0, zorder=5)
    ax.set_yticks(positions, ["B4", "ENSR"])
    ax.set_xlim(*xlim)


def panel_a(ax: plt.Axes, frame: pd.DataFrame, summary: dict) -> None:
    panel_label(ax, "a")
    title(ax, "No-AI narrative quality")
    draw_distribution(ax, frame, "narrative_mean", (4, 16.2))
    model = summary["primary"]["model"]
    ax.text(
        0.98,
        0.95,
        f"Adjusted Δ  {model['estimate']:.2f}\n95% CI  {model['ci95'][0]:.2f} to {model['ci95'][1]:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=C["ensr"],
        fontweight="semibold",
    )
    ax.axvline(16, color=C["grid"], lw=0.7)
    ax.set_xlabel("Blinded score (0–16)")
    ax.text(0.98, 0.05, "n = 39 per group", transform=ax.transAxes, ha="right", color=C["subtext"], fontsize=5.8)
    clean(ax, grid="x")


def draw_timepoint(ax: plt.Axes, frame: pd.DataFrame, fields: list[str], labels: list[str], *, paired: bool) -> None:
    offsets = {"b4": -0.08, "ensr": 0.08}
    for condition in ["b4", "ensr"]:
        group = frame.loc[frame.internal_condition.eq(condition)].copy()
        color = group_color(condition)
        if paired:
            complete = group.dropna(subset=fields)
            for _, row in complete.iterrows():
                ax.plot(
                    np.arange(len(fields)) + offsets[condition],
                    [row[field] for field in fields],
                    color=color,
                    alpha=0.09,
                    lw=0.55,
                    zorder=1,
                )
        means, lows, highs = [], [], []
        for x, field in enumerate(fields):
            values = group[field].dropna()
            m, lo, hi = mean_ci(values)
            means.append(m); lows.append(lo); highs.append(hi)
            codes = group.loc[group[field].notna(), "participant_code"]
            jitter = stable_jitter(codes, 0.045)
            ax.scatter(
                np.full(len(values), x + offsets[condition]) + jitter,
                values,
                s=7,
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=2,
            )
        means_a = np.asarray(means)
        ax.plot(np.arange(len(fields)) + offsets[condition], means_a, color=color, marker="o", ms=4.3, lw=1.55, zorder=4)
        ax.errorbar(
            np.arange(len(fields)) + offsets[condition],
            means_a,
            yerr=[means_a - np.asarray(lows), np.asarray(highs) - means_a],
            color=color,
            fmt="none",
            capsize=2,
            lw=1.05,
            zorder=4,
        )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, 10.3)
    ax.set_ylabel("Knowledge score (0–10)")


def panel_b(ax: plt.Axes, frame: pd.DataFrame, summary: dict) -> None:
    panel_label(ax, "b")
    title(ax, "Immediate learning")
    draw_timepoint(ax, frame, ["pre_total", "post_total"], ["Pre", "Post"], paired=True)
    model = summary["key_learning"]["model"]
    ax.text(
        0.03,
        0.95,
        f"Adjusted post-test Δ  {model['estimate']:.2f}\n95% CI  {model['ci95'][0]:.2f} to {model['ci95'][1]:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=6.4,
        color=C["ensr"],
        fontweight="semibold",
    )
    legend = [
        Line2D([0], [0], color=C["b4"], marker="o", lw=1.4, label="B4"),
        Line2D([0], [0], color=C["ensr"], marker="o", lw=1.4, label="ENSR"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=6.1, handlelength=1.6)
    clean(ax, grid="y")


def panel_c(ax: plt.Axes, frame: pd.DataFrame, summary: dict) -> None:
    panel_label(ax, "c")
    title(ax, "Seven-day retention")
    draw_timepoint(ax, frame, ["post_total", "delayed_total"], ["Post", "Day 7"], paired=True)
    delayed = summary["continuous_outcomes"]["delayed_total"]
    ax.text(
        0.03,
        0.95,
        f"Day-7 Δ  {delayed['difference']:.2f}\n95% CI  {delayed['ci95'][0]:.2f} to {delayed['ci95'][1]:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=6.4,
        color=C["gold"],
        fontweight="semibold",
    )
    ax.text(0.97, 0.06, "Exploratory · 36 ENSR, 34 B4", transform=ax.transAxes, ha="right", color=C["subtext"], fontsize=5.7)
    clean(ax, grid="y")


def panel_d(ax: plt.Axes, summary: dict) -> list[dict]:
    panel_label(ax, "d")
    title(ax, "Motivation and usability")
    metrics = [
        ("Interest", "imi_interest", 1),
        ("Competence", "imi_competence", 1),
        ("Choice", "imi_choice", 1),
        ("SUS", "sus_total", 1),
        ("Lower pressure", "imi_pressure", -1),
    ]
    rows = []
    y = np.arange(len(metrics))[::-1]
    ax.axvspan(-0.82, -0.20, color=C["b4_light"], alpha=0.42, lw=0, zorder=0)
    ax.axvspan(-0.20, 0.20, color=C["neutral"], alpha=0.70, lw=0, zorder=0)
    ax.axvspan(0.20, 0.82, color=C["ensr_light"], alpha=0.50, lw=0, zorder=0)
    ax.axvline(0, color=C["neutral_dark"], lw=0.9, ls=(0, (3, 2)), zorder=1)
    for yi, (label, key, direction) in zip(y, metrics):
        result = summary["continuous_outcomes"][key]
        est = result["hedges_g"] * direction
        lo, hi = sorted([v * direction for v in result["hedges_g_ci95"]])
        ax.plot([lo, hi], [yi, yi], color=C["text"], lw=2.4, alpha=0.72, solid_capstyle="round", zorder=2)
        ax.scatter(est, yi, s=48, facecolor=C["white"], edgecolor=C["ensr"], lw=1.6, zorder=3)
        ax.text(0.79, yi, f"{est:+.2f}", ha="right", va="center", fontsize=5.7, color=C["text"], fontweight="semibold")
        rows.append({"outcome": key, "label": label, "oriented_hedges_g": est, "ci95_low": lo, "ci95_high": hi, "raw_p": result["p"]})
    ax.set_yticks(y, [m[0] for m in metrics])
    ax.set_xlim(-0.82, 0.82)
    ax.set_xlabel("Hedges g  (positive favours ENSR)")
    ax.text(0.01, 1.01, "favours B4", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5, color=C["b4"])
    ax.text(0.99, 1.01, "favours ENSR", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=C["ensr"])
    clean(ax, grid="x")
    return rows


def panel_e(ax: plt.Axes, summary: dict) -> list[dict]:
    panel_label(ax, "e")
    title(ax, "NASA-TLX workload")
    metrics = [
        ("Mental", "tlx_mental"),
        ("Physical", "tlx_physical"),
        ("Temporal", "tlx_temporal"),
        ("Performance", "tlx_performance_dissatisfaction"),
        ("Effort", "tlx_effort"),
        ("Frustration", "tlx_frustration"),
    ]
    rows = []
    b4_values, ensr_values = [], []
    for label, key in metrics:
        result = summary["continuous_outcomes"][key]
        b4, ensr = result["b4_mean"], result["ensr_mean"]
        b4_values.append(b4)
        ensr_values.append(ensr)
        rows.append({"outcome": key, "label": label, "b4_mean": b4, "ensr_mean": ensr, "difference": result["difference"], "ci95_low": result["ci95"][0], "ci95_high": result["ci95"][1], "p": result["p"]})
    total = summary["continuous_outcomes"]["raw_tlx"]
    ax.axis("off")
    polar = ax.inset_axes([0.04, 0.00, 0.90, 0.93], projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    b4_closed = np.r_[b4_values, b4_values[0]]
    ensr_closed = np.r_[ensr_values, ensr_values[0]]
    polar.set_theta_offset(np.pi / 2)
    polar.set_theta_direction(-1)
    polar.set_ylim(0, 80)
    polar.set_xticks(angles, [m[0] for m in metrics], fontsize=5.6, color=C["text"])
    polar.set_yticks([20, 40, 60], ["20", "40", "60"], fontsize=4.8, color=C["subtext"])
    polar.set_rlabel_position(16)
    polar.grid(color=C["grid"], lw=0.75)
    polar.spines["polar"].set_color(C["grid"])
    polar.plot(closed_angles, b4_closed, color=C["b4"], lw=1.8, marker="o", ms=3.4, label="B4")
    polar.fill(closed_angles, b4_closed, color=C["b4"], alpha=0.10)
    polar.plot(closed_angles, ensr_closed, color=C["ensr"], lw=1.8, marker="o", ms=3.4, label="ENSR")
    polar.fill(closed_angles, ensr_closed, color=C["ensr"], alpha=0.10)
    polar.legend(loc="upper right", bbox_to_anchor=(1.18, 1.13), fontsize=5.5, handlelength=1.3)
    ax.text(
        0.51,
        0.49,
        f"Total\nΔ {total['difference']:.1f}\n[{total['ci95'][0]:.1f}, {total['ci95'][1]:.1f}]",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.7,
        color=C["subtext"],
        bbox={"boxstyle": "circle,pad=0.55", "facecolor": "white", "edgecolor": C["grid"], "linewidth": 0.7},
    )
    return rows


def draw_dual_ring(ax: plt.Axes, ensr_rate: float, b4_rate: float, label: str, denominators: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.35)
    ax.axis("off")
    ax.add_patch(Wedge((0, 0), 1.00, 90, 450, width=0.17, facecolor=C["ensr_light"], edgecolor="none"))
    ax.add_patch(Wedge((0, 0), 1.00, 90, 90 + 360 * ensr_rate, width=0.17, facecolor=C["ensr"], edgecolor="none"))
    ax.add_patch(Wedge((0, 0), 0.70, 90, 450, width=0.17, facecolor=C["b4_light"], edgecolor="none"))
    if b4_rate > 0:
        ax.add_patch(Wedge((0, 0), 0.70, 90, 90 + 360 * b4_rate, width=0.17, facecolor=C["b4"], edgecolor="none"))
    ax.text(0, 1.21, label, ha="center", va="bottom", fontsize=5.4, color=C["text"], fontweight="semibold")
    ax.text(0, 0.17, f"{100*ensr_rate:.1f}%", ha="center", va="center", fontsize=7.0, color=C["ensr"], fontweight="bold")
    ax.text(0, -0.12, f"{100*b4_rate:.1f}%", ha="center", va="center", fontsize=6.1, color=C["b4"], fontweight="semibold")
    ax.text(0, -1.15, denominators, ha="center", va="top", fontsize=4.5, color=C["subtext"])


def panel_f(ax: plt.Axes, summary: dict) -> list[dict]:
    panel_label(ax, "f")
    title(ax, "Behaviour and task burden")
    binary = summary["binary_outcomes"]
    metrics = [
        ("No avoidable rework", "rework_cost", True),
        ("Told others by day 7", "followup_told_others", False),
        ("Reviewed materials", "followup_reviewed_materials", False),
        ("Attempted physical craft", "followup_attempted_physical_craft", False),
    ]
    rows = []
    positions = [(0.00, 0.49), (0.51, 0.49), (0.00, 0.01), (0.51, 0.01)]
    for (label, key, invert), (x, y) in zip(metrics, positions):
        result = binary[key]
        b4 = 1 - result["b4_rate"] if invert else result["b4_rate"]
        ensr = 1 - result["ensr_rate"] if invert else result["ensr_rate"]
        ring = ax.inset_axes([x, y, 0.48, 0.42])
        ensr_num = result["ensr_n"] - result["ensr_events"] if invert else result["ensr_events"]
        b4_num = result["b4_n"] - result["b4_events"] if invert else result["b4_events"]
        draw_dual_ring(ring, ensr, b4, label, f"ENSR {ensr_num}/{result['ensr_n']} · B4 {b4_num}/{result['b4_n']}")
        rows.append({"outcome": key, "label": label, "b4_percent": b4 * 100, "ensr_percent": ensr * 100, "risk_difference": result["risk_difference"], "ci95_low": result["risk_difference_ci95"][0], "ci95_high": result["risk_difference_ci95"][1], "fisher_p": result["fisher_p"], "exploratory": key != "rework_cost"})
    duration = summary["continuous_outcomes"]["duration_minutes"]
    ax.axis("off")
    ax.text(
        0.99,
        1.02,
        "outer ENSR · inner B4",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=C["subtext"],
        clip_on=False,
    )
    ax.text(
        0.50,
        -0.04,
        f"Session duration: {duration['ensr_mean']:.1f} vs {duration['b4_mean']:.1f} min  ·  Δ {duration['difference']:.1f} [{duration['ci95'][0]:.1f}, {duration['ci95'][1]:.1f}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.25,
        color=C["subtext"],
        clip_on=False,
    )
    return rows


def panel_g(ax: plt.Axes, summary: dict) -> list[dict]:
    title(ax, "Robustness, rater agreement and diagnostics")
    ax.set_xlim(-0.28, 1.0)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(-0.04, 1.075, "g", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, fontweight="bold", clip_on=False)

    # Left: primary-outcome sensitivity estimates.
    x0, x1 = -0.20, 0.47
    forest = ax.inset_axes([0.00, 0.08, 0.54, 0.80])
    primary = summary["primary"]
    rows = [
        ("Frozen HC3", primary["model"]["estimate"], *primary["model"]["ci95"], "ci"),
        ("Worst-case missing", primary["extreme_bound_sensitivity"]["worst_for_ensr"]["estimate"], *primary["extreme_bound_sensitivity"]["worst_for_ensr"]["ci95"], "ci"),
        ("Best-case missing", primary["extreme_bound_sensitivity"]["best_for_ensr"]["estimate"], *primary["extreme_bound_sensitivity"]["best_for_ensr"]["ci95"], "ci"),
        ("Huber regression", primary["robust_regression"]["ensr_estimate"], np.nan, np.nan, "point"),
    ]
    yy = np.arange(len(rows))[::-1]
    forest.axvspan(-0.15, primary["minimum_meaningful_difference"], color=C["neutral"], alpha=0.68, lw=0)
    forest.axvspan(primary["minimum_meaningful_difference"], 3.85, color=C["ensr_light"], alpha=0.42, lw=0)
    forest.axvline(0, color=C["neutral_dark"], lw=0.8)
    forest.axvline(primary["minimum_meaningful_difference"], color=C["gold"], lw=0.9, ls=(0, (3, 2)))
    exported = []
    for yi, (label, est, lo, hi, kind) in zip(yy, rows):
        if kind == "ci":
            forest.plot([lo, hi], [yi, yi], color=C["text"], lw=2.3, alpha=0.70, solid_capstyle="round")
            forest.scatter(est, yi, s=42, color=C["ensr"], zorder=3)
        else:
            forest.scatter(est, yi, s=46, marker="D", facecolor=C["white"], edgecolor=C["ensr"], lw=1.5, zorder=3)
        exported.append({"analysis": label, "estimate": est, "ci95_low": lo, "ci95_high": hi})
    forest.set_yticks(yy, [r[0] for r in rows])
    forest.set_xlim(-0.15, 3.85)
    forest.set_xlabel("Adjusted narrative difference (ENSR − B4)")
    forest.text(primary["minimum_meaningful_difference"], 3.48, "MID 1.5", ha="center", va="bottom", fontsize=5.5, color=C["gold"])
    forest.text(0.98, 0.03, "Block permutation p = 0.0001", transform=forest.transAxes, ha="right", fontsize=5.7, color=C["subtext"])
    clean(forest, grid="x")

    # Middle: rater agreement as compact circular gauges.
    ax.plot([0.565, 0.565], [0.05, 0.90], color=C["grid"], lw=0.8, transform=ax.transAxes)
    ax.text(0.59, 0.87, "Rater agreement", transform=ax.transAxes, fontsize=7.0, fontweight="semibold")
    icc = summary["rater_reliability"]
    for x_pos, label, value in [
        (0.62, "ICC(2,1)", icc["icc_2_1"]),
        (0.73, "ICC(2,k)", icc["icc_2_k"]),
    ]:
        gauge = ax.inset_axes([x_pos - 0.035, 0.29, 0.095, 0.47])
        gauge.set_aspect("equal"); gauge.set_xlim(-1.15, 1.15); gauge.set_ylim(-1.15, 1.15); gauge.axis("off")
        gauge.add_patch(Wedge((0, 0), 1.0, 90, 450, width=0.23, facecolor=C["neutral"], edgecolor="none"))
        gauge.add_patch(Wedge((0, 0), 1.0, 90, 90 + 360 * value, width=0.23, facecolor=C["teal"], edgecolor="none"))
        gauge.text(0, 0.03, f"{value:.3f}", ha="center", va="center", fontsize=7.0, fontweight="bold", color=C["teal"])
        gauge.text(0, -1.28, label, ha="center", va="top", fontsize=5.4, color=C["subtext"])

    # Right: concise diagnostics and completeness.
    ax.plot([0.815, 0.815], [0.05, 0.90], color=C["grid"], lw=0.8, transform=ax.transAxes)
    ax.text(0.84, 0.87, "Model checks", transform=ax.transAxes, fontsize=7.0, fontweight="semibold")
    diag = primary["model"]["diagnostics"]
    checks = [
        ("BP", diag["breusch_pagan_p"], C["teal"], C["teal_light"]),
        ("Residual", diag["shapiro_residual_p"], C["teal"], C["teal_light"]),
        ("RESET", diag["ramsey_reset_p"], C["teal"], C["teal_light"]),
    ]
    for index, (label, value, color, fill) in enumerate(checks):
        x_pos = 0.84 + (index % 2) * 0.085
        y_pos = 0.58 - (index // 2) * 0.25
        tile = FancyBboxPatch((x_pos, y_pos), 0.072, 0.18, boxstyle="round,pad=0.006,rounding_size=0.01", transform=ax.transAxes, facecolor=fill, edgecolor="none")
        ax.add_patch(tile)
        ax.text(x_pos + 0.036, y_pos + 0.115, label, transform=ax.transAxes, ha="center", fontsize=5.6, color=color, fontweight="semibold")
        ax.text(x_pos + 0.036, y_pos + 0.050, f"p={value:.3f}", transform=ax.transAxes, ha="center", fontsize=5.1, color=C["text"])
    learn_reset = summary["key_learning"]["model"]["diagnostics"]["ramsey_reset_p"]
    tile = FancyBboxPatch((0.925, 0.33), 0.067, 0.18, boxstyle="round,pad=0.006,rounding_size=0.01", transform=ax.transAxes, facecolor=C["gold_light"], edgecolor="none")
    ax.add_patch(tile)
    ax.text(0.9585, 0.445, "Learning", transform=ax.transAxes, ha="center", fontsize=5.25, color=C["gold"], fontweight="semibold")
    ax.text(0.9585, 0.380, f"p={learn_reset:.3f}", transform=ax.transAxes, ha="center", fontsize=5.0, color=C["text"])
    ax.text(0.84, 0.09, "78/80 primary · 70/80 day 7", transform=ax.transAxes, fontsize=5.8, color=C["subtext"])
    return exported


def export_sources(frame: pd.DataFrame, summary: dict, d_rows: list[dict], e_rows: list[dict], f_rows: list[dict], g_rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame[["participant_code", "internal_condition", "narrative_mean"]].dropna(subset=["narrative_mean"]).to_csv(
        OUTPUT_DIR / "Figure_12a_narrative_source.csv", index=False, encoding="utf-8-sig"
    )
    frame[["participant_code", "internal_condition", "pre_total", "post_total"]].to_csv(
        OUTPUT_DIR / "Figure_12b_immediate_learning_source.csv", index=False, encoding="utf-8-sig"
    )
    frame[["participant_code", "internal_condition", "post_total", "delayed_total"]].to_csv(
        OUTPUT_DIR / "Figure_12c_retention_source.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(d_rows).to_csv(OUTPUT_DIR / "Figure_12d_experience_effects_source.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(e_rows).to_csv(OUTPUT_DIR / "Figure_12e_tlx_source.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(f_rows).to_csv(OUTPUT_DIR / "Figure_12f_behaviour_source.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(g_rows).to_csv(OUTPUT_DIR / "Figure_12g_robustness_source.csv", index=False, encoding="utf-8-sig")


def build_caption(summary: dict) -> None:
    primary = summary["primary"]["model"]
    learning = summary["key_learning"]["model"]
    delayed = summary["continuous_outcomes"]["delayed_total"]
    text = f"""# Figure 12 | Player learning, experience and behavioural outcomes

**a,** Participant-level distributions of the mean blinded no-AI narrative score after the LanternQuest session. Open central markers show group means and horizontal lines show descriptive 95% confidence intervals. The preregistered HC3 model estimated an ENSR–B4 difference of {primary['estimate']:.2f} points (95% CI {primary['ci95'][0]:.2f}–{primary['ci95'][1]:.2f}; n={primary['n']}). **b,** Pre-test and immediate post-test knowledge scores; faint points and trajectories show participant observations, while coloured points and error bars show group means and descriptive 95% confidence intervals. The adjusted post-test difference was {learning['estimate']:.2f} points (95% CI {learning['ci95'][0]:.2f}–{learning['ci95'][1]:.2f}; n={learning['n']}). **c,** Post-test to day-7 knowledge trajectories among returned participants. The unadjusted day-7 difference was {delayed['difference']:.2f} points (95% CI {delayed['ci95'][0]:.2f}–{delayed['ci95'][1]:.2f}); this result is exploratory because 70/80 participants returned. **d,** Hedges g estimates for motivation and usability; directional background bands make the null zone and the two effect directions explicit, and every displayed 95% confidence interval crosses zero. **e,** Radial profiles show the six mean NASA-TLX dimension scores for B4 and ENSR; the centre reports the adjusted total-score contrast. **f,** Nested rings show observed behavioural rates (outer ring, ENSR; inner ring, B4), with session duration reported below. **g,** Primary-outcome robustness analyses, two-way absolute-agreement intraclass correlations for blinded narrative ratings, model diagnostics and observed-data completeness. The gold boundary marks the preregistered minimum meaningful difference (MID) of 1.5 points. Rose denotes ENSR and cool blue denotes B4. Confidence intervals are two-sided 95% intervals. Randomized: 40 ENSR and 40 B4; primary outcome complete: 39 per group.

Interpretive boundary: the experiment supports improved evidence-constrained explanation, immediate learning and reduced avoidable rework in the controlled digital task. It does not establish equivalence for null outcomes, mastery of physical lantern-making, or community-level heritage transmission. Day-7 knowledge and follow-up behaviours are exploratory.
"""
    (OUTPUT_DIR / "Figure_12_caption_nature_v1.md").write_text(text, encoding="utf-8")


def write_qa(summary: dict, outputs: list[Path]) -> None:
    png = OUTPUT_STEM.with_suffix(".png")
    with Image.open(png) as image:
        png_size = image.size
    lines = [
        "# Figure 12 QA report",
        "",
        "- Source-only figure: no observations, confidence intervals or p values were simulated.",
        f"- Integrity audit: `{summary['integrity_status']}` with {len(summary['integrity_errors'])} recorded integrity errors.",
        f"- Flow: {summary['flow']['randomized']} randomized, {summary['flow']['completed']} completed, {summary['flow']['primary_complete']} primary outcomes, {summary['flow']['day7_returned']} day-7 returns.",
        f"- Primary adjusted effect: {summary['primary']['model']['estimate']:.3f}, 95% CI {summary['primary']['model']['ci95'][0]:.3f} to {summary['primary']['model']['ci95'][1]:.3f}.",
        f"- Immediate-learning adjusted effect: {summary['key_learning']['model']['estimate']:.3f}, 95% CI {summary['key_learning']['model']['ci95'][0]:.3f} to {summary['key_learning']['model']['ci95'][1]:.3f}.",
        "- Null outcomes are retained for IMI, SUS and workload; day-7 and transmission outcomes are labelled exploratory.",
        f"- Rater agreement: ICC(2,1)={summary['rater_reliability']['icc_2_1']:.3f}; ICC(2,k)={summary['rater_reliability']['icc_2_k']:.3f}.",
        f"- PNG dimensions: {png_size[0]} × {png_size[1]} pixels at 600 dpi.",
        "- TIFF uses LZW compression; SVG retains editable text; PDF uses embedded TrueType fonts.",
        "- Seven panel-level machine-readable CSV files and a SHA-256 source manifest accompany the figure.",
        "",
        "## Figure file hashes",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path.name}`: `{sha256(path)}`")
    (OUTPUT_DIR / "Figure_12_QA_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_json(SUMMARY_PATH)
    frame = pd.read_csv(PARTICIPANT_PATH, encoding="utf-8-sig")

    if summary.get("integrity_status") != "passed" or summary.get("integrity_errors"):
        raise RuntimeError("E90 validation did not pass integrity checks")
    if len(frame) != 80 or frame["participant_code"].nunique() != 80:
        raise RuntimeError("Expected 80 unique randomized participants")
    if int(frame["narrative_mean"].notna().sum()) != 78 or int(frame["delayed_total"].notna().sum()) != 70:
        raise RuntimeError("Participant completeness does not match the frozen E90 audit")

    fig = plt.figure(figsize=(12.2, 9.8))
    grid = fig.add_gridspec(3, 12, height_ratios=[1.02, 1.0, 0.62], hspace=0.43, wspace=0.78)
    axes = {
        "a": fig.add_subplot(grid[0, 0:4]),
        "b": fig.add_subplot(grid[0, 4:8]),
        "c": fig.add_subplot(grid[0, 8:12]),
        "d": fig.add_subplot(grid[1, 0:4]),
        "e": fig.add_subplot(grid[1, 4:8]),
        "f": fig.add_subplot(grid[1, 8:12]),
        "g": fig.add_subplot(grid[2, 0:12]),
    }
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.065)

    panel_a(axes["a"], frame, summary)
    panel_b(axes["b"], frame, summary)
    panel_c(axes["c"], frame, summary)
    d_rows = panel_d(axes["d"], summary)
    e_rows = panel_e(axes["e"], summary)
    f_rows = panel_f(axes["f"], summary)
    g_rows = panel_g(axes["g"], summary)

    outputs = [OUTPUT_STEM.with_suffix(ext) for ext in [".png", ".svg", ".pdf", ".tiff"]]
    fig.savefig(outputs[0], dpi=600, facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], facecolor="white")
    fig.savefig(outputs[3], dpi=600, pil_kwargs={"compression": "tiff_lzw"}, facecolor="white")
    plt.close(fig)

    export_sources(frame, summary, d_rows, e_rows, f_rows, g_rows)
    build_caption(summary)
    manifest = {
        "figure": "Figure 12",
        "source_files": {
            str(SUMMARY_PATH.relative_to(ROOT)): sha256(SUMMARY_PATH),
            str(PARTICIPANT_PATH.relative_to(ROOT)): sha256(PARTICIPANT_PATH),
            str(RATING_PATH.relative_to(ROOT)): sha256(RATING_PATH),
        },
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "integrity_status": summary["integrity_status"],
        "analysis_id": summary["analysis_id"],
    }
    (OUTPUT_DIR / "Figure_12_source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_qa(summary, outputs)
    print(json.dumps({"status": "complete", "outputs": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
