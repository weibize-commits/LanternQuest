"""Render Figure 8 from frozen efficiency, adapter, and provider audits.

The six panels separate algorithmic resource use from deployment boundaries:
  a) paired performance-resource trajectories across ten internal settings,
  b) relative resource use (full ENSR / ENSR-base),
  c) adapter-error burden before and after frozen adapter changes,
  d) external-provider directional effects with paired-bootstrap 95% CIs,
  e) external-provider outcome composition and estimated API-cost ranges,
  f) the prospective freeze sequence for E15 and E15R.

No observations, uncertainty intervals, or costs are simulated. Currency ranges
are the frozen low/high estimates in the E17 cost audit and are not invoices.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENCEWORLD_DIR = REPO_ROOT / "artifacts" / "benchmarks" / "scienceworld"
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "paper" / "figures" / "figure8"
OUTPUT_STEM = OUTPUT_DIR / "Figure_8_efficiency_service_boundaries_nature_v1"

EFFICIENCY_JSON = SCIENCEWORLD_DIR / "ensr_efficiency_summary_v1.json"
PAPER_SUMMARY_JSON = SCIENCEWORLD_DIR / "paper_results_summary_v1.json"
E15_JSON = SCIENCEWORLD_DIR / "ensr_clean_adapter_v1_audit.json"
E15R_JSON = SCIENCEWORLD_DIR / "qwen3_8b_recovery_v1_audit.json"
E17_JSON = SCIENCEWORLD_DIR / "external_provider_v1_audit.json"
E17_COST_JSON = SCIENCEWORLD_DIR / "external_provider_v1_cost_summary.json"
CLEAN_CONFIG_JSON = CONFIG_DIR / "scienceworld_clean_adapter_profiles_v1.json"
RECOVERY_CONFIG_JSON = CONFIG_DIR / "scienceworld_qwen3_8b_recovery_profiles_v1.json"


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
    "gold": "#C7954A",
    "green": "#4F806B",
    "green_light": "#DDEDE6",
    "failure": "#6576A5",
    "failure_light": "#DDE2EF",
}

EXPERIMENT_MARKERS = {"E13": "o", "E14": "s", "E15": "^", "E15R": "D"}

PROFILE_LABELS = {
    ("E13", "bit_qwen3_235b"): "E13",
    ("E14", "bit_qwen3_235b"): "235B",
    ("E14", "bit_ceep_70b"): "CEEP",
    ("E14", "bit_qwen3_32b"): "32B",
    ("E14", "bit_qwen3_8b"): "8B",
    ("E14", "bit_qwen3_0_6b"): "0.6B",
    ("E15", "bit_ceep_70b_clean"): "C-70B",
    ("E15", "bit_qwen3_32b_clean"): "C-32B",
    ("E15", "bit_qwen3_8b_clean"): "C-8B",
    ("E15R", "bit_qwen3_8b_recovery_v1"): "R-8B",
}

LABEL_OFFSETS = {
    "E13": (5, -7),
    "235B": (5, 7),
    "CEEP": (5, 0),
    "32B": (5, 1),
    "8B": (5, -8),
    "0.6B": (5, -1),
    "C-70B": (5, 2),
    "C-32B": (5, -8),
    "C-8B": (5, -8),
    "R-8B": (5, 2),
}

PROVIDER_LABELS = {
    "deepseek_v4_pro_external": "DeepSeek",
    "glm_5_3_external": "GLM",
    "gpt_6_astra_external": "GPT",
    "kimi_k2_6_external": "Kimi",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.07) -> None:
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_efficiency(efficiency: dict) -> tuple[list[dict], list[dict]]:
    allowed = {"E13", "E14", "E15", "E15R"}
    rows = [
        dict(row)
        for row in efficiency["rows"]
        if row["experiment"] in allowed and row["method"] in {"iper_rag", "ensr_v2"}
    ]
    pairs: list[dict] = []
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["experiment"], row["profile"]), {})[row["method"]] = row
    if len(grouped) != 10:
        raise RuntimeError(f"Expected ten paired efficiency settings, found {len(grouped)}.")
    for key, methods in grouped.items():
        if set(methods) != {"iper_rag", "ensr_v2"}:
            raise RuntimeError(f"Unpaired efficiency setting: {key}")
        experiment, profile = key
        iper = methods["iper_rag"]
        ensr = methods["ensr_v2"]
        pairs.append(
            {
                "experiment": experiment,
                "profile": profile,
                "label": PROFILE_LABELS[key],
                "iper_tokens_per_episode": iper["total_tokens"] / iper["episodes"],
                "ensr_tokens_per_episode": ensr["total_tokens"] / ensr["episodes"],
                "iper_success_rate": 100.0 * iper["successes"] / iper["episodes"],
                "ensr_success_rate": 100.0 * ensr["successes"] / ensr["episodes"],
                "iper_mean_score": iper["mean_clipped_score"],
                "ensr_mean_score": ensr["mean_clipped_score"],
                "iper_model_calls": iper["model_calls"],
                "ensr_model_calls": ensr["model_calls"],
                "iper_retrieval_calls": iper["retrieval_calls"],
                "ensr_retrieval_calls": ensr["retrieval_calls"],
                "iper_total_tokens": iper["total_tokens"],
                "ensr_total_tokens": ensr["total_tokens"],
                "model_call_ratio": ensr["model_calls"] / iper["model_calls"],
                "token_ratio": ensr["total_tokens"] / iper["total_tokens"],
                "retrieval_ratio": ensr["retrieval_calls"] / iper["retrieval_calls"],
                "additional_successes": ensr["successes"] - iper["successes"],
            }
        )
    order = {"E13": 0, "E14": 1, "E15": 2, "E15R": 3}
    pairs.sort(key=lambda row: (order[row["experiment"]], row["label"]))
    return rows, pairs


def extract_adapter_rows(summary: dict, e15: dict, e15r: dict) -> list[dict]:
    rows: list[dict] = []
    source_specs = [
        ("E14", "CEEP–70B", summary["e14"]["models"]["bit_ceep_70b"], "original"),
        ("E15", "CEEP–70B", e15["models"]["bit_ceep_70b_clean"], "clean"),
        ("E14", "Qwen3–32B", summary["e14"]["models"]["bit_qwen3_32b"], "original"),
        ("E15", "Qwen3–32B", e15["models"]["bit_qwen3_32b_clean"], "clean"),
        ("E14", "Qwen3–8B", summary["e14"]["models"]["bit_qwen3_8b"], "original"),
        ("E15", "Qwen3–8B", e15["models"]["bit_qwen3_8b_clean"], "clean"),
        ("E15R", "Qwen3–8B", e15r, "recovery"),
    ]
    for experiment, model, audit, stage in source_specs:
        summaries = audit["summaries"]
        if experiment == "E14":
            iper_errors = summaries["iper_rag"]["provider_or_environment_failures"]
            ensr_errors = summaries["ensr_v2"]["provider_or_environment_failures"]
        else:
            iper_errors = summaries["iper_rag"]["external_failures"]
            ensr_errors = summaries["ensr_v2"]["external_failures"]
        operational_pass = iper_errors <= 1 and ensr_errors <= 1
        comparison_key = "ensr_minus_baseline" if experiment == "E14" else "ensr_minus_iper"
        delta = audit[comparison_key]["clipped_score"]
        rows.append(
            {
                "experiment": experiment,
                "model": model,
                "stage": stage,
                "label": f"{model}  {stage}",
                "iper_external_failures": iper_errors,
                "ensr_external_failures": ensr_errors,
                "operational_threshold_each_cell": 1,
                "operational_pass": operational_pass,
                "score_delta": delta["point_estimate"],
                "score_ci95_low": delta["ci95_low"],
                "score_ci95_high": delta["ci95_high"],
            }
        )
    return rows


def extract_provider_rows(e17: dict, costs: dict) -> tuple[list[dict], list[dict]]:
    effect_rows: list[dict] = []
    status_rows: list[dict] = []
    for profile, audit in e17["profiles"].items():
        label = PROVIDER_LABELS[profile]
        score = audit["ensr_minus_iper"]["clipped_score"]
        success = audit["ensr_minus_iper"]["task_success_rate"]
        effect_rows.append(
            {
                "profile": profile,
                "provider": label,
                "score_delta": score["point_estimate"],
                "score_ci95_low": score["ci95_low"],
                "score_ci95_high": score["ci95_high"],
                "success_delta_pp": 100.0 * success["point_estimate"],
                "success_ci95_low_pp": 100.0 * success["ci95_low"],
                "success_ci95_high_pp": 100.0 * success["ci95_high"],
                "directional_pass": audit["directional_pass"],
                "strong_success": audit["strong_success"],
                "cost_low_cny": costs["profiles"][profile]["estimated_low_cny"],
                "cost_high_cny": costs["profiles"][profile]["estimated_high_cny"],
            }
        )
        for method in ["iper_rag", "ensr_v2"]:
            counts = audit["summaries"][method]["status_counts"]
            standardized = {
                "terminal_success": counts.get("terminal_success", 0),
                "terminal_failure": counts.get("terminal_failure", 0),
                "budget_exhausted": counts.get("model_budget_exhausted", 0)
                + counts.get("environment_budget_exhausted", 0),
                "plan_exhausted": counts.get("plan_exhausted", 0),
                "model_stop": counts.get("model_stop", 0),
                "invalid_decision": counts.get("invalid_decision_limit", 0),
                "external_error": counts.get("adapter_error", 0) + counts.get("environment_error", 0),
            }
            if sum(standardized.values()) != 30:
                raise RuntimeError(f"Unexpected E17 status total for {profile}/{method}: {standardized}")
            status_rows.append(
                {
                    "profile": profile,
                    "provider": label,
                    "method": method,
                    **standardized,
                    "episodes": 30,
                }
            )
    return effect_rows, status_rows


def draw_panel_a(ax: plt.Axes, pairs: list[dict]) -> None:
    ax.set_title("Performance–resource trajectories", loc="left", pad=6)
    panel_label(ax, "a")
    ax.set_xscale("log")
    ax.set_xlim(4.5e3, 5.8e5)
    ax.set_ylim(-2, 55)
    ax.grid(axis="both", which="major", color=COLORS["grid"], linewidth=0.7, zorder=0)

    for row in pairs:
        x0, y0 = row["iper_tokens_per_episode"], row["iper_success_rate"]
        x1, y1 = row["ensr_tokens_per_episode"], row["ensr_success_rate"]
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.75,
            color=COLORS["neutral_dark"],
            alpha=0.68,
            zorder=1,
        )
        ax.add_patch(arrow)
        size0 = 18 + 0.55 * row["iper_mean_score"]
        size1 = 18 + 0.55 * row["ensr_mean_score"]
        marker = EXPERIMENT_MARKERS[row["experiment"]]
        ax.scatter(
            x0,
            y0,
            s=size0,
            marker=marker,
            facecolor=COLORS["white"],
            edgecolor=COLORS["iper"],
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            x1,
            y1,
            s=size1,
            marker=marker,
            facecolor=COLORS["ensr"],
            edgecolor=COLORS["white"],
            linewidth=0.55,
            zorder=4,
        )
        dx, dy = LABEL_OFFSETS[row["label"]]
        ax.annotate(
            row["label"],
            (x1, y1),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.4,
            color=COLORS["text"],
            ha="left",
            va="center",
        )

    ax.set_xlabel("Total tokens per episode (log scale)")
    ax.set_ylabel("Task success rate (%)")
    ax.xaxis.set_major_locator(FixedLocator([1e4, 3e4, 1e5, 3e5]))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: {1e4: "10k", 3e4: "30k", 1e5: "100k", 3e5: "300k"}.get(value, ""))
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    clean_axis(ax)
    method_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["iper"], label="ENSR-base"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["ensr"], markeredgecolor="white", label="ENSR"),
    ]
    shape_handles = [
        Line2D([], [], marker=EXPERIMENT_MARKERS[key], linestyle="none", color=COLORS["subtext"], label=key)
        for key in ["E13", "E14", "E15", "E15R"]
    ]
    legend1 = ax.legend(handles=method_handles, loc="upper right", ncol=2, fontsize=5.6, handletextpad=0.35, columnspacing=0.8)
    ax.add_artist(legend1)
    ax.legend(handles=shape_handles, loc="lower right", ncol=4, fontsize=5.2, handletextpad=0.1, columnspacing=0.55)
    ax.text(0.01, 0.98, "Arrow: ENSR-base → full ENSR; point area: mean score", transform=ax.transAxes, fontsize=5.3, color=COLORS["subtext"], va="top")


def draw_panel_b(ax: plt.Axes, pairs: list[dict]) -> None:
    ax.set_title("Relative resource use", loc="left", pad=6)
    panel_label(ax, "b", x=-0.17)
    x = np.arange(3)
    metrics = ["model_call_ratio", "token_ratio", "retrieval_ratio"]
    for row in pairs:
        y = [row[key] for key in metrics]
        ax.plot(x, y, color=COLORS["ensr"], alpha=0.25, linewidth=0.8, zorder=1)
        ax.scatter(x, y, s=12, facecolor=COLORS["white"], edgecolor=COLORS["ensr"], linewidth=0.65, alpha=0.75, zorder=2)
    medians = np.array([[row[key] for key in metrics] for row in pairs], dtype=float)
    median = np.median(medians, axis=0)
    ax.plot(x, median, color=COLORS["text"], linewidth=1.8, zorder=4)
    ax.scatter(x, median, s=28, color=COLORS["text"], edgecolor=COLORS["white"], linewidth=0.55, zorder=5)
    for xi, value in zip(x, median):
        offset = (0, -12) if xi == 2 else (0, 6)
        ax.annotate(f"{value:.2f}×", (xi, value), xytext=offset, textcoords="offset points", ha="center", fontsize=5.8, fontweight="semibold")
    ax.axhline(1.0, color=COLORS["subtext"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_ylim(0.015, 7.5)
    ax.set_xticks(x, ["Model calls", "Tokens", "Retrievals"])
    ax.set_ylabel("Full ENSR / ENSR-base")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.7)
    clean_axis(ax)
    ax.text(0.02, 0.04, "<1 uses fewer resources", transform=ax.transAxes, fontsize=5.4, color=COLORS["subtext"])
    ax.text(0.02, 0.96, "thin lines: 10 paired settings\nthick line: median", transform=ax.transAxes, fontsize=5.3, color=COLORS["subtext"], ha="left", va="top")


def draw_panel_c(ax: plt.Axes, rows: list[dict]) -> None:
    ax.set_title("Adapter errors after prospective profile changes", loc="left", pad=6)
    panel_label(ax, "c")
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        if row["stage"] in {"clean", "recovery"}:
            ax.axhspan(yi - 0.43, yi + 0.43, color=COLORS["ensr_light"], alpha=0.28, zorder=0)
        ax.plot(
            [row["iper_external_failures"], row["ensr_external_failures"]],
            [yi, yi],
            color=COLORS["neutral_dark"],
            linewidth=1.0,
            zorder=1,
        )
        ax.scatter(row["iper_external_failures"], yi, marker="s", s=31, facecolor="white", edgecolor=COLORS["iper"], linewidth=1.0, zorder=3)
        ax.scatter(row["ensr_external_failures"], yi, marker="o", s=34, facecolor=COLORS["ensr"], edgecolor="white", linewidth=0.55, zorder=4)
        symbol = "PASS" if row["operational_pass"] else "FAIL"
        color = COLORS["green"] if row["operational_pass"] else COLORS["iper"]
        ax.text(28.4, yi, symbol, fontsize=5.5, fontweight="bold", ha="center", va="center", color=color)
    ax.axvline(1, color=COLORS["subtext"], linewidth=0.75, linestyle=(0, (2, 2)))
    ax.text(1.2, y[0] + 0.62, "≤1 per method", fontsize=5.2, color=COLORS["subtext"], va="bottom")
    ax.set_xlim(-1, 31.2)
    ax.set_ylim(-0.8, len(rows) - 0.15)
    ax.set_yticks(y, [row["label"] for row in rows])
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("Adapter/environment failures per 30 episodes")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    clean_axis(ax)
    ax.text(28.4, y[0] + 0.62, "Operational gate", fontsize=5.2, color=COLORS["subtext"], ha="center", va="bottom")
    ax.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["iper"], label="ENSR-base"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["ensr"], markeredgecolor="white", label="ENSR"),
        ],
        loc="lower right",
        ncol=2,
        fontsize=5.4,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def draw_panel_d(ax: plt.Axes, rows: list[dict]) -> None:
    ax.set_title("External-provider directional effects", loc="left", pad=6)
    panel_label(ax, "d", x=-0.17)
    ax.axhline(0, color=COLORS["subtext"], linewidth=0.75, linestyle=(0, (3, 2)), zorder=0)
    ax.axvline(0, color=COLORS["subtext"], linewidth=0.75, linestyle=(0, (3, 2)), zorder=0)
    offsets = {"DeepSeek": (5, 5), "GLM": (5, -9), "GPT": (5, 4), "Kimi": (5, 5)}
    for row in rows:
        x = row["score_delta"]
        y = row["success_delta_pp"]
        xerr = np.array([[x - row["score_ci95_low"]], [row["score_ci95_high"] - x]])
        yerr = np.array([[y - row["success_ci95_low_pp"]], [row["success_ci95_high_pp"] - y]])
        pass_flag = row["directional_pass"]
        color = COLORS["ensr"] if pass_flag else COLORS["iper"]
        marker_face = color if pass_flag else COLORS["white"]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=5.7,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=1.0,
            ecolor=color,
            elinewidth=0.85,
            capsize=2.2,
            zorder=3,
        )
        dx, dy = offsets[row["provider"]]
        ax.annotate(row["provider"], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.6, fontweight="semibold" if pass_flag else "normal")
    ax.set_xlim(-35, 67)
    ax.set_ylim(-56, 57)
    ax.set_xlabel("Full ENSR − ENSR-base mean score")
    ax.set_ylabel("Success-rate difference (pp)")
    ax.grid(color=COLORS["grid"], linewidth=0.7)
    clean_axis(ax)
    ax.text(0.97, 0.96, "upper right: both point estimates positive", transform=ax.transAxes, ha="right", va="top", fontsize=5.2, color=COLORS["subtext"])
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["ensr"], markeredgecolor=COLORS["ensr"], label="directional pass"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["iper"], label="did not pass"),
        ],
        loc="lower left",
        fontsize=5.3,
        handletextpad=0.35,
    )


def draw_panel_e(ax_status: plt.Axes, ax_cost: plt.Axes, effect_rows: list[dict], status_rows: list[dict], costs: dict) -> None:
    ax_status.set_title("Provider outcomes", loc="left", pad=6)
    panel_label(ax_status, "e")
    provider_order = ["DeepSeek", "GLM", "GPT", "Kimi"]
    methods = [("iper_rag", "I"), ("ensr_v2", "E")]
    categories = [
        ("terminal_success", "Success", COLORS["ensr"]),
        ("terminal_failure", "Terminal failure", COLORS["iper"]),
        ("budget_exhausted", "Budget exhausted", COLORS["iper_light"]),
        ("plan_exhausted", "Plan exhausted", "#C9CFD7"),
        ("model_stop", "Model stop", "#8D98A6"),
        ("invalid_decision", "Invalid decision", "#D7B77E"),
        ("external_error", "External error", "#7F5265"),
    ]
    status_map = {(row["provider"], row["method"]): row for row in status_rows}
    y_positions: list[float] = []
    y_labels: list[str] = []
    for p_idx, provider in enumerate(provider_order):
        base = (len(provider_order) - 1 - p_idx) * 2.5
        for method, short in methods:
            yi = base + (0.55 if method == "iper_rag" else -0.55)
            y_positions.append(yi)
            y_labels.append(f"{provider}  {short}")
            left = 0
            row = status_map[(provider, method)]
            for key, _label, color in categories:
                value = row[key]
                if value:
                    ax_status.barh(yi, value, left=left, height=0.72, color=color, edgecolor="white", linewidth=0.4)
                    if value >= 4:
                        ax_status.text(left + value / 2, yi, str(value), ha="center", va="center", fontsize=4.9, color="white" if color in {COLORS["iper"], COLORS["ensr"], "#8D98A6", "#7F5265"} else COLORS["text"])
                left += value
    ax_status.set_xlim(0, 30)
    ax_status.set_yticks(y_positions, y_labels)
    ax_status.set_xlabel("Episodes")
    ax_status.set_xticks([0, 10, 20, 30])
    ax_status.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    clean_axis(ax_status)
    ax_status.text(0.99, 1.01, "B: ENSR-base   F: full ENSR", transform=ax_status.transAxes, fontsize=5.1, color=COLORS["subtext"], ha="right", va="bottom")
    ax_status.legend(
        handles=[Line2D([], [], color=color, linewidth=5, label=label) for _, label, color in categories[:-1]],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=3,
        fontsize=4.7,
        handlelength=1.4,
        columnspacing=0.7,
    )

    ax_cost.set_title("Estimated API cost", loc="left", pad=6)
    y = np.arange(len(provider_order))[::-1]
    effect_map = {row["provider"]: row for row in effect_rows}
    for yi, provider in zip(y, provider_order):
        row = effect_map[provider]
        low, high = row["cost_low_cny"], row["cost_high_cny"]
        midpoint = np.sqrt(low * high)
        ax_cost.plot([low, high], [yi, yi], color=COLORS["subtext"], linewidth=2.0, solid_capstyle="round")
        ax_cost.scatter(midpoint, yi, s=26, color=COLORS["gold"], edgecolor="white", linewidth=0.55, zorder=3)
        ax_cost.text(high + 5, yi, f"{low:.0f}–{high:.0f}", va="center", fontsize=5.2, color=COLORS["text"])
    ax_cost.set_xlim(0, 270)
    ax_cost.set_ylim(-0.7, 3.7)
    ax_cost.set_yticks(y, provider_order)
    ax_cost.set_xlabel("Estimated CNY, both methods")
    ax_cost.set_xticks([0, 100, 200])
    ax_cost.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    clean_axis(ax_cost)
    ax_cost.text(
        0.02,
        0.04,
        f"Total: {costs['total_estimated_low_cny']:.0f}–{costs['total_estimated_high_cny']:.0f} CNY",
        transform=ax_cost.transAxes,
        fontsize=5.3,
        color=COLORS["subtext"],
    )


def draw_panel_f(ax: plt.Axes, e15: dict, e15r: dict, clean_config: dict, recovery_config: dict) -> list[dict]:
    ax.set_title("Prospective freeze timeline", loc="left", pad=6)
    panel_label(ax, "f", x=-0.17)
    clean = clean_config["common_controls"]
    recovery = recovery_config["profiles"][0]
    stages = [
        {
            "stage": "E14-SENS",
            "lane": "E15 protocol",
            "x_position": 0,
            "boundary": "Diagnostic only",
            "detail": "Failure modes identified; primary E14 retained",
            "status": "audit",
        },
        {
            "stage": "Freeze 1",
            "lane": "E15 protocol",
            "x_position": 1,
            "boundary": "Before E15 calls",
            "detail": f"{clean['max_input_characters']//1000}k chars · T={clean['temperature']:.0f} · 1 structured retry",
            "status": "freeze",
        },
        {
            "stage": "E15 held-out",
            "lane": "E15 protocol",
            "x_position": 2,
            "boundary": "180 episodes",
            "detail": f"{e15['clean_adapter_summary']['passing_profile_count']}/{e15['clean_adapter_summary']['required_profile_count']} profiles pass · overall FAIL",
            "status": "fail",
        },
        {
            "stage": "Freeze 2",
            "lane": "E15R recovery",
            "x_position": 3,
            "boundary": "After E15, before E15R",
            "detail": f"{recovery['max_input_characters']//1000}k · T={recovery['temperature']:.1f} · p={recovery['top_p']:.1f} · {recovery['max_output_tokens']} out",
            "status": "freeze",
        },
        {
            "stage": "E15R held-out",
            "lane": "E15R recovery",
            "x_position": 4,
            "boundary": "60 episodes",
            "detail": "Direction PASS · operation FAIL",
            "status": "fail",
        },
    ]

    y_main, y_recovery = 1.00, 0.14
    ax.set_xlim(-0.28, 4.28)
    ax.set_ylim(-0.64, 1.58)
    ax.set_yticks([y_main, y_recovery], ["E15 protocol", "E15R recovery"])
    ax.set_xticks(
        np.arange(5),
        ["E14\ndiagnostic", "Freeze 1", "E15\nheld-out", "Freeze 2", "E15R\nheld-out"],
    )
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, zorder=0)
    clean_axis(ax, left=False, bottom=True)
    ax.tick_params(axis="y", length=0, pad=5)

    # Main prospective sequence and the explicitly separate post-E15 recovery lane.
    ax.plot([0, 2], [y_main, y_main], color=COLORS["neutral_dark"], linewidth=1.2, zorder=1)
    ax.annotate(
        "",
        xy=(2, y_recovery),
        xytext=(2, y_main - 0.05),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["neutral_dark"], lw=0.8, linestyle=(0, (3, 2)), mutation_scale=7),
    )
    ax.plot([2, 4], [y_recovery, y_recovery], color=COLORS["neutral_dark"], linewidth=1.2, linestyle=(0, (3, 2)), zorder=1)

    ax.scatter(0, y_main, s=38, marker="o", facecolor=COLORS["white"], edgecolor=COLORS["neutral_dark"], linewidth=1.0, zorder=3)
    for x, y in [(1, y_main), (3, y_recovery)]:
        ax.scatter(x, y, s=49, marker="D", facecolor=COLORS["green"], edgecolor=COLORS["white"], linewidth=0.7, zorder=4)
    for x, y in [(2, y_main), (4, y_recovery)]:
        ax.scatter(x, y, s=52, marker="X", facecolor=COLORS["iper"], edgecolor=COLORS["white"], linewidth=0.7, zorder=4)

    annotations = [
        (0, y_main, "Failure-mode\ndiagnostic", COLORS["subtext"]),
        (1, y_main, f"FROZEN\n{clean['max_input_characters']//1000}k · T={clean['temperature']:.0f} · retry 1", COLORS["green"]),
        (2, y_main, f"{e15['clean_adapter_summary']['passing_profile_count']}/{e15['clean_adapter_summary']['required_profile_count']} profiles pass\noverall FAIL", COLORS["iper"]),
        (3, y_recovery, f"FROZEN\n{recovery['max_input_characters']//1000}k · T={recovery['temperature']:.1f} · p={recovery['top_p']:.1f} · out {recovery['max_output_tokens']}", COLORS["green"]),
        (4, y_recovery, "Direction PASS\noperation FAIL", COLORS["iper"]),
    ]
    for x, y, label, color in annotations:
        ax.annotate(
            label,
            (x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.25,
            color=color,
            fontweight="semibold" if "FROZEN" in label or "FAIL" in label else "normal",
        )

    ax.text(
        1.5,
        1.48,
        "profile fixed before held-out calls",
        ha="center",
        va="top",
        fontsize=5.2,
        color=COLORS["subtext"],
    )
    ax.text(
        2.0,
        -0.49,
        "E15R remains a separate, non-replacement result.",
        ha="center",
        va="center",
        fontsize=5.35,
        color=COLORS["iper"],
        fontweight="semibold",
    )
    return stages


def render() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    efficiency = read_json(EFFICIENCY_JSON)
    summary = read_json(PAPER_SUMMARY_JSON)
    e15 = read_json(E15_JSON)
    e15r = read_json(E15R_JSON)
    e17 = read_json(E17_JSON)
    costs = read_json(E17_COST_JSON)
    clean_config = read_json(CLEAN_CONFIG_JSON)
    recovery_config = read_json(RECOVERY_CONFIG_JSON)

    if not e15["complete_all_profiles"] or e15["observed_episode_count"] != 180:
        raise RuntimeError("E15 frozen audit is incomplete.")
    if not e15r["complete_matrix"] or e15r["observed_episode_count"] != 60:
        raise RuntimeError("E15R frozen audit is incomplete.")
    if not e17["complete_all_profiles"] or e17["observed_episode_count"] != 240:
        raise RuntimeError("E17 frozen audit is incomplete.")

    efficiency_rows, pairs = extract_efficiency(efficiency)
    adapter_rows = extract_adapter_rows(summary, e15, e15r)
    provider_rows, status_rows = extract_provider_rows(e17, costs)

    fig = plt.figure(figsize=(12.2, 9.6))
    outer = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.48, 1.0],
        height_ratios=[1.0, 1.0, 1.08],
        left=0.075,
        right=0.985,
        bottom=0.12,
        top=0.965,
        wspace=0.31,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])
    lower_left = outer[2, 0].subgridspec(1, 2, width_ratios=[1.23, 0.9], wspace=0.42)
    ax_e_status = fig.add_subplot(lower_left[0, 0])
    ax_e_cost = fig.add_subplot(lower_left[0, 1])
    ax_f = fig.add_subplot(outer[2, 1])

    draw_panel_a(ax_a, pairs)
    draw_panel_b(ax_b, pairs)
    draw_panel_c(ax_c, adapter_rows)
    draw_panel_d(ax_d, provider_rows)
    draw_panel_e(ax_e_status, ax_e_cost, provider_rows, status_rows, costs)
    freeze_rows = draw_panel_f(ax_f, e15, e15r, clean_config, recovery_config)

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, facecolor="white")
    plt.close(fig)

    write_csv(OUTPUT_DIR / "Figure_8a_efficiency_trajectories_source.csv", pairs)
    write_csv(
        OUTPUT_DIR / "Figure_8b_resource_ratios_source.csv",
        [
            {
                "experiment": row["experiment"],
                "profile": row["profile"],
                "label": row["label"],
                "model_call_ratio_ensr_over_iper": row["model_call_ratio"],
                "token_ratio_ensr_over_iper": row["token_ratio"],
                "retrieval_ratio_ensr_over_iper": row["retrieval_ratio"],
            }
            for row in pairs
        ],
    )
    write_csv(OUTPUT_DIR / "Figure_8c_adapter_audit_source.csv", adapter_rows)
    write_csv(OUTPUT_DIR / "Figure_8d_external_effects_source.csv", provider_rows)
    write_csv(OUTPUT_DIR / "Figure_8e_external_status_source.csv", status_rows)
    write_csv(
        OUTPUT_DIR / "Figure_8e_external_cost_source.csv",
        [
            {
                "provider": row["provider"],
                "profile": row["profile"],
                "estimated_low_cny": row["cost_low_cny"],
                "estimated_high_cny": row["cost_high_cny"],
                "pricing_retrieved_at": costs["pricing_retrieved_at"],
                "usd_to_cny": costs["usd_to_cny"],
            }
            for row in provider_rows
        ],
    )
    write_csv(OUTPUT_DIR / "Figure_8f_freeze_sequence_source.csv", freeze_rows)

    manifest = {
        "figure": "Figure 8",
        "output_stem": str(OUTPUT_STEM.relative_to(REPO_ROOT)),
        "sources": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in [
                EFFICIENCY_JSON,
                PAPER_SUMMARY_JSON,
                E15_JSON,
                E15R_JSON,
                E17_JSON,
                E17_COST_JSON,
                CLEAN_CONFIG_JSON,
                RECOVERY_CONFIG_JSON,
            ]
        },
        "audit_counts": {
            "internal_paired_settings": len(pairs),
            "adapter_rows": len(adapter_rows),
            "e15_episodes": e15["observed_episode_count"],
            "e15r_episodes": e15r["observed_episode_count"],
            "e17_episodes": e17["observed_episode_count"],
            "external_directional_pass": e17["robustness_summary"]["directional_pass_count"],
            "external_directional_required": e17["robustness_summary"]["required_count"],
        },
    }
    (OUTPUT_DIR / "Figure_8_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    render()
