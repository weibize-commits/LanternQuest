"""Render the Nature-style Figure 5 from frozen ScienceWorld results.

The composition deliberately uses four different evidence forms:
  a) episode-level raincloud distributions,
  b) a paired-task success matrix,
  c) a cross-model slopegraph,
  d) a paired-effect interval plot.

No observations are simulated. All episode points are read from the frozen E13
run files, and all cross-model effects are read from the frozen paper summary.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENCEWORLD_DIR = REPO_ROOT / "artifacts" / "benchmarks" / "scienceworld"
SOURCE_JSON = SCIENCEWORLD_DIR / "paper_results_summary_v1.json"
E13_RUN_ROOT = (
    SCIENCEWORLD_DIR
    / "ensr_v3_runs_formal"
    / "scienceworld_ensr_v3"
    / "bit_qwen3_235b"
    / "test"
)
OUTPUT_DIR = REPO_ROOT / "artifacts" / "paper" / "figures" / "figure5"
OUTPUT_STEM = OUTPUT_DIR / "Figure_5_ScienceWorld_performance_nature_v2"


COLORS = {
    "text": "#26364A",
    "subtext": "#667085",
    "grid": "#E8EBEF",
    "sequential": "#B8C3DF",
    "iper": "#6576A5",
    "ensr": "#C95E72",
    "neutral": "#A8AFB8",
    "zero_band": "#F2F3F5",
    "white": "#FFFFFF",
}

METHODS = [
    ("b4_sequential_planner", "Sequential planner", COLORS["sequential"]),
    ("iper_rag", "ENSR-base", COLORS["iper"]),
    ("ensr_v2", "ENSR", COLORS["ensr"]),
]

MODELS = [
    ("bit_qwen3_235b", "Qwen3-235B"),
    ("bit_ceep_70b", "CEEP-70B"),
    ("bit_qwen3_32b", "Qwen3-32B"),
    ("bit_qwen3_8b", "Qwen3-8B"),
    ("bit_qwen3_0_6b", "Qwen3-0.6B"),
]


def load_summary() -> dict:
    with SOURCE_JSON.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not data["e13"]["complete"] or not data["e14"]["complete"]:
        raise RuntimeError("Frozen E13 or E14 results are incomplete.")
    return data


def load_e13_episodes(summary: dict) -> dict[str, list[dict]]:
    episodes: dict[str, list[dict]] = {}
    for method_key, _, _ in METHODS:
        rows: list[dict] = []
        for path in sorted((E13_RUN_ROOT / method_key).glob("*.json")):
            if path.name.startswith("summary"):
                continue
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            rows.append(
                {
                    "method": method_key,
                    "task_id": str(record["task_id"]),
                    "variation": int(record["variation"]),
                    "seed": int(record["seed"]),
                    "score": float(np.clip(record.get("final_score", 0), 0, 100)),
                    "success": int(bool(record.get("task_success", False))),
                    "status": str(record.get("status", "unknown")),
                }
            )

        expected = summary["e13"]["summaries"][method_key]
        if len(rows) != int(expected["episodes"]):
            raise RuntimeError(f"Unexpected E13 episode count for {method_key}.")
        mean_score = float(np.mean([row["score"] for row in rows]))
        success_rate = float(np.mean([row["success"] for row in rows]))
        if not np.isclose(mean_score, expected["mean_clipped_score"]):
            raise RuntimeError(f"Score mismatch for {method_key}.")
        if not np.isclose(success_rate, expected["task_success_rate"]):
            raise RuntimeError(f"Success mismatch for {method_key}.")
        episodes[method_key] = rows
    return episodes


def build_task_matrix(
    episodes: dict[str, list[dict]],
) -> tuple[list[tuple[str, int]], np.ndarray, list[dict]]:
    by_method: dict[str, dict[tuple[str, int], list[dict]]] = {}
    for method_key, _, _ in METHODS:
        grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in episodes[method_key]:
            grouped[(row["task_id"], row["variation"])].append(row)
        by_method[method_key] = grouped

    task_keys = set(by_method[METHODS[0][0]])
    for method_key, _, _ in METHODS[1:]:
        task_keys &= set(by_method[method_key])
    if len(task_keys) != 30:
        raise RuntimeError(f"Expected 30 paired task variants, found {len(task_keys)}.")

    task_stats: list[dict] = []
    for key in task_keys:
        iper_rows = by_method["iper_rag"][key]
        ensr_rows = by_method["ensr_v2"][key]
        task_stats.append(
            {
                "key": key,
                "score_gain": np.mean([r["score"] for r in ensr_rows])
                - np.mean([r["score"] for r in iper_rows]),
                "success_gain": sum(r["success"] for r in ensr_rows)
                - sum(r["success"] for r in iper_rows),
            }
        )

    task_stats.sort(
        key=lambda row: (row["score_gain"], row["success_gain"]), reverse=True
    )
    ordered_keys = [row["key"] for row in task_stats]
    matrix = np.zeros((len(METHODS), len(ordered_keys)), dtype=int)
    task_rows: list[dict] = []
    for col, key in enumerate(ordered_keys):
        task_row = {
            "rank": col + 1,
            "task_id": key[0],
            "variation": key[1],
            "ensr_minus_iper_score": task_stats[col]["score_gain"],
        }
        for method_index, (method_key, _, _) in enumerate(METHODS):
            cell_rows = by_method[method_key][key]
            success_count = int(sum(r["success"] for r in cell_rows))
            matrix[method_index, col] = success_count
            task_row[f"{method_key}_successes_of_3"] = success_count
            task_row[f"{method_key}_mean_score"] = float(
                np.mean([r["score"] for r in cell_rows])
            )
        task_rows.append(task_row)
    return ordered_keys, matrix, task_rows


def extract_cross_model(summary: dict) -> list[dict]:
    rows: list[dict] = []
    for model_key, model_label in MODELS:
        model = summary["e14"]["models"][model_key]
        iper = model["summaries"]["iper_rag"]
        ensr = model["summaries"]["ensr_v2"]
        score = model["ensr_minus_baseline"]["clipped_score"]
        success = model["ensr_minus_baseline"]["task_success_rate"]
        rows.append(
            {
                "model_key": model_key,
                "model": model_label,
                "episodes_per_method": int(ensr["episodes"]),
                "iper_score": float(iper["mean_clipped_score"]),
                "ensr_score": float(ensr["mean_clipped_score"]),
                "score_delta": float(score["point_estimate"]),
                "score_low": float(score["ci95_low"]),
                "score_high": float(score["ci95_high"]),
                "iper_success": 100.0 * float(iper["task_success_rate"]),
                "ensr_success": 100.0 * float(ensr["task_success_rate"]),
                "success_delta": 100.0 * float(success["point_estimate"]),
                "success_low": 100.0 * float(success["ci95_low"]),
                "success_high": 100.0 * float(success["ci95_high"]),
            }
        )
    return rows


def write_source_tables(
    episodes: dict[str, list[dict]], task_rows: list[dict], cross_model: list[dict]
) -> None:
    episode_path = OUTPUT_DIR / "Figure_5_episode_scores_v2.csv"
    with episode_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "task_id",
                "variation",
                "seed",
                "score",
                "success",
                "status",
            ],
        )
        writer.writeheader()
        for method_key, _, _ in METHODS:
            writer.writerows(episodes[method_key])

    task_path = OUTPUT_DIR / "Figure_5_task_success_matrix_v2.csv"
    with task_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    cross_path = OUTPUT_DIR / "Figure_5_cross_model_effects_v2.csv"
    with cross_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cross_model[0]))
        writer.writeheader()
        writer.writerows(cross_model)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8.4,
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


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["text"])
    ax.spines["bottom"].set_color(COLORS["text"])
    ax.tick_params(axis="both", colors=COLORS["text"], width=0.7, length=3)


def panel_label(ax: plt.Axes, label: str, x: float = -0.14) -> None:
    ax.text(
        x,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9.8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["text"],
    )


def draw_half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    y: float,
    color: str,
    rng: np.random.Generator,
) -> None:
    grid = np.linspace(0, 100, 350)
    kde = gaussian_kde(values, bw_method=0.22)
    density = kde(grid)
    density = density / max(density.max(), 1e-12) * 0.30
    ax.fill_between(
        grid,
        y,
        y + density,
        color=color,
        alpha=0.72,
        linewidth=0,
        zorder=1,
    )
    ax.plot(grid, y + density, color=color, linewidth=0.9, zorder=2)

    jitter_y = y - rng.uniform(0.07, 0.28, size=len(values))
    jitter_x = np.clip(values + rng.normal(0, 0.7, size=len(values)), 0, 100)
    ax.scatter(
        jitter_x,
        jitter_y,
        s=7,
        color=color,
        alpha=0.38,
        linewidth=0,
        zorder=2,
    )

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    mean = float(np.mean(values))
    ax.plot([q1, q3], [y - 0.01, y - 0.01], color=COLORS["text"], lw=3.0, zorder=4)
    ax.plot([median, median], [y - 0.09, y + 0.07], color="white", lw=1.2, zorder=5)
    ax.scatter(
        [mean],
        [y - 0.01],
        marker="D",
        s=28,
        facecolor=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    ax.text(
        min(mean + 3.0, 96),
        y + 0.10,
        f"mean {mean:.1f}",
        fontsize=5.7,
        color=color,
        fontweight="bold",
        ha="left" if mean < 93 else "right",
        va="bottom",
    )


def plot_episode_distributions(
    ax: plt.Axes, episodes: dict[str, list[dict]], summary: dict
) -> None:
    rng = np.random.default_rng(20260925)
    y_positions = [2.1, 1.05, 0.0]
    for y, (method_key, _, color) in zip(y_positions, METHODS):
        values = np.array([row["score"] for row in episodes[method_key]], dtype=float)
        draw_half_violin(ax, values, y, color, rng)

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Clipped episode score")
    ax.set_ylim(-0.42, 2.78)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label, _ in METHODS])
    for tick, (_, _, color) in zip(ax.get_yticklabels(), METHODS):
        tick.set_color(color)
        tick.set_fontweight("semibold")
    for x in [25, 50, 75]:
        ax.axvline(x, color=COLORS["grid"], lw=0.6, zorder=0)
    ax.set_title("Held-out episode-score distributions", loc="left", pad=7)
    effect = summary["e13"]["ensr_minus_iper"]["clipped_score"]
    ax.text(
        99,
        2.62,
        "ENSR − ENSR-base\n"
        f"+{effect['point_estimate']:.1f}  "
        f"[{effect['ci95_low']:.1f}, {effect['ci95_high']:.1f}]",
        ha="right",
        va="top",
        fontsize=6.3,
        color=COLORS["ensr"],
        fontweight="bold",
        linespacing=1.25,
    )
    ax.text(
        99,
        2.30,
        "paired 95% bootstrap CI",
        ha="right",
        va="top",
        fontsize=5.2,
        color=COLORS["subtext"],
    )
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    panel_label(ax, "a", x=-0.13)


def plot_success_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    episodes: dict[str, list[dict]],
) -> None:
    cmap = mcolors.ListedColormap(["#F4F6F7", "#D9E9EA", "#8FC0C4", "#367982"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="none")
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([label for _, label, _ in METHODS])
    for tick, (_, _, color) in zip(ax.get_yticklabels(), METHODS):
        tick.set_color(color)
        tick.set_fontweight("semibold")
    ax.set_xticks([0, 9, 19, 29])
    ax.set_xticklabels(["1", "10", "20", "30"])
    ax.set_xlabel("Paired task rank (ordered by ENSR − ENSR-base score gain)")
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_xlim(-0.5, 35.2)
    ax.set_ylim(2.5, -0.5)
    for row_index, (method_key, _, _) in enumerate(METHODS):
        successes = int(sum(row["success"] for row in episodes[method_key]))
        rate = 100.0 * successes / len(episodes[method_key])
        ax.text(
            30.7,
            row_index,
            f"{successes}/90  ({rate:.1f}%)",
            ha="left",
            va="center",
            fontsize=5.8,
            color=COLORS["text"],
        )
    ax.set_title("Where task success occurs", loc="left", pad=7)
    ax.text(
        35.1,
        -0.67,
        "0/3 → 3/3 successful seeds",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=COLORS["subtext"],
        clip_on=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "b", x=-0.13)


def repel_labels(values: list[float], minimum_gap: float = 2.2) -> list[float]:
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for previous, current in zip(order[:-1], order[1:]):
        if placed[current] - placed[previous] < minimum_gap:
            placed[current] = placed[previous] + minimum_gap
    if placed.max() > 54.0:
        placed -= placed.max() - 54.0
    return placed.tolist()


def plot_cross_model_slopes(ax: plt.Axes, rows: list[dict]) -> None:
    endpoint_values = [row["ensr_score"] for row in rows]
    label_y = repel_labels(endpoint_values)
    for row, ly in zip(rows, label_y):
        uncertain = row["score_low"] <= 0
        ax.plot(
            [0, 1],
            [row["iper_score"], row["ensr_score"]],
            color="#98A0AA" if uncertain else "#7F8894",
            lw=1.05,
            ls="--" if uncertain else "-",
            alpha=0.9,
            zorder=1,
        )
        ax.scatter(
            [0],
            [row["iper_score"]],
            s=28,
            color=COLORS["iper"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.scatter(
            [1],
            [row["ensr_score"]],
            s=32,
            facecolor="white" if uncertain else COLORS["ensr"],
            edgecolor=COLORS["ensr"],
            linewidth=1.1,
            zorder=3,
        )
        ax.text(
            0.48,
            (row["iper_score"] + row["ensr_score"]) / 2 + 0.7,
            f"+{row['score_delta']:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color=COLORS["ensr"],
            fontweight="bold",
        )
        ax.plot(
            [1.03, 1.18, 1.40],
            [row["ensr_score"], ly, ly],
            color="#A4ABB4",
            lw=0.55,
            clip_on=False,
        )
        ax.text(
            1.43,
            ly,
            row["model"],
            ha="left",
            va="center",
            fontsize=5.4,
            color=COLORS["text"],
        )

    ax.set_xlim(-0.18, 1.95)
    ax.set_ylim(0, 57)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ENSR-base", "ENSR"])
    ax.set_ylabel("Mean clipped score")
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_title("Scores rise across model profiles", loc="left", pad=7)
    for y in [10, 20, 30, 40, 50]:
        ax.axhline(y, color=COLORS["grid"], lw=0.55, zorder=0)
    ax.text(
        1.93,
        2.5,
        "open endpoint: score CI crosses zero",
        ha="right",
        va="bottom",
        fontsize=4.8,
        color=COLORS["subtext"],
    )
    clean_axis(ax)
    panel_label(ax, "c", x=-0.18)


def plot_success_intervals(ax: plt.Axes, rows: list[dict]) -> None:
    y = np.arange(len(rows) - 1, -1, -1)
    ax.axvspan(-2, 2, color=COLORS["zero_band"], zorder=0)
    ax.axvline(0, color="#6F7780", lw=0.8, zorder=1)
    for yi, row in zip(y, rows):
        ax.hlines(
            yi,
            row["success_low"],
            row["success_high"],
            color=COLORS["text"],
            lw=1.1,
            zorder=2,
        )
        ax.vlines(
            [row["success_low"], row["success_high"]],
            yi - 0.08,
            yi + 0.08,
            color=COLORS["text"],
            lw=0.75,
            zorder=2,
        )
        fully_positive = row["success_low"] > 0
        ax.scatter(
            row["success_delta"],
            yi,
            s=34,
            facecolor=COLORS["ensr"] if fully_positive else "white",
            edgecolor=COLORS["ensr"],
            linewidth=1.1,
            zorder=3,
        )
        ax.text(
            66.5,
            yi,
            f"{row['success_delta']:+.1f} "
            f"[{row['success_low']:.1f}, {row['success_high']:.1f}]",
            ha="right",
            va="center",
            fontsize=5.1,
            color=COLORS["text"],
        )
        ax.axhline(yi, color="#F0F2F4", lw=0.45, zorder=0)

    ax.set_xlim(-15, 68)
    ax.set_xticks([-10, 0, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.65, 4.85)
    ax.set_yticks(y)
    ax.set_yticklabels([row["model"] for row in rows])
    ax.set_xlabel("ENSR − ENSR-base success rate (percentage points)")
    ax.set_title("Success effects and uncertainty", loc="left", pad=7)
    ax.text(
        67,
        4.58,
        "robustness 5/5  |  threshold 4/5",
        ha="right",
        va="center",
        fontsize=5.2,
        color=COLORS["ensr"],
        fontweight="bold",
    )
    clean_axis(ax)
    ax.tick_params(axis="y", length=0, pad=5)
    panel_label(ax, "d", x=-0.18)


def render_figure(
    summary: dict,
    episodes: dict[str, list[dict]],
    matrix: np.ndarray,
    cross_model: list[dict],
) -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(7.25, 5.65), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.48, 1.0],
        height_ratios=[1.12, 0.88],
        left=0.115,
        right=0.985,
        top=0.945,
        bottom=0.11,
        wspace=0.34,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_episode_distributions(ax_a, episodes, summary)
    plot_success_matrix(ax_b, matrix, episodes)
    plot_cross_model_slopes(ax_c, cross_model)
    plot_success_intervals(ax_d, cross_model)

    fig.text(
        0.115,
        0.035,
        "a,b, E13 held-out results (90 episodes per method; 30 paired task variants × 3 seeds). "
        "c,d, E14 cross-model results (30 episodes per method and model). "
        "Intervals are paired 95% bootstrap CIs from 20,000 resamples; open markers touch or cross zero.",
        ha="left",
        va="bottom",
        fontsize=5.15,
        color=COLORS["subtext"],
    )
    return fig


def write_caption() -> None:
    caption = """# Figure 5 caption draft — Nature-style v2

**Figure 5 | Held-out ScienceWorld performance and cross-model robustness.**
**a**, Episode-level clipped-score distributions for the sequential planner,
ENSR-base and full ENSR in E13. ENSR-base is the internal controller
preregistered under the protocol identifier IPER-RAG. Half violins show kernel-density estimates, points
show all 90 episodes per method, thick segments show the interquartile range,
white ticks show medians and diamonds show means. The annotation gives the
paired full-ENSR minus ENSR-base mean-score difference and 95% bootstrap confidence
interval. **b**, Number of successful seeds (0–3) for each of 30 paired task
variants, ordered by the full-ENSR minus ENSR-base score gain. Right-hand labels give
aggregate successes and rates. **c**, Absolute ENSR-base and full-ENSR mean scores
across five model profiles in E14; labels on each slope give the point-estimate
gain. The open endpoint identifies the model profile for which the paired score
confidence interval crosses zero. **d**, Paired full-ENSR minus ENSR-base task-success
differences across the same model profiles. Lines denote 95% bootstrap
confidence intervals; open markers indicate intervals that touch or cross zero.
E14 contained 30 episodes per method and model. All confidence intervals use
20,000 paired bootstrap resamples.
"""
    (OUTPUT_DIR / "Figure_5_caption_nature_v2.md").write_text(
        caption, encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    episodes = load_e13_episodes(summary)
    _, matrix, task_rows = build_task_matrix(episodes)
    cross_model = extract_cross_model(summary)
    write_source_tables(episodes, task_rows, cross_model)
    write_caption()
    fig = render_figure(summary, episodes, matrix, cross_model)

    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
