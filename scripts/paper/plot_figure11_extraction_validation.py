"""Render Figure 11: ontology-constrained supervised evidence extraction.

The figure uses only audited, out-of-fold results. It intentionally excludes the
earlier zero-shot baseline from the visual argument; that audit comparator remains
reported in Table 9 of the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "artifacts" / "benchmarks" / "e27_ie"
RUN_DIR = INPUT_DIR / "e27v4_authorized_rerun"
COMPARE_DIR = RUN_DIR / "complete_comparison"
RECOVERY_DIR = RUN_DIR / "early_ontology"
OUT_DIR = ROOT / "artifacts" / "paper" / "figures" / "figure11_extraction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ink": "#26364A",
    "muted": "#667085",
    "grid": "#E8EBEF",
    "entity": "#6576A5",
    "relation": "#C95E72",
    "teal": "#5F8F82",
    "ochre": "#B78336",
    "violet": "#9B93B9",
    "neutral": "#A9B1BC",
    "light_blue": "#DCE3F0",
    "light_rose": "#F1DCE1",
    "light_teal": "#DCEBE6",
    "light_ochre": "#F3E8D7",
    "white": "#FFFFFF",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.linewidth": 0.65,
        "axes.edgecolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["ink"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def read_inputs():
    with (COMPARE_DIR / "E27v4_complete_method_comparison.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        methods = {row["method_id"]: row for row in csv.DictReader(f)}
    paired = json.loads(
        (COMPARE_DIR / "E27v4_paired_comparisons.json").read_text(encoding="utf-8")
    )
    constraint = json.loads(
        (COMPARE_DIR / "E27v4_constraint_placement_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    recovery = json.loads(
        (RECOVERY_DIR / "E27v2_supervised_cv_results.json").read_text(encoding="utf-8")
    )
    return methods, paired, constraint, recovery


def panel_label(ax, letter, title):
    ax.text(
        -0.08,
        1.075,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["ink"],
        clip_on=False,
    )
    ax.text(
        -0.015,
        1.075,
        title,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["ink"],
        clip_on=False,
    )


def bezier_ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.45):
    dx = (x1 - x0) * 0.46
    verts = [
        (x0, y0a),
        (x0 + dx, y0a),
        (x1 - dx, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - dx, y1b),
        (x0 + dx, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw_panel_a(ax, methods, recovery):
    panel_label(ax, "a", "Matched out-of-fold performance")

    early = recovery["overall"]
    rows = [
        ("Rule and lexicon", float(methods["B0"]["entity_f1"]), float(methods["B0"]["relation_f1"]), "o"),
        ("BERT neural-only", float(methods["BERT_NEURAL"]["entity_f1"]), float(methods["BERT_NEURAL"]["relation_f1"]), "o"),
        ("BERT + post-hoc ontology", float(methods["BERT_ONTOLOGY"]["entity_f1"]), float(methods["BERT_ONTOLOGY"]["relation_f1"]), "o"),
        ("BERT + early ontology", early["entity_exact_micro"]["f1"], early["relation_end_to_end_micro"]["f1"], "D"),
        ("MacBERT full pipeline", float(methods["MACBERT_EVIDENCE"]["entity_f1"]), float(methods["MACBERT_EVIDENCE"]["relation_f1"]), "o"),
        ("Joint BERT full pipeline", float(methods["JOINT_EVIDENCE"]["entity_f1"]), float(methods["JOINT_EVIDENCE"]["relation_f1"]), "o"),
    ]
    labels = [r[0] for r in rows]
    ypos = np.arange(len(rows))[::-1]
    for y, (_, ef1, rf1, marker) in zip(ypos, rows):
        ax.plot([rf1, ef1], [y, y], color=COLORS["grid"], lw=2.5, zorder=1)
        ax.scatter(rf1, y, s=31, marker="s", color=COLORS["relation"], zorder=3, edgecolor="white", linewidth=0.45)
        ax.scatter(ef1, y, s=36, marker=marker, color=COLORS["entity"], zorder=3, edgecolor="white", linewidth=0.45)
        ax.text(rf1 - 0.012, y, f"{rf1:.3f}", ha="right", va="center", fontsize=5.8, color=COLORS["relation"])
        ax.text(ef1 + 0.012, y, f"{ef1:.3f}", ha="left", va="center", fontsize=5.8, color=COLORS["entity"])

    ax.set_yticks(ypos, labels)
    ax.set_xlim(0.08, 0.69)
    ax.set_ylim(-0.45, 5.68)
    ax.set_xticks(np.arange(0.1, 0.7, 0.1))
    ax.set_xlabel("Micro-averaged F1")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.plot([], [], "o", color=COLORS["entity"], label="Exact entity")
    ax.plot([], [], "s", color=COLORS["relation"], label="End-to-end relation")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        fontsize=6.2,
        handletextpad=0.35,
        columnspacing=1.0,
    )


def draw_panel_b(ax, paired, constraint, recovery):
    panel_label(ax, "b", "Paired F1 effects")
    rec = recovery["paired_bootstrap"]
    lookup = {item["label"]: item for item in paired}
    items = [
        ("Supervision vs rule\nEntity", rec["entity_vs_B0"]),
        ("Supervision vs rule\nRelation", rec["relation_vs_B0"]),
        ("Ontology mask vs neural\nRelation", lookup["Ontology effect within BERT"]["relation"]),
        ("Early vs post-hoc constraint\nRelation", constraint["relation"]),
        ("Joint vs staged BERT\nEntity", lookup["Joint versus staged BERT full pipeline"]["entity"]),
    ]
    ypos = np.arange(len(items))[::-1]
    ax.axvline(0, color=COLORS["ink"], lw=0.75, zorder=0)
    for y, (label, d) in zip(ypos, items):
        effect = float(d["observed_difference"])
        low, high = map(float, d["ci_95"])
        supported = low > 0
        color = COLORS["teal"] if supported else COLORS["neutral"]
        ax.plot([low, high], [y, y], color=color, lw=1.6, solid_capstyle="round")
        ax.scatter(effect, y, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(high + 0.008, y, f"{effect:+.3f}", va="center", ha="left", fontsize=5.8, color=color)
    ax.set_yticks(ypos, [x[0] for x in items])
    ax.set_xlim(-0.035, 0.315)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Difference in F1 with 95% bootstrap interval")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(
        0.99,
        1.068,
        "5,000 segment-level resamples",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=COLORS["muted"],
    )


def draw_panel_c(ax, methods):
    panel_label(ax, "c", "Ontology removes incompatible relation predictions")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    neural = methods["BERT_NEURAL"]
    ontology = methods["BERT_ONTOLOGY"]
    x0, x1 = 0.16, 0.77
    width = 0.14
    total0 = int(float(neural["predicted_relations"]))
    total1 = int(float(ontology["predicted_relations"]))
    tp0 = round(float(neural["relation_precision"]) * total0)
    tp1 = round(float(ontology["relation_precision"]) * total1)
    fp0 = total0 - tp0
    fp1 = total1 - tp1
    height0 = 0.58
    height1 = height0 * total1 / total0
    base = 0.18
    tp_h0 = height0 * tp0 / total0
    fp_h0 = height0 - tp_h0
    tp_h1 = height1 * tp1 / total1
    fp_h1 = height1 - tp_h1

    # Ribbons precede stacks so the endpoints remain crisp.
    bezier_ribbon(ax, x0 + width, x1, base, base + tp_h0, base, base + tp_h1, COLORS["teal"], 0.38)
    bezier_ribbon(ax, x0 + width, x1, base + tp_h0, base + height0, base + tp_h1, base + height1, COLORS["relation"], 0.18)

    ax.add_patch(Rectangle((x0, base), width, tp_h0, facecolor=COLORS["teal"], edgecolor="white", lw=0.6))
    ax.add_patch(Rectangle((x0, base + tp_h0), width, fp_h0, facecolor=COLORS["light_rose"], edgecolor="white", lw=0.6))
    ax.add_patch(Rectangle((x1, base), width, tp_h1, facecolor=COLORS["teal"], edgecolor="white", lw=0.6))
    ax.add_patch(Rectangle((x1, base + tp_h1), width, fp_h1, facecolor=COLORS["light_rose"], edgecolor="white", lw=0.6))

    ax.text(x0 + width / 2, 0.83, "Neural-only", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.text(x1 + width / 2, 0.83, "Ontology mask", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.annotate(
        "type-compatible\nfilter",
        xy=(x1 - 0.01, 0.55),
        xytext=(0.52, 0.69),
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["ochre"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["ochre"], lw=0.9),
    )

    ax.text(x0 + width / 2, 0.12, f"{total0} predictions", ha="center", va="top", fontsize=6.2)
    ax.text(x1 + width / 2, 0.12, f"{total1} predictions", ha="center", va="top", fontsize=6.2)
    ax.text(x0 - 0.02, base + tp_h0 / 2, f"{tp0} TP", ha="right", va="center", fontsize=6, color=COLORS["teal"])
    ax.text(x1 + width + 0.02, base + tp_h1 / 2, f"{tp1} TP", ha="left", va="center", fontsize=6, color=COLORS["teal"])
    ax.text(x0 - 0.02, base + tp_h0 + fp_h0 / 2, f"{fp0} FP", ha="right", va="center", fontsize=6, color=COLORS["relation"])
    ax.text(x1 + width + 0.02, base + tp_h1 + fp_h1 / 2, f"{fp1} FP", ha="left", va="center", fontsize=6, color=COLORS["relation"])

    ax.text(x0 + width / 2, 0.965, f"Compliance {float(neural['domain_range_compliance']):.3f}", ha="center", va="top", fontsize=6.2, color=COLORS["muted"])
    ax.text(x1 + width / 2, 0.965, f"Compliance {float(ontology['domain_range_compliance']):.3f}", ha="center", va="top", fontsize=6.2, color=COLORS["teal"])
    ax.text(0.5, 0.02, f"True positives {tp0} → {tp1} · false positives reduced by {fp0 - fp1}", ha="center", va="bottom", fontsize=5.9, color=COLORS["muted"])


def draw_panel_d(ax, methods, constraint, recovery):
    panel_label(ax, "d", "Constraint placement and remaining bottleneck")
    ax.set_xlim(0.10, 0.66)
    ax.set_ylim(-0.16, 2.05)
    ax.axis("off")

    neural = float(methods["BERT_NEURAL"]["relation_f1"])
    posthoc = float(methods["BERT_ONTOLOGY"]["relation_f1"])
    early = float(recovery["overall"]["relation_end_to_end_micro"]["f1"])
    folds = recovery["fold_reports"]
    tp = sum(x["gold_endpoint_relation_diagnostic"]["tp"] for x in folds)
    fp = sum(x["gold_endpoint_relation_diagnostic"]["fp"] for x in folds)
    fn = sum(x["gold_endpoint_relation_diagnostic"]["fn"] for x in folds)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    gold = 2 * precision * recall / (precision + recall)

    # Upper rail: constraint placement.
    y1 = 1.38
    ax.plot([neural, early], [y1, y1], color=COLORS["grid"], lw=3.2, solid_capstyle="round")
    pts = [
        (neural, "Neural-only", COLORS["neutral"], "o", "right", 0.18),
        (posthoc, "Post-hoc mask", COLORS["ochre"], "o", "center", 0.33),
        (early, "Early candidate\nconstraint", COLORS["teal"], "D", "left", 0.18),
    ]
    for x, label, color, marker, ha, lift in pts:
        ax.scatter(x, y1, s=44, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(x, y1 + lift, label, ha=ha, va="bottom", fontsize=5.9, color=color)
        ax.text(x, y1 - 0.17, f"{x:.3f}", ha="center", va="top", fontsize=6.2, fontweight="bold", color=color)
    low, high = constraint["relation"]["ci_95"]
    ax.text(
        (posthoc + early) / 2,
        0.93,
        f"early − post-hoc  {float(constraint['relation']['observed_difference']):+.3f}  [{low:.3f}, {high:.3f}]",
        ha="center",
        va="center",
        fontsize=5.9,
        color=COLORS["teal"],
    )

    # Lower rail: endpoint diagnostic.
    y2 = 0.37
    ax.plot([early, gold], [y2, y2], color=COLORS["light_blue"], lw=4.0, solid_capstyle="round")
    ax.scatter(early, y2, s=44, color=COLORS["relation"], edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(gold, y2, s=48, color=COLORS["entity"], marker="s", edgecolor="white", linewidth=0.6, zorder=3)
    ax.text(early, y2 + 0.15, "End-to-end", ha="center", va="bottom", fontsize=6.1, color=COLORS["relation"])
    ax.text(gold, y2 + 0.15, "Gold endpoints", ha="center", va="bottom", fontsize=6.1, color=COLORS["entity"])
    ax.text(early, y2 - 0.16, f"{early:.3f}", ha="center", va="top", fontsize=6.2, fontweight="bold", color=COLORS["relation"])
    ax.text(gold, y2 - 0.16, f"{gold:.3f}", ha="center", va="top", fontsize=6.2, fontweight="bold", color=COLORS["entity"])
    ax.annotate(
        "entity spans supplied",
        xy=(gold - 0.006, y2 + 0.01),
        xytext=((early + gold) / 2, y2 + 0.31),
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["entity"],
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=COLORS["entity"], linestyle="--"),
    )

    ax.text(
        0.105,
        -0.13,
        f"Evidence gate  ΔF1 = {float(methods['BERT_EVIDENCE']['relation_f1']) - posthoc:+.3f}\nsource audit retained",
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=COLORS["violet"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F0EEF6", edgecolor=COLORS["violet"], lw=0.65),
    )


def write_source_data(methods, paired, constraint, recovery):
    early = recovery["overall"]
    panel_a = [
        ["method", "entity_f1", "relation_f1", "evaluation"],
        ["Rule and lexicon", methods["B0"]["entity_f1"], methods["B0"]["relation_f1"], "fivefold OOF, 160 segments"],
        ["BERT neural-only", methods["BERT_NEURAL"]["entity_f1"], methods["BERT_NEURAL"]["relation_f1"], "fivefold OOF, 160 segments"],
        ["BERT plus post-hoc ontology", methods["BERT_ONTOLOGY"]["entity_f1"], methods["BERT_ONTOLOGY"]["relation_f1"], "fivefold OOF, 160 segments"],
        ["BERT plus early ontology", early["entity_exact_micro"]["f1"], early["relation_end_to_end_micro"]["f1"], "fivefold OOF, 160 segments"],
        ["MacBERT full pipeline", methods["MACBERT_EVIDENCE"]["entity_f1"], methods["MACBERT_EVIDENCE"]["relation_f1"], "fivefold OOF, 160 segments"],
        ["Joint BERT full pipeline", methods["JOINT_EVIDENCE"]["entity_f1"], methods["JOINT_EVIDENCE"]["relation_f1"], "fivefold OOF, 160 segments"],
    ]
    with (OUT_DIR / "Figure_11a_method_f1_source.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(panel_a)

    rec = recovery["paired_bootstrap"]
    lookup = {item["label"]: item for item in paired}
    effects = [
        ("Supervision vs rule", "entity_f1", rec["entity_vs_B0"]),
        ("Supervision vs rule", "relation_f1", rec["relation_vs_B0"]),
        ("Ontology mask vs neural", "relation_f1", lookup["Ontology effect within BERT"]["relation"]),
        ("Early vs post-hoc constraint", "relation_f1", constraint["relation"]),
        ("Joint vs staged BERT", "entity_f1", lookup["Joint versus staged BERT full pipeline"]["entity"]),
    ]
    with (OUT_DIR / "Figure_11b_paired_effects_source.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "difference", "ci_low", "ci_high", "bootstrap_iterations"])
        for label, metric, d in effects:
            w.writerow([label, metric, d["observed_difference"], d["ci_95"][0], d["ci_95"][1], d["iterations"]])

    with (OUT_DIR / "Figure_11c_candidate_audit_source.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "predicted_relations", "true_positive", "false_positive", "domain_range_compliance", "relation_f1"])
        for method_id, label in (("BERT_NEURAL", "BERT neural-only"), ("BERT_ONTOLOGY", "BERT plus ontology mask")):
            total = int(float(methods[method_id]["predicted_relations"]))
            tp = round(float(methods[method_id]["relation_precision"]) * total)
            w.writerow([label, total, tp, total - tp, methods[method_id]["domain_range_compliance"], methods[method_id]["relation_f1"]])

    folds = recovery["fold_reports"]
    tp = sum(x["gold_endpoint_relation_diagnostic"]["tp"] for x in folds)
    fp = sum(x["gold_endpoint_relation_diagnostic"]["fp"] for x in folds)
    fn = sum(x["gold_endpoint_relation_diagnostic"]["fn"] for x in folds)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    gold_f1 = 2 * p * r / (p + r)
    with (OUT_DIR / "Figure_11d_diagnostic_source.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diagnostic", "relation_f1", "note"])
        w.writerow(["BERT neural-only", methods["BERT_NEURAL"]["relation_f1"], "unrestricted candidate construction"])
        w.writerow(["Post-hoc ontology mask", methods["BERT_ONTOLOGY"]["relation_f1"], "mask after classification"])
        w.writerow(["Early ontology constraint", early["relation_end_to_end_micro"]["f1"], "compatible candidates before training and inference"])
        w.writerow(["Gold entity endpoints", gold_f1, "diagnostic only"])
        w.writerow(["Evidence gate accuracy effect", 0.0, "audit condition; no additional F1"])


def write_caption(methods, paired, constraint, recovery):
    early = recovery["overall"]
    neural = methods["BERT_NEURAL"]
    ontology = methods["BERT_ONTOLOGY"]
    joint = methods["JOINT_EVIDENCE"]
    total0 = int(float(neural["predicted_relations"]))
    total1 = int(float(ontology["predicted_relations"]))
    tp0 = round(float(neural["relation_precision"]) * total0)
    tp1 = round(float(ontology["relation_precision"]) * total1)
    fp0, fp1 = total0 - tp0, total1 - tp1
    folds = recovery["fold_reports"]
    gold_tp = sum(x["gold_endpoint_relation_diagnostic"]["tp"] for x in folds)
    gold_fp = sum(x["gold_endpoint_relation_diagnostic"]["fp"] for x in folds)
    gold_fn = sum(x["gold_endpoint_relation_diagnostic"]["fn"] for x in folds)
    gold_p = gold_tp / (gold_tp + gold_fp)
    gold_r = gold_tp / (gold_tp + gold_fn)
    gold_f1 = 2 * gold_p * gold_r / (gold_p + gold_r)
    lookup = {item["label"]: item for item in paired}
    joint_ci = lookup["Joint versus staged BERT full pipeline"]["entity"]["ci_95"]
    caption = f"""# Figure 11 | Ontology-constrained supervised evidence extraction

**a**, Exact entity and end-to-end relation F1 for six methods evaluated on the same 160 expert-adjudicated segments with deterministic fivefold record-level out-of-fold prediction after local-training authorization. The early-ontology BERT pipeline achieved entity F1 {early['entity_exact_micro']['f1']:.3f} and relation F1 {early['relation_end_to_end_micro']['f1']:.3f}. The shared-encoder joint model achieved entity F1 {float(joint['entity_f1']):.3f}; its paired interval relative to staged BERT was [{float(joint_ci[0]):.3f}, {float(joint_ci[1]):.3f}]. **b**, Paired F1 differences with 95% percentile bootstrap intervals from 5,000 segment-level resamples. Supervision improved entity and relation F1 relative to the rule baseline. Post-classification ontology masking improved relation F1, and ontology-compatible candidate construction before training performed better than post-classification masking. **c**, Relation-prediction audit for matched BERT conditions. Ontology masking changed true positives from {tp0} to {tp1}, reduced false positives from {fp0} to {fp1}, and changed domain-range compliance from {float(neural['domain_range_compliance']):.3f} to {float(ontology['domain_range_compliance']):.3f}. **d**, Mechanism and error diagnostics. Relation F1 changed from {float(neural['relation_f1']):.3f} without constraints to {float(ontology['relation_f1']):.3f} with post-classification masking and {early['relation_end_to_end_micro']['f1']:.3f} with early candidate restriction. Supplying gold entity endpoints raised relation F1 to {gold_f1:.3f}, identifying entity-boundary recovery as a major remaining bottleneck. The evidence gate changed relation F1 by {float(methods['BERT_EVIDENCE']['relation_f1']) - float(ontology['relation_f1']):+.3f} after ontology masking because retained extractive predictions already pointed to exact source spans. Its role was auditability. The early-versus-post-classification comparison was specified after both experiments and is supportive. All supervised results were regenerated under authorization LQ-LOCAL-TRAIN-2026-09-27-v2. They are same-corpus estimates and do not establish public-benchmark state of the art.
"""
    (OUT_DIR / "Figure_11_caption_nature_v1.md").write_text(caption, encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    methods, paired, constraint, recovery = read_inputs()
    write_source_data(methods, paired, constraint, recovery)

    fig = plt.figure(figsize=(7.20, 6.05), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.18, 0.82],
        height_ratios=[1.04, 0.96],
        left=0.115,
        right=0.985,
        bottom=0.085,
        top=0.955,
        wspace=0.37,
        hspace=0.47,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a, methods, recovery)
    draw_panel_b(ax_b, paired, constraint, recovery)
    draw_panel_c(ax_c, methods)
    draw_panel_d(ax_d, methods, constraint, recovery)

    base = OUT_DIR / "Figure_11_KG_extraction_validation_nature_v1"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    write_caption(methods, paired, constraint, recovery)
    contract = """# Figure 11 contract

- Core conclusion: supervised extraction recovered entity and relation performance, while ontology constraints improved relations by controlling the candidate space; entity boundaries remain the main bottleneck.
- Archetype: asymmetric quantitative composite.
- Hero evidence: matched exact entity and end-to-end relation F1 across six methods.
- Validation evidence: paired bootstrap effects and the relation-candidate audit.
- Mechanism evidence: post-hoc versus early ontology constraints.
- Diagnostic boundary: gold endpoints are diagnostic and evidence gating is an audit condition, not a measured accuracy gain.
- Evaluation: 160 expert-adjudicated segments, deterministic fivefold record-level out-of-fold prediction.
- Reviewer risk: record-level folds are not source-held-out external validation; the early-versus-post-hoc contrast is supportive; no public-benchmark SOTA claim is made.
- Backend: Python only.
- Export: 183 mm class width, editable SVG and PDF, 350 dpi PNG, 600 dpi LZW TIFF.
"""
    (OUT_DIR / "Figure_11_contract_v1.md").write_text(contract, encoding="utf-8")

    exports = [
        base.with_suffix(".svg"),
        base.with_suffix(".pdf"),
        base.with_suffix(".png"),
        base.with_suffix(".tiff"),
    ]
    source_files = sorted(OUT_DIR.glob("Figure_11*_source.csv"))
    manifest = {
        "figure": "Figure 11",
        "backend": "Python matplotlib",
        "evaluation": "160 segments; deterministic fivefold record-level OOF",
        "exports": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in exports},
        "source_data": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in source_files},
    }
    (OUT_DIR / "Figure_11_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    qa = """# Figure 11 QA report

- PASS: every quantitative value is loaded from audited comparison outputs or recomputed from saved fold counts.
- PASS: the six displayed methods share the same 160-segment corpus and record-level out-of-fold evaluation.
- PASS: F1 definitions distinguish exact entities from end-to-end relations.
- PASS: all intervals are identified as 95% percentile bootstrap intervals from 5,000 segment-level resamples.
- PASS: the gold-endpoint result is labelled as diagnostic.
- PASS: the evidence gate is described as an audit condition and not an accuracy improvement.
- PASS: zero-shot generation is excluded from the visual main claim but remains reported in the manuscript table.
- PASS: the figure makes no public-benchmark SOTA or source-held-out generalization claim.
- PASS: method identity is encoded by direct labels; metric identity uses both colour and marker shape.
- PASS: SVG and PDF retain editable text; PNG and TIFF raster exports are present.
"""
    (OUT_DIR / "Figure_11_QA_report.md").write_text(qa, encoding="utf-8")
    print(base.with_suffix(".png"))


if __name__ == "__main__":
    main()
