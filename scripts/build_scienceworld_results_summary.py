from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "benchmarks" / "scienceworld"
OUTPUT_JSON = ARTIFACTS / "paper_results_summary_v1.json"
OUTPUT_MD = ROOT / "docs" / "scienceworld_results_summary_v1.md"


def load_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8-sig"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def bool_text(value: bool) -> str:
    return "通过" if value else "未通过"


def score_ci(effect: dict) -> str:
    return (
        f"{effect['point_estimate']:+.2f} "
        f"[{effect['ci95_low']:.2f}, {effect['ci95_high']:.2f}]"
    )


def success_ci(effect: dict) -> str:
    return (
        f"{100 * effect['point_estimate']:+.2f} pp "
        f"[{100 * effect['ci95_low']:.2f}, {100 * effect['ci95_high']:.2f}]"
    )


def external_failures(summary: dict) -> int:
    return int(
        summary.get(
            "external_failures", summary.get("provider_or_environment_failures", 0)
        )
    )


def mechanism_conclusion(comparison: dict) -> str:
    adjusted_p = comparison["score_permutation"]["holm_adjusted_p_value"]
    if comparison["score_ci_excludes_zero"] and adjusted_p < 0.05:
        return "严格支持"
    if comparison["score_ci_excludes_zero"]:
        return "区间支持；多重校正后边缘"
    return "仅方向性证据"


def main() -> None:
    e13 = load_json("ensr_v3_test_audit.json")
    kg = load_json("ensr_v3_ablation_no_semantic_kg_audit.json")
    e14 = load_json("ensr_cross_model_v1_audit.json")
    sensitivity = load_json("ensr_cross_model_v1_sensitivity.json")
    e15 = load_json("ensr_clean_adapter_v1_audit.json")
    e15r = load_json("qwen3_8b_recovery_v1_audit.json")
    e16 = load_json("ensr_mechanism_ablation_v1_audit.json")
    efficiency = load_json("ensr_efficiency_summary_v1.json")

    source_files = {
        "e13": "ensr_v3_test_audit.json",
        "kg_ablation": "ensr_v3_ablation_no_semantic_kg_audit.json",
        "e14": "ensr_cross_model_v1_audit.json",
        "e14_sensitivity": "ensr_cross_model_v1_sensitivity.json",
        "e15": "ensr_clean_adapter_v1_audit.json",
        "e15r": "qwen3_8b_recovery_v1_audit.json",
        "e16": "ensr_mechanism_ablation_v1_audit.json",
        "efficiency": "ensr_efficiency_summary_v1.json",
    }
    summary = {
        "schema_version": "2.0",
        "generated_from": source_files,
        "e13": {
            "complete": e13["complete_matrix"],
            "episodes": e13["observed_episode_count"],
            "formal_full_success": e13["formal_full_success"],
            "summaries": e13["summaries"],
            "ensr_minus_iper": e13["ensr_minus_baseline"],
        },
        "kg_ablation": {
            "complete": kg["complete_pairing"],
            "reporting_boundary": kg["reporting_boundary"],
            "without_semantic_kg": kg["old_summary"],
            "with_semantic_kg": kg["new_summary"],
            "with_minus_without": kg["new_minus_old"],
        },
        "e14": {
            "complete": e14["complete_all_models"],
            "episodes": e14["expected_episode_count"],
            "robustness_summary": e14["robustness_summary"],
            "models": e14["models"],
        },
        "e14_sensitivity": sensitivity,
        "e15": {
            "complete": e15["complete_all_profiles"],
            "episodes": e15["observed_episode_count"],
            "models": e15["models"],
            "pooled_ensr_minus_iper": e15["pooled_ensr_minus_iper"],
            "clean_adapter_summary": e15["clean_adapter_summary"],
            "reporting_boundary": e15["reporting_boundary"],
        },
        "e15r": e15r,
        "e16": {
            "complete": e16["complete_matrix"],
            "episodes": e16["observed_episode_count"],
            "implementation_match": e16["implementation_match"],
            "operational_pass": e16["operational_pass"],
            "arms": e16["arms"],
            "comparisons": e16["comparisons"],
            "interpretation_boundary": e16["interpretation_boundary"],
        },
        "efficiency": {
            "e16_status": efficiency["e16_status"],
            "currency_cost_status": efficiency["currency_cost_status"],
            "rows": efficiency["rows"],
            "paired_system_comparisons": efficiency["paired_system_comparisons"],
        },
        "next_required_work": {
            "name": "本地多模态知识图谱基准冻结（E20）",
            "status": "等待来源授权与领域专家标注规则",
            "reason": (
                "ScienceWorld 的主效应、跨模型、适配器恢复和机制消融均已完成；"
                "下一阶段必须用有权使用的宣纸图像、文本与专家标注建立本地多模态 KG 基准。"
            ),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# ScienceWorld 已完成实验结果摘要 v2",
        "",
        "本文件由 `scripts/build_scienceworld_results_summary.py` 从冻结审计文件生成。",
        "",
        "## E13：全新留出集正式验证",
        "",
        f"完整矩阵：{e13['observed_episode_count']}/{e13['expected_episode_count']}；"
        f"正式门槛：{bool_text(e13['formal_full_success'])}。",
        "",
        "| 方法 | 平均截断分 | 成功率 | 模型调用 | 输入 tokens | 外部失败 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("b4_sequential_planner", "iper_rag", "ensr_v2"):
        item = e13["summaries"][method]
        lines.append(
            f"| {method} | {item['mean_clipped_score']:.2f} | "
            f"{pct(item['task_success_rate'])} | {item['model_calls']:,} | "
            f"{item['input_tokens']:,} | {external_failures(item)} |"
        )
    e13_score = e13["ensr_minus_baseline"]["clipped_score"]
    e13_success = e13["ensr_minus_baseline"]["task_success_rate"]
    lines.extend(
        [
            "",
            f"ENSR − IPER 分数差：**{score_ci(e13_score)}**。",
            "",
            f"ENSR − IPER 成功率差：**{success_ci(e13_success)}**。",
            "",
            "## 语义 KG 关键消融",
            "",
            f"配对完整：{bool_text(kg['complete_pairing'])}；每个版本 "
            f"{kg['expected_episode_count_per_version']} 个 episode。报告边界："
            f"`{kg['reporting_boundary']}`。",
            "",
            "| 条件 | 平均截断分 | 成功数 | 成功率 |",
            "|---|---:|---:|---:|",
            f"| 关闭语义 KG | {kg['old_summary']['mean_clipped_score']:.2f} | "
            f"{kg['old_summary']['success_count']} | {pct(kg['old_summary']['success_rate'])} |",
            f"| 开启语义 KG | {kg['new_summary']['mean_clipped_score']:.2f} | "
            f"{kg['new_summary']['success_count']} | {pct(kg['new_summary']['success_rate'])} |",
            "",
            "开启 − 关闭分数差：**"
            + score_ci(kg["new_minus_old"]["mean_clipped_score"])
            + "**。",
            "",
            "开启 − 关闭成功率差：**"
            + success_ci(kg["new_minus_old"]["success_rate"])
            + "**。",
            "",
            "## E14：五模型跨模型稳健性",
            "",
            f"完整矩阵：{e14['expected_episode_count']}/{e14['expected_episode_count']}；"
            f"预注册稳健性门槛：{bool_text(e14['robustness_summary']['cross_model_robustness_pass'])}；"
            f"满足方向标准 {e14['robustness_summary']['both_criteria_count']}/"
            f"{e14['profile_count']} 个模型。",
            "",
            "| 模型 | IPER 分数 | ENSR 分数 | 分数差 [95% CI] | 成功率差 [95% CI] | "
            "ENSR 外部失败 | 单模型正式门槛 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_id, model in e14["models"].items():
        iper = model["summaries"]["iper_rag"]
        ensr = model["summaries"]["ensr_v2"]
        effects = model["ensr_minus_baseline"]
        lines.append(
            f"| {model_id} | {iper['mean_clipped_score']:.2f} | "
            f"{ensr['mean_clipped_score']:.2f} | "
            f"{score_ci(effects['clipped_score'])} | "
            f"{success_ci(effects['task_success_rate'])} | "
            f"{external_failures(ensr)} | {bool_text(model['formal_full_success'])} |"
        )

    lines.extend(
        [
            "",
            "## E15：统一适配器跨模型扩展",
            "",
            f"完整矩阵：{e15['observed_episode_count']}/{e15['expected_episode_count']}；"
            f"整体门槛：{bool_text(e15['clean_adapter_summary']['clean_adapter_pass'])}（"
            f"{e15['clean_adapter_summary']['passing_profile_count']}/"
            f"{e15['clean_adapter_summary']['required_profile_count']} 个模型通过）。",
            "",
            "| 模型 | IPER 分数 | ENSR 分数 | 分数差 [95% CI] | 成功率差 [95% CI] | "
            "IPER/ENSR 外部失败 | 单模型门槛 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model_id, model in e15["models"].items():
        iper = model["summaries"]["iper_rag"]
        ensr = model["summaries"]["ensr_v2"]
        effects = model["ensr_minus_iper"]
        lines.append(
            f"| {model_id} | {iper['mean_clipped_score']:.2f} | "
            f"{ensr['mean_clipped_score']:.2f} | "
            f"{score_ci(effects['clipped_score'])} | "
            f"{success_ci(effects['task_success_rate'])} | "
            f"{external_failures(iper)}/{external_failures(ensr)} | "
            f"{bool_text(model['profile_pass'])} |"
        )
    pooled = e15["pooled_ensr_minus_iper"]
    lines.extend(
        [
            "",
            f"三模型合并 ENSR − IPER 分数差：**{score_ci(pooled['clipped_score'])}**；"
            f"成功率差：**{success_ci(pooled['task_success_rate'])}**。",
            "",
            "Qwen3-8B 因 IPER 出现 3 次外部失败，超过预注册上限 1，故整体操作门槛未通过；"
            "该失败不会被后续恢复实验替换。",
            "",
            "## E15R：Qwen3-8B 前瞻性恢复实验",
            "",
            f"完整矩阵：{e15r['observed_episode_count']}/{e15r['expected_episode_count']}；"
            f"方向门槛：{bool_text(e15r['direction_pass'])}；"
            f"操作门槛：{bool_text(e15r['operational_pass'])}；"
            f"恢复门槛：{bool_text(e15r['recovery_pass'])}。",
            "",
            f"ENSR − IPER 分数差：**{score_ci(e15r['ensr_minus_iper']['clipped_score'])}**；"
            f"成功率差：**{success_ci(e15r['ensr_minus_iper']['task_success_rate'])}**。",
            "",
            "IPER 仍有 4 次外部失败，因此 E15R 提供正向性能证据，但没有修复操作可靠性结论。",
            "",
            "## E16：神经符号机制消融",
            "",
            f"完整矩阵：{e16['observed_episode_count']}/{e16['expected_episode_count']}；"
            f"实现哈希一致：{bool_text(e16['implementation_match'])}；"
            f"操作审计：{bool_text(e16['operational_pass'])}。",
            "",
            "| 机制条件 | 平均截断分 | 成功率 | 模型调用 | 检索调用 | 每 100 次模型调用成功数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    arm_labels = {
        "full_control": "完整 ENSR",
        "without_hierarchical_subgoals": "移除层级子目标",
        "without_symbolic_transition_verifier": "移除符号转移验证器",
        "without_obligation_conditioned_retrieval": "移除义务条件检索",
        "accumulating_fragment_memory": "累积式片段记忆（对照替换式）",
    }
    for arm_id, arm in e16["arms"].items():
        lines.append(
            f"| {arm_labels[arm_id]} | {arm['mean_clipped_score']:.2f} | "
            f"{pct(arm['task_success_rate'])} | {arm['model_calls']} | "
            f"{arm['retrieval_calls']} | {arm['successes_per_100_model_calls']:.2f} |"
        )

    comparison_labels = {
        "full_control_minus_without_hierarchical_subgoals": "层级子目标",
        "full_control_minus_without_symbolic_transition_verifier": "符号转移验证器",
        "full_control_minus_without_obligation_conditioned_retrieval": "义务条件检索",
        "full_control_minus_accumulating_fragment_memory": "替换式片段记忆",
    }
    lines.extend(
        [
            "",
            "| 被检验机制 | 完整版 − 消融版分数 [95% CI] | 成功率差 [95% CI] | "
            "原始 p | Holm p | 结论 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for comparison_id, comparison in e16["comparisons"].items():
        permutation = comparison["score_permutation"]
        lines.append(
            f"| {comparison_labels[comparison_id]} | "
            f"{score_ci(comparison['clipped_score'])} | "
            f"{success_ci(comparison['task_success_rate'])} | "
            f"{permutation['p_value']:.4f} | "
            f"{permutation['holm_adjusted_p_value']:.4f} | "
            f"{mechanism_conclusion(comparison)} |"
        )

    lines.extend(
        [
            "",
            "符号转移验证器在多重比较校正后仍有严格支持；层级子目标的 bootstrap 区间不跨 0，"
            "但 Holm 校正后为边缘结果；义务条件检索和替换式片段记忆目前只支持方向性解释。",
            "",
            "## 效率证据",
            "",
            "E13 中，ENSR 用 273 次模型调用获得 29 次成功，IPER 用 2,110 次调用获得 7 次成功；"
            "ENSR 的成功/100 次调用分别为 10.62 和 0.33。E14、E15 和 E15R 的配对结果也显示，"
            "ENSR 在提高分数与成功数的同时，大多减少模型调用与总 token。货币成本没有估算，"
            "因为学校 MaaS 实验记录中没有冻结的模型级费率。",
            "",
            "## 当前证据边界",
            "",
            "- E13 是正式确认性结果，支持 ENSR 优于冻结 IPER 基线的主结论。",
            "- 开发集语义 KG 消融支持 KG 组件有效，但不能替代新的确认性测试。",
            "- E14 支持五模型方向稳健性；完整案例敏感性分析只在 "
            f"{sensitivity['summary']['clean_direction_preserved_count']}/"
            f"{sensitivity['summary']['profile_count']} 个模型保留方向。",
            "- E15 与 E15R 的性能方向均为正，但 Qwen3-8B 的操作可靠性门槛失败必须保留报告。",
            "- E16 给出了神经符号机制的因果证据，其中符号转移验证器证据最强。",
            "- ScienceWorld 机器实验阶段已经完成；不能把开发集消融或适配器失败隐藏成全模型 SOTA。",
            "",
            "## 下一阶段",
            "",
            "进入 E20 本地多模态知识图谱基准冻结。需要有权使用的宣纸图像与文本、来源清单、"
            "专家标注规范和数据划分；在这些输入到位前，可完成数据契约、标注模板、质检脚本和预注册协议，"
            "但不能生成或伪造正式实验数据。",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
