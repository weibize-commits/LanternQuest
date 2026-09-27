import type { DecisionResponse, GameState, PlanOption } from "./types";
import { MATERIALS, PATTERNS, TOOLS } from "./data";

function version(prefix: string, value: number): string {
  return `${prefix}-${value}`;
}

function actionForIntent(state: GameState, dimension: "pattern" | "light"): string {
  const ids = new Set(state.intents.map((intent) => intent.intentId));
  if (dimension === "pattern") {
    return ids.has("intent-vary-patterns")
      ? "select_vary_pattern_plan"
      : "select_repeat_pattern_plan";
  }
  return ids.has("intent-background-brighter")
    ? "select_negative_piercing_region"
    : "select_positive_piercing_region";
}

function groundedPlan(state: GameState): PlanOption {
  const patternAction = actionForIntent(state, "pattern");
  const lightAction = actionForIntent(state, "light");
  const varying = patternAction === "select_vary_pattern_plan";
  const background = lightAction === "select_negative_piercing_region";
  const material = MATERIALS.find(
    (item) => item.materialId === state.selectedMaterialId,
  );
  const pattern = PATTERNS.find(
    (item) => item.patternId === state.selectedPatternId,
  );
  const tools = TOOLS.filter((item) => state.selectedToolIds.includes(item.toolId));
  return {
    plan_id: "plan-controlled-local-sample",
    title: "完成受控的局部针刺透光样片",
    summary: `在${material?.label ?? "数字纸样"}上使用${pattern?.label ?? "玩家图案"}，拿起${tools.map((tool) => tool.label).join("、") || "三类制作工具"}亲手绘制、针刺和装片；采用${varying ? "不同图案逐片处理" : "相同图案叠放定位"}，并选择${
      background ? "背景透光" : "主体透光"
    }的针刺区域。`,
    action_ids: [
      patternAction,
      lightAction,
      "prepare_controlled_local_sample",
      "pierce_sample_with_board",
      "explain_pattern_and_light_relation",
    ],
    retained_intent_ids: state.intents.map((intent) => intent.intentId),
    proposed_changes: [],
    required_consent: [],
    evidence_ids: [
      "ev_n001_pattern_branch",
      "ev_n001_light_branch",
      "ev_n001_depth_control",
      "ev_field06_process_check",
      "ev_field07_complexity",
      "ev_field06_hands_on_boundary",
    ],
    unresolved_conditions: [],
    effects: {
      form: varying ? "vary_patterns" : "repeat_pattern",
      material: state.selectedMaterialId ?? "unselected",
      pattern: state.selectedPatternId ?? "unselected",
      tools: state.selectedToolIds,
      palette: background ? "background_brighter" : "subject_brighter",
      panelTreatment: "controlled_local_piercing_sample",
      assembly: "ready_to_show",
    },
    tradeoff: "只形成受控的局部体验结果，不能据此声称已经掌握完整万眼萝制作工艺。",
    rework_cost: 0,
  };
}

function revisionPlans(state: GameState): PlanOption[] {
  const retained = state.intents.map((intent) => intent.intentId);
  const overwrite: PlanOption = {
    plan_id: "plan-overwrite-existing-sample",
    title: "在现有样片上重做图案",
    summary: "沿用初始顺序路径，在现有样片上调整为新图案；已完成的针刺部分需要重做。",
    action_ids: ["overwrite_existing_sample"],
    retained_intent_ids: retained,
    proposed_changes: [],
    required_consent: [],
    evidence_ids: ["ev_n001_pattern_branch"],
    unresolved_conditions: [],
    effects: { panelTreatment: "overwritten_for_new_pattern", assembly: "ready_to_show" },
    tradeoff: "能够完成新图案，但会产生一项可避免返工。",
    rework_cost: 1,
  };
  if (state.condition !== "ensr") return [overwrite];
  const preserve: PlanOption = {
    plan_id: "plan-new-panel-preserve-work",
    title: "新增灯片并保留已完成样片",
    summary: "把已完成样片保留为现有成果，另备一张灯片逐片处理新图案。",
    action_ids: ["prepare_new_panel_and_preserve_work"],
    retained_intent_ids: retained,
    proposed_changes: [],
    required_consent: [],
    evidence_ids: ["ev_field06_hands_on_boundary", "ev_n001_pattern_branch"],
    unresolved_conditions: [],
    effects: { panelTreatment: "new_panel_preserving_completed_sample", assembly: "ready_to_show" },
    tradeoff: "增加一张局部样片的材料与操作，但不推翻已完成工作。",
    rework_cost: 0,
  };
  return [preserve, overwrite];
}

export function buildMockDecision(
  state: GameState,
  backendKind: "mock" | "precomputed_e21" = "mock",
): DecisionResponse {
  const isMock = backendKind === "mock";
  const unresolvedSelections = [
    !state.selectedMaterialId ? "state:selected_material" : null,
    !state.selectedPatternId ? "state:selected_pattern" : null,
    state.selectedToolIds.length !== 3 ? "state:selected_toolset" : null,
    !state.selectedReferenceId ? "state:selected_reference" : null,
  ].filter((value): value is string => Boolean(value));
  if (unresolvedSelections.length > 0) {
    return {
      decision_type: "clarify",
      state_version: version("state", state.stateVersion),
      intent_version: version("intent", state.intentVersion),
      plans: [],
      unresolved_conditions: unresolvedSelections,
      user_visible_explanation: "请先确认材料、玩家图案和专家审核参照，再核对制作路径。",
      mock_backend: isMock,
      backend_kind: backendKind,
    };
  }
  if (state.revisionStage === "change_active") {
    return {
      decision_type: "propose",
      state_version: version("state", state.stateVersion),
      intent_version: version("intent", state.intentVersion),
      plans: revisionPlans(state),
      unresolved_conditions: [],
      user_visible_explanation:
        state.condition === "ensr"
          ? "系统核对了新委托、已完成工作、锁定意图与证据范围，再组织可继续的路径。"
          : "系统沿用开始时固定取得的证据与顺序路径，更新了后续制作步骤。",
      mock_backend: isMock,
      backend_kind: backendKind,
    };
  }
  return {
    decision_type: "propose",
    state_version: version("state", state.stateVersion),
    intent_version: version("intent", state.intentVersion),
    plans: [groundedPlan(state)],
    unresolved_conditions: [],
    user_visible_explanation: "系统依据专家审核证据给出第一段局部样片路径。",
    mock_backend: isMock,
    backend_kind: backendKind,
  };
}
