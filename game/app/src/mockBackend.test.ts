import { describe, expect, it } from "vitest";

import { createInitialState, gameReducer } from "./gameState";
import { buildMockDecision } from "./mockBackend";
import type { IntentCard } from "./types";

const intents: IntentCard[] = [
  {
    intentId: "intent-repeat-pattern",
    label: "相同图案",
    statement: "保留相同图案。",
    priority: 3,
    locked: true,
    confirmedAt: "2026-09-17T00:00:00.000Z",
  },
  {
    intentId: "intent-subject-brighter",
    label: "主体更亮",
    statement: "保留主体更亮。",
    priority: 2,
    locked: true,
    confirmedAt: "2026-09-17T00:00:00.000Z",
  },
];

function preparedState(condition: "b4" | "ensr" | "iper") {
  let state = createInitialState(condition);
  state = gameReducer(state, {
    type: "CONFIRM_WISH",
    commissionId: "friend",
    intents,
  });
  state = gameReducer(state, {
    type: "SELECT_MATERIAL",
    materialId: "material-warm-translucent",
    kgNodeId: "task:task_myl_needle_piercing_v0",
    knowledgeStatus: "task_approved",
  });
  state = gameReducer(state, {
    type: "SELECT_PATTERN",
    patternId: "pattern-geometric-rhythm",
  });
  state = gameReducer(state, {
    type: "SELECT_TOOLSET",
    toolIds: ["tool-freehand-stylus", "tool-fine-awl", "tool-edge-clips"],
  });
  return gameReducer(state, {
    type: "SELECT_REFERENCE",
    referenceId: "ref-light-branch",
  });
}

describe("buildMockDecision", () => {
  it("keeps legacy IPER development sessions readable", () => {
    const decision = buildMockDecision(preparedState("iper"));
    expect(decision.decision_type).toBe("propose");
    expect(decision.plans).toHaveLength(1);
    expect(decision.mock_backend).toBe(true);
  });

  it("uses the evidence-grounded path for the ENSR condition", () => {
    const decision = buildMockDecision(preparedState("ensr"));
    expect(decision.decision_type).toBe("propose");
    expect(decision.plans).toHaveLength(1);
    expect(decision.plans[0].action_ids).toEqual([
      "select_repeat_pattern_plan",
      "select_positive_piercing_region",
      "prepare_controlled_local_sample",
      "pierce_sample_with_board",
      "explain_pattern_and_light_relation",
    ]);
  });

  it("never proposes changing a locked intent", () => {
    const decision = buildMockDecision(preparedState("iper"));
    for (const plan of decision.plans) {
      expect(plan.proposed_changes).toEqual([]);
      expect(plan.retained_intent_ids).toEqual(
        expect.arrayContaining(intents.map((intent) => intent.intentId)),
      );
    }
  });

  it("returns a clarification when no reference has been selected", () => {
    const decision = buildMockDecision(createInitialState("b4"));
    expect(decision.decision_type).toBe("clarify");
    expect(decision.plans).toEqual([]);
    expect(decision.unresolved_conditions).toEqual([
      "state:selected_material",
      "state:selected_pattern",
      "state:selected_toolset",
      "state:selected_reference",
    ]);
  });

  it("carries the chosen material and player pattern into the plan", () => {
    const plan = buildMockDecision(preparedState("ensr")).plans[0];
    expect(plan.summary).toContain("暖白半透纸样");
    expect(plan.summary).toContain("几何连续纹");
    expect(plan.summary).toContain("自由纹样笔");
    expect(plan.effects).toMatchObject({
      material: "material-warm-translucent",
      pattern: "pattern-geometric-rhythm",
      tools: ["tool-freehand-stylus", "tool-fine-awl", "tool-edge-clips"],
    });
  });

  it("offers a preserve-work path only in the ENSR change condition", () => {
    const ensr = {
      ...preparedState("ensr"),
      revisionStage: "change_active" as const,
    };
    const b4 = {
      ...preparedState("b4"),
      revisionStage: "change_active" as const,
    };
    expect(buildMockDecision(ensr).plans.map((plan) => plan.rework_cost)).toEqual([0, 1]);
    expect(buildMockDecision(b4).plans.map((plan) => plan.rework_cost)).toEqual([1]);
  });
});
