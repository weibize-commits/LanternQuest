import { describe, expect, it } from "vitest";

import {
  conditionFromLocation,
  createInitialState,
  gameReducer,
  participantCodeFromLocation,
} from "./gameState";
import type { IntentCard, PlanOption } from "./types";

const confirmedAt = "2026-09-17T00:00:00.000Z";
const intents: IntentCard[] = [
  {
    intentId: "intent-locked",
    label: "锁定选择",
    statement: "保持当前选择。",
    priority: 3,
    locked: true,
    confirmedAt,
  },
  {
    intentId: "intent-flexible",
    label: "可商量选择",
    statement: "可以在确认后调整。",
    priority: 2,
    locked: false,
    confirmedAt,
  },
];

function stateWithIntents() {
  return gameReducer(createInitialState("iper"), {
    type: "CONFIRM_WISH",
    commissionId: "friend",
    intents,
    at: confirmedAt,
  });
}

describe("gameReducer", () => {
  it("reads blinded study arms and anonymous participant codes", () => {
    const search = "?arm=B&participant=LQ-999";
    expect(conditionFromLocation(search)).toBe("ensr");
    expect(participantCodeFromLocation(search)).toBe("LQ-999");
    expect(participantCodeFromLocation("?participant=name")).toBeNull();
  });

  it("freezes exactly two confirmed intents and increments their version", () => {
    const state = stateWithIntents();
    expect(state.intents).toEqual(intents);
    expect(state.intentVersion).toBe(1);
    expect(state.completedActs).toContain("wish");
    expect(state.events.at(-1)?.eventType).toBe("intent_confirmed");
  });

  it("records KG material and player pattern as versioned task state", () => {
    let state = stateWithIntents();
    state = gameReducer(state, {
      type: "SELECT_MATERIAL",
      materialId: "material-warm-translucent",
      kgNodeId: "task:task_myl_needle_piercing_v0",
      knowledgeStatus: "task_approved",
    });
    state = gameReducer(state, {
      type: "SELECT_PATTERN",
      patternId: "pattern-botanical-outline",
    });
    state = gameReducer(state, {
      type: "SELECT_TOOLSET",
      toolIds: ["tool-freehand-stylus", "tool-fine-awl", "tool-edge-clips"],
    });
    expect(state.selectedMaterialId).toBe("material-warm-translucent");
    expect(state.selectedPatternId).toBe("pattern-botanical-outline");
    expect(state.lantern.material).toBe("material-warm-translucent");
    expect(state.lantern.pattern).toBe("pattern-botanical-outline");
    expect(state.selectedToolIds).toHaveLength(3);
    expect(state.stateVersion).toBe(3);
    expect(state.events.slice(-3).map((event) => event.eventType)).toEqual([
      "kg_material_selected",
      "player_pattern_selected",
      "simulation_toolset_selected",
    ]);
  });

  it("persists hands-on craft checkpoints and their observable counts", () => {
    const state = gameReducer(stateWithIntents(), {
      type: "UPDATE_CRAFT_PROGRESS",
      reason: "pierce",
      progress: {
        phase: "pierce",
        strokes: 2,
        holes: 8,
        assembledPanels: 0,
        marks: [
          { x: 0.5, y: 0.5, kind: "hole", size: 5, strokeId: 1 },
        ],
        completedAt: null,
      },
    });
    expect(state.craftProgress).toMatchObject({ phase: "pierce", strokes: 2, holes: 8 });
    expect(state.events.at(-1)?.eventType).toBe("craft_pierce");
    expect(state.events.at(-1)?.payload).toMatchObject({ holes: 8, markCount: 1 });
  });

  it("blocks a plan that silently changes a locked intent", () => {
    const state = stateWithIntents();
    const plan: PlanOption = {
      plan_id: "bad-plan",
      title: "错误计划",
      summary: "尝试覆盖锁定意图。",
      action_ids: ["rewrite"],
      retained_intent_ids: [],
      proposed_changes: ["intent-locked"],
      required_consent: [],
      evidence_ids: [],
      unresolved_conditions: [],
      effects: { palette: "changed" },
      tradeoff: "无",
    };
    const next = gameReducer(state, { type: "APPLY_PLAN", plan });
    expect(next.stateVersion).toBe(state.stateVersion);
    expect(next.lantern.palette).toBe(state.lantern.palette);
    expect(next.events.at(-1)?.eventType).toBe("plan_blocked_locked_intent");
  });

  it("blocks a plan that omits a locked intent from the retained list", () => {
    const state = stateWithIntents();
    const plan: PlanOption = {
      plan_id: "omission-plan",
      title: "遗漏计划",
      summary: "没有声明保留锁定意图。",
      action_ids: ["continue"],
      retained_intent_ids: ["intent-flexible"],
      proposed_changes: [],
      required_consent: [],
      evidence_ids: [],
      unresolved_conditions: [],
      effects: { palette: "changed" },
      tradeoff: "无",
    };
    const next = gameReducer(state, { type: "APPLY_PLAN", plan });
    expect(next.stateVersion).toBe(state.stateVersion);
    expect(next.events.at(-1)?.payload).toMatchObject({
      blockedIntentIds: ["intent-locked"],
    });
  });

  it("turns off AI assistance in the independent explanation stage", () => {
    const state = gameReducer(stateWithIntents(), {
      type: "GO_TO_ACT",
      act: "showcase",
    });
    expect(state.aiAssistanceEnabled).toBe(false);
    expect(state.noAiStageStarted).toBe(true);
    expect(state.currentAct).toBe("showcase");

    const reviewState = gameReducer(state, {
      type: "GO_TO_ACT",
      act: "craft",
    });
    expect(reviewState.currentAct).toBe("craft");
    expect(reviewState.aiAssistanceEnabled).toBe(false);

    const blockedDecision = gameReducer(reviewState, {
      type: "RECEIVE_DECISION",
      decision: {
        decision_type: "clarify",
        state_version: "state-0",
        intent_version: "intent-1",
        plans: [],
        unresolved_conditions: [],
        user_visible_explanation: "should be blocked",
        mock_backend: true,
      },
    });
    expect(blockedDecision.currentDecision).toBeNull();
    expect(blockedDecision.events.at(-1)?.eventType).toBe("decision_blocked_no_ai");
  });

  it("allows completion without display consent", () => {
    const state = gameReducer(stateWithIntents(), {
      type: "COMPLETE_SESSION",
      at: "2026-09-17T01:00:00.000Z",
    });
    expect(state.completedAt).toBe("2026-09-17T01:00:00.000Z");
    expect(state.shareConsent).toBe(false);
    expect(state.events.at(-1)?.payload).toEqual({ shareConsent: false });
  });

  it("adds the standardized preserve-work intent at the change node", () => {
    const baseState = {
      ...stateWithIntents(),
      revisionStage: "change_pending" as const,
    };
    const state = gameReducer(baseState, {
      type: "TRIGGER_REVISION",
      at: "2026-09-17T00:30:00.000Z",
    });
    expect(state.revisionStage).toBe("change_active");
    expect(state.intents.at(-1)?.intentId).toBe("intent-preserve-work");
    expect(state.intents.at(-1)?.locked).toBe(true);
    expect(state.events.at(-1)?.eventType).toBe("standardized_change_presented");
  });
});
