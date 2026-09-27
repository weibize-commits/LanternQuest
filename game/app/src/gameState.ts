import type {
  ActId,
  GameAction,
  GameState,
  IntentCard,
  PlanOption,
  SessionEvent,
  StudyCondition,
} from "./types";

export const STORAGE_KEY = "lanternquest.session.v4";

function makeId(prefix: string): string {
  const suffix =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${suffix}`;
}

function timestamp(at?: string): string {
  return at ?? new Date().toISOString();
}

function blankCraftProgress(): GameState["craftProgress"] {
  return {
    phase: "design",
    strokes: 0,
    holes: 0,
    assembledPanels: 0,
    marks: [],
    completedAt: null,
  };
}

export function createInitialState(
  condition: StudyCondition = "b4",
  participantCode: string | null = null,
): GameState {
  const startedAt = timestamp();
  const sessionId = makeId("session");
  const initial: GameState = {
    schemaVersion: "0.5.0",
    sessionId,
    participantCode,
    startedAt,
    completedAt: null,
    condition,
    currentAct: "wish",
    completedActs: [],
    commissionId: null,
    intents: [],
    intentVersion: 0,
    selectedMaterialId: null,
    selectedPatternId: null,
    generatedPattern: null,
    selectedToolIds: [],
    selectedReferenceId: null,
    lantern: {
      form: "rounded_hexagonal",
      material: "unselected",
      pattern: "unselected",
      tools: [],
      palette: "rice_paper_neutral",
      panelTreatment: "unselected",
      personalMark: "unplaced",
      assembly: "not_started",
    },
    stateVersion: 0,
    currentDecision: null,
    acceptedPlanId: null,
    initialPlanId: null,
    acceptedRevisionPlanId: null,
    revisionStage: "initial",
    independentExplanation: "",
    shareConsent: false,
    aiAssistanceEnabled: true,
    noAiStageStarted: false,
    craftProgress: blankCraftProgress(),
    events: [],
  };
  return appendEvent(initial, "session_started", { condition }, startedAt);
}

function appendEvent(
  state: GameState,
  eventType: string,
  payload: Record<string, unknown>,
  at?: string,
): GameState {
  const event: SessionEvent = {
    eventId: makeId("event"),
    sessionId: state.sessionId,
    recordedAt: timestamp(at),
    act: state.currentAct,
    eventType,
    stateVersion: state.stateVersion,
    intentVersion: state.intentVersion,
    payload,
  };
  return { ...state, events: [...state.events, event] };
}

function completedWith(state: GameState, act: ActId): ActId[] {
  return state.completedActs.includes(act)
    ? state.completedActs
    : [...state.completedActs, act];
}

function planTouchesLockedIntent(
  intents: IntentCard[],
  plan: PlanOption,
): string[] {
  const proposedChanges = new Set(plan.proposed_changes);
  const retainedIntents = new Set(plan.retained_intent_ids);
  return intents
    .filter((intent) => intent.locked)
    .map((intent) => intent.intentId)
    .filter(
      (intentId) =>
        proposedChanges.has(intentId) || !retainedIntents.has(intentId),
    );
}

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "CONFIRM_WISH": {
      if (action.intents.length !== 2) return state;
      const next = {
        ...state,
        commissionId: action.commissionId,
        intents: action.intents,
        intentVersion: state.intentVersion + 1,
        completedActs: completedWith(state, "wish"),
      };
      return appendEvent(
        next,
        "intent_confirmed",
        {
          commissionId: action.commissionId,
          intents: action.intents.map(({ intentId, priority, locked }) => ({
            intentId,
            priority,
            locked,
          })),
        },
        action.at,
      );
    }
    case "SELECT_REFERENCE": {
      const next = {
        ...state,
        selectedReferenceId: action.referenceId,
        stateVersion: state.stateVersion + 1,
        currentDecision: null,
        acceptedPlanId: null,
        craftProgress: blankCraftProgress(),
        completedActs: completedWith(state, "reference"),
      };
      return appendEvent(
        next,
        "reference_selected",
        { referenceId: action.referenceId },
        action.at,
      );
    }
    case "SELECT_MATERIAL": {
      const next = {
        ...state,
        selectedMaterialId: action.materialId,
        lantern: { ...state.lantern, material: action.materialId },
        stateVersion: state.stateVersion + 1,
        currentDecision: null,
        acceptedPlanId: null,
        craftProgress: blankCraftProgress(),
      };
      return appendEvent(
        next,
        "kg_material_selected",
        {
          materialId: action.materialId,
          kgNodeId: action.kgNodeId,
          knowledgeStatus: action.knowledgeStatus,
        },
        action.at,
      );
    }
    case "SELECT_PATTERN": {
      const next = {
        ...state,
        selectedPatternId: action.patternId,
        generatedPattern: null,
        lantern: { ...state.lantern, pattern: action.patternId },
        stateVersion: state.stateVersion + 1,
        currentDecision: null,
        acceptedPlanId: null,
        craftProgress: blankCraftProgress(),
      };
      return appendEvent(
        next,
        "player_pattern_selected",
        { patternId: action.patternId, provenance: "player_expression" },
        action.at,
      );
    }
    case "SET_AI_PATTERN": {
      const next = {
        ...state,
        selectedPatternId: "pattern-ai-generated",
        generatedPattern: action.pattern,
        lantern: { ...state.lantern, pattern: "pattern-ai-generated" },
        stateVersion: state.stateVersion + 1,
        currentDecision: null,
        acceptedPlanId: null,
        craftProgress: blankCraftProgress(),
      };
      return appendEvent(
        next,
        "ai_pattern_generated",
        {
          assetId: action.pattern.assetId,
          model: action.pattern.model,
          provenance: "player_expression",
        },
        action.at,
      );
    }
    case "SELECT_TOOLSET": {
      if (action.toolIds.length !== 3) return state;
      const next = {
        ...state,
        selectedToolIds: [...action.toolIds],
        lantern: { ...state.lantern, tools: [...action.toolIds] },
        stateVersion: state.stateVersion + 1,
        currentDecision: null,
        acceptedPlanId: null,
      };
      return appendEvent(
        next,
        "simulation_toolset_selected",
        { toolIds: action.toolIds },
        action.at,
      );
    }
    case "UPDATE_CRAFT_PROGRESS": {
      const next = {
        ...state,
        craftProgress: action.progress,
        stateVersion: state.stateVersion + 1,
      };
      return appendEvent(
        next,
        `craft_${action.reason}`,
        {
          phase: action.progress.phase,
          strokes: action.progress.strokes,
          holes: action.progress.holes,
          assembledPanels: action.progress.assembledPanels,
          markCount: action.progress.marks.length,
        },
        action.at,
      );
    }
    case "RECORD_EVIDENCE_VIEW":
      return appendEvent(
        state,
        "evidence_opened",
        { evidenceId: action.evidenceId },
        action.at,
      );
    case "RECEIVE_DECISION": {
      if (!state.aiAssistanceEnabled || state.noAiStageStarted) {
        return appendEvent(
          state,
          "decision_blocked_no_ai",
          { decisionType: action.decision.decision_type },
          action.at,
        );
      }
      const expectedStateVersion = `state-${state.stateVersion}`;
      const expectedIntentVersion = `intent-${state.intentVersion}`;
      if (
        action.decision.state_version !== expectedStateVersion ||
        action.decision.intent_version !== expectedIntentVersion
      ) {
        return appendEvent(
          state,
          "decision_rejected_stale_version",
          {
            receivedStateVersion: action.decision.state_version,
            receivedIntentVersion: action.decision.intent_version,
            expectedStateVersion,
            expectedIntentVersion,
          },
          action.at,
        );
      }
      const next = { ...state, currentDecision: action.decision };
      return appendEvent(
        next,
        "decision_received",
        {
          decisionType: action.decision.decision_type,
          planIds: action.decision.plans.map((plan) => plan.plan_id),
          mockBackend: action.decision.mock_backend,
          requestId: action.decision.run_metadata?.request_id ?? null,
          requestSha256: action.decision.run_metadata?.request_sha256 ?? null,
          modelId: action.decision.run_metadata?.model_id ?? null,
          symbolicRepairs: action.decision.run_metadata?.symbolic_repairs ?? [],
        },
        action.at,
      );
    }
    case "APPLY_PLAN": {
      if (state.noAiStageStarted) {
        return appendEvent(
          state,
          "plan_blocked_no_ai",
          { planId: action.plan.plan_id },
          action.at,
        );
      }
      const blocked = planTouchesLockedIntent(state.intents, action.plan);
      if (blocked.length > 0) {
        return appendEvent(
          state,
          "plan_blocked_locked_intent",
          { planId: action.plan.plan_id, blockedIntentIds: blocked },
          action.at,
        );
      }
      const isRevision = state.revisionStage === "change_active";
      const next = {
        ...state,
        lantern: { ...state.lantern, ...action.plan.effects },
        stateVersion: state.stateVersion + 1,
        acceptedPlanId: action.plan.plan_id,
        initialPlanId: isRevision ? state.initialPlanId : action.plan.plan_id,
        acceptedRevisionPlanId: isRevision
          ? action.plan.plan_id
          : state.acceptedRevisionPlanId,
        revisionStage: isRevision ? ("complete" as const) : ("change_pending" as const),
        currentDecision: isRevision ? state.currentDecision : null,
        completedActs: isRevision
          ? completedWith(state, "craft")
          : state.completedActs,
      };
      return appendEvent(
        next,
        isRevision ? "revision_plan_accepted" : "initial_plan_accepted",
        {
          planId: action.plan.plan_id,
          retainedIntentIds: action.plan.retained_intent_ids,
          evidenceIds: action.plan.evidence_ids,
          reworkCost: action.plan.rework_cost ?? 0,
        },
        action.at,
      );
    }
    case "TRIGGER_REVISION": {
      if (state.revisionStage !== "change_pending" || state.noAiStageStarted) {
        return state;
      }
      const at = timestamp(action.at);
      const preserveIntent: IntentCard = {
        intentId: "intent-preserve-work",
        label: "保留已完成样片",
        statement: "新增一张不同图案的灯片，同时尽量保留已经完成的样片。",
        priority: 3,
        locked: true,
        confirmedAt: at,
      };
      const intents = state.intents.some(
        (intent) => intent.intentId === preserveIntent.intentId,
      )
        ? state.intents
        : [...state.intents, preserveIntent];
      const next = {
        ...state,
        intents,
        intentVersion: state.intentVersion + 1,
        stateVersion: state.stateVersion + 1,
        currentDecision: null,
        revisionStage: "change_active" as const,
      };
      return appendEvent(
        next,
        "standardized_change_presented",
        {
          scenarioId: "lq_055_test_intent_and_rework",
          newLockedIntentId: preserveIntent.intentId,
          completedWork: "controlled_local_piercing_sample",
        },
        at,
      );
    }
    case "GO_TO_ACT": {
      const noAi = action.act === "showcase" || action.act === "handoff";
      const noAiStageStarted = state.noAiStageStarted || noAi;
      const completedActs =
        action.act === "handoff"
          ? completedWith(state, "showcase")
          : state.completedActs;
      const next = {
        ...state,
        currentAct: action.act,
        completedActs,
        noAiStageStarted,
        aiAssistanceEnabled: !noAiStageStarted,
      };
      return appendEvent(
        next,
        "act_entered",
        { act: action.act, aiAssistanceEnabled: !noAiStageStarted },
        action.at,
      );
    }
    case "SET_EXPLANATION": {
      const next = { ...state, independentExplanation: action.value };
      return appendEvent(
        next,
        "independent_explanation_updated",
        { characterCount: action.value.length },
        action.at,
      );
    }
    case "SET_SHARE_CONSENT": {
      const next = { ...state, shareConsent: action.value };
      return appendEvent(
        next,
        "share_consent_changed",
        { granted: action.value },
        action.at,
      );
    }
    case "COMPLETE_SESSION": {
      const at = timestamp(action.at);
      const next = {
        ...state,
        completedAt: at,
        completedActs: completedWith(state, "handoff"),
        noAiStageStarted: true,
        aiAssistanceEnabled: false,
      };
      return appendEvent(
        next,
        "session_completed",
        { shareConsent: state.shareConsent },
        at,
      );
    }
    case "LOG_EVENT":
      return appendEvent(
        state,
        action.eventType,
        action.payload ?? {},
        action.at,
      );
  }
  return state;
}

export function conditionFromLocation(search: string): StudyCondition {
  const params = new URLSearchParams(search);
  const arm = params.get("arm")?.toUpperCase();
  if (arm === "B") return "ensr";
  if (arm === "A") return "b4";
  const value = params.get("condition");
  if (value === "ensr") return "ensr";
  if (value === "iper") return "iper";
  return "b4";
}

export function participantCodeFromLocation(search: string): string | null {
  const value = new URLSearchParams(search).get("participant")?.trim() ?? "";
  return /^LQ-\d{3}$/.test(value) ? value : null;
}

export function loadStoredState(): GameState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as GameState;
    if (parsed.schemaVersion !== "0.5.0") return null;
    const noAiStageStarted =
      parsed.noAiStageStarted ??
      (parsed.currentAct === "showcase" ||
        parsed.currentAct === "handoff" ||
        Boolean(parsed.completedAt));
    return {
      ...parsed,
      noAiStageStarted,
      aiAssistanceEnabled: noAiStageStarted ? false : parsed.aiAssistanceEnabled,
    };
  } catch {
    return null;
  }
}

export function saveState(state: GameState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    const compact = state.generatedPattern
      ? { ...state, generatedPattern: { ...state.generatedPattern, imageDataUrl: "" } }
      : state;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
  }
}

export function clearStoredState(): void {
  localStorage.removeItem(STORAGE_KEY);
}
