import { describe, expect, it, vi } from "vitest";

import {
  buildDecisionRequest,
  DecisionClientError,
  requestDecision,
  validateDecisionResponse,
} from "./decisionClient";
import { createInitialState, gameReducer } from "./gameState";
import { buildMockDecision } from "./mockBackend";
import type { GameState, IntentCard } from "./types";

function preparedState(): GameState {
  const intents: IntentCard[] = [
    {
      intentId: "intent-one",
      label: "one",
      statement: "Keep one",
      priority: 3,
      locked: true,
      confirmedAt: "2026-09-23T00:00:00Z",
    },
    {
      intentId: "intent-two",
      label: "two",
      statement: "Keep two",
      priority: 2,
      locked: true,
      confirmedAt: "2026-09-23T00:00:00Z",
    },
  ];
  let state = gameReducer(createInitialState("iper"), {
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
  state = gameReducer(state, {
    type: "SELECT_REFERENCE",
    referenceId: "ref-modular",
  });
  return state;
}

describe("decision client", () => {
  it("builds a versioned request without exporting the event log", () => {
    const request = buildDecisionRequest(preparedState());
    expect(request.state_version).toBe("state-4");
    expect(request.intent_version).toBe("intent-1");
    expect(request.intents).toHaveLength(2);
    expect(request.selected_material_id).toBe("material-warm-translucent");
    expect(request.selected_pattern_id).toBe("pattern-geometric-rhythm");
    expect(request.selected_tool_ids).toHaveLength(3);
    expect(request).not.toHaveProperty("events");
  });

  it("uses the reproducible mock only in explicit mock mode", async () => {
    const decision = await requestDecision(preparedState(), { mode: "mock" });
    expect(decision.mock_backend).toBe(true);
    expect(decision.plans).toHaveLength(1);
  });

  it("posts to the proxy and requires a non-mock response", async () => {
    const state = preparedState();
    const response = { ...buildMockDecision(state), mock_backend: false };
    const fetcher = vi.fn(async () =>
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const decision = await requestDecision(state, {
      mode: "proxy",
      endpoint: "https://example.test/api/decision",
      fetcher: fetcher as typeof fetch,
    });

    expect(decision.mock_backend).toBe(false);
    expect(fetcher).toHaveBeenCalledOnce();
    const [, options] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(options?.method).toBe("POST");
  });

  it("rejects stale decision versions", () => {
    const state = preparedState();
    const request = buildDecisionRequest(state);
    const stale = { ...buildMockDecision(state), state_version: "state-999" };

    expect(() => validateDecisionResponse(stale, request)).toThrow(
      DecisionClientError,
    );
  });

  it("does not silently fall back after a proxy error", async () => {
    const fetcher = vi.fn(async () => new Response("unavailable", { status: 503 }));

    await expect(
      requestDecision(preparedState(), {
        mode: "proxy",
        fetcher: fetcher as typeof fetch,
      }),
    ).rejects.toMatchObject({ code: "http_503" });
  });
});
