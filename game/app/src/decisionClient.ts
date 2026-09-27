import { buildMockDecision } from "./mockBackend";
import type { DecisionResponse, GameState, PlanOption } from "./types";

export type DecisionMode = "study" | "mock" | "proxy";

export interface DecisionRequest {
  schema_version: "0.1.0";
  session_id: string;
  condition: "b4" | "ensr" | "iper";
  state_version: string;
  intent_version: string;
  selected_material_id: string | null;
  selected_pattern_id: string | null;
  selected_tool_ids: string[];
  selected_reference_id: string | null;
  intents: Array<{
    intent_id: string;
    statement: string;
    priority: 1 | 2 | 3;
    locked: boolean;
  }>;
  lantern: GameState["lantern"];
}

export class DecisionClientError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "DecisionClientError";
  }
}

function version(prefix: string, value: number): string {
  return `${prefix}-${value}`;
}

export function buildDecisionRequest(state: GameState): DecisionRequest {
  return {
    schema_version: "0.1.0",
    session_id: state.sessionId,
    condition: state.condition,
    state_version: version("state", state.stateVersion),
    intent_version: version("intent", state.intentVersion),
    selected_material_id: state.selectedMaterialId,
    selected_pattern_id: state.selectedPatternId,
    selected_tool_ids: state.selectedToolIds,
    selected_reference_id: state.selectedReferenceId,
    intents: state.intents.map((intent) => ({
      intent_id: intent.intentId,
      statement: intent.statement,
      priority: intent.priority,
      locked: intent.locked,
    })),
    lantern: state.lantern,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isPlan(value: unknown): value is PlanOption {
  if (!isRecord(value) || !isRecord(value.effects)) return false;
  const stringFields = ["plan_id", "title", "summary", "tradeoff"];
  const arrayFields = [
    "action_ids",
    "retained_intent_ids",
    "proposed_changes",
    "required_consent",
    "evidence_ids",
    "unresolved_conditions",
  ];
  return (
    stringFields.every((field) => typeof value[field] === "string") &&
    arrayFields.every((field) => isStringArray(value[field]))
  );
}

export function validateDecisionResponse(
  value: unknown,
  request: DecisionRequest,
  expectedMockBackend?: boolean,
): DecisionResponse {
  if (!isRecord(value)) {
    throw new DecisionClientError("invalid_response", "决策服务返回了无效数据。 ");
  }
  const allowedTypes = new Set([
    "propose",
    "clarify",
    "conflict",
    "evidence_insufficient",
  ]);
  if (
    typeof value.decision_type !== "string" ||
    !allowedTypes.has(value.decision_type) ||
    value.state_version !== request.state_version ||
    value.intent_version !== request.intent_version ||
    !Array.isArray(value.plans) ||
    value.plans.length > 3 ||
    !value.plans.every(isPlan) ||
    !isStringArray(value.unresolved_conditions) ||
    typeof value.user_visible_explanation !== "string" ||
    !value.user_visible_explanation.trim() ||
    typeof value.mock_backend !== "boolean"
  ) {
    throw new DecisionClientError(
      "invalid_response",
      "决策结果未通过结构或版本校验，请重试。",
    );
  }
  if (
    expectedMockBackend !== undefined &&
    value.mock_backend !== expectedMockBackend
  ) {
    throw new DecisionClientError(
      "backend_mode_mismatch",
      "决策服务模式与研究配置不一致。",
    );
  }
  return value as unknown as DecisionResponse;
}

export async function requestDecision(
  state: GameState,
  options: {
    mode?: DecisionMode;
    endpoint?: string;
    timeoutMs?: number;
    fetcher?: typeof fetch;
  } = {},
): Promise<DecisionResponse> {
  const mode =
    options.mode ??
    (import.meta.env.VITE_DECISION_MODE === "proxy"
      ? "proxy"
      : import.meta.env.VITE_DECISION_MODE === "mock"
        ? "mock"
        : "study");
  const request = buildDecisionRequest(state);
  if (mode === "study" || mode === "mock") {
    return validateDecisionResponse(
      buildMockDecision(
        state,
        mode === "study" ? "precomputed_e21" : "mock",
      ),
      request,
      mode === "mock",
    );
  }

  const endpoint =
    options.endpoint ?? import.meta.env.VITE_DECISION_API_URL ?? "/api/decision";
  const configuredTimeout = Number(import.meta.env.VITE_DECISION_TIMEOUT_MS);
  const defaultTimeout =
    Number.isFinite(configuredTimeout) && configuredTimeout >= 1_000
      ? configuredTimeout
      : 420_000;
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? defaultTimeout,
  );
  try {
    const response = await (options.fetcher ?? fetch)(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new DecisionClientError(
        `http_${response.status}`,
        response.status === 409
          ? "状态已经更新，正在等待重新核对。"
          : "决策服务暂时不可用，请稍后重试。",
      );
    }
    return validateDecisionResponse(await response.json(), request, false);
  } catch (error) {
    if (error instanceof DecisionClientError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new DecisionClientError("timeout", "决策请求超时，请重试。 ");
    }
    throw new DecisionClientError("network_error", "无法连接决策服务，请稍后重试。 ");
  } finally {
    globalThis.clearTimeout(timeout);
  }
}
