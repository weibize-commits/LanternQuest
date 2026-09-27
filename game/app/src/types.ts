export type ActId = "wish" | "reference" | "craft" | "showcase" | "handoff";
export type StudyCondition = "b4" | "ensr" | "iper";
export type DecisionType =
  | "propose"
  | "clarify"
  | "conflict"
  | "evidence_insufficient";

export interface IntentCard {
  intentId: string;
  label: string;
  statement: string;
  priority: 1 | 2 | 3;
  locked: boolean;
  confirmedAt: string;
}

export interface ReferenceOption {
  referenceId: string;
  title: string;
  subtitle: string;
  description: string;
  evidenceIds: string[];
  accent: "cinnabar" | "celadon" | "indigo";
}

export interface MaterialOption {
  materialId: string;
  label: string;
  description: string;
  kgNodeId: string;
  knowledgeStatus: "task_approved" | "simulation_only" | "unknown_pending";
  selectable: boolean;
  evidenceIds: string[];
  surfaceColor: string;
  transmission: number;
  durability: number;
}

export interface PatternOption {
  patternId: string;
  label: string;
  description: string;
  provenance: "player_expression";
  motif: "ai" | "geometry" | "botanical" | "memory" | "freeform";
}

export interface ToolOption {
  toolId: string;
  category: "drawing" | "piercing" | "assembly";
  label: string;
  description: string;
  simulationEffect: string;
  model: "stylus" | "stencil" | "fine_awl" | "broad_awl" | "clip" | "hand";
}

export interface EvidenceCard {
  evidenceId: string;
  title: string;
  summary: string;
  scope: string;
  sourceLabel: string;
  reviewStatus: "development_placeholder" | "domain_approved";
  media?: Array<{
    assetId: string;
    label: string;
    alt: string;
    endpoint: string;
  }>;
}

export interface LanternState {
  form: string;
  material: string;
  pattern: string;
  tools: string[];
  palette: string;
  panelTreatment: string;
  personalMark: string;
  assembly: "not_started" | "in_progress" | "ready_to_show";
}

export interface PlanOption {
  plan_id: string;
  title: string;
  summary: string;
  action_ids: string[];
  retained_intent_ids: string[];
  proposed_changes: string[];
  required_consent: string[];
  evidence_ids: string[];
  unresolved_conditions: string[];
  effects: Partial<LanternState>;
  tradeoff: string;
  rework_cost?: number;
}

export interface DecisionResponse {
  decision_type: DecisionType;
  state_version: string;
  intent_version: string;
  plans: PlanOption[];
  unresolved_conditions: string[];
  user_visible_explanation: string;
  mock_backend: boolean;
  backend_kind?: "precomputed_e21" | "mock" | "local_proxy";
  run_metadata?: {
    request_id: string;
    request_sha256: string;
    model_id: string | null;
    system_fingerprint: string | null;
    prompt_version: string;
    retrieval_snapshot_id: string | null;
    latency_ms: number;
    retry_count: number;
    symbolic_repairs?: string[];
    validation_status: "passed";
  };
}

export interface SessionEvent {
  eventId: string;
  sessionId: string;
  recordedAt: string;
  act: ActId;
  eventType: string;
  stateVersion: number;
  intentVersion: number;
  payload: Record<string, unknown>;
}

export interface CraftMark {
  x: number;
  y: number;
  kind: "draw" | "hole";
  size: number;
  strokeId: number;
}

export interface CraftProgress {
  phase: "design" | "pierce" | "assemble" | "complete";
  strokes: number;
  holes: number;
  assembledPanels: number;
  marks: CraftMark[];
  completedAt: string | null;
}

export interface GeneratedPattern {
  assetId: string;
  imageDataUrl: string;
  model: string;
  createdAt: string;
}

export interface GameState {
  schemaVersion: "0.5.0";
  sessionId: string;
  participantCode: string | null;
  startedAt: string;
  completedAt: string | null;
  condition: StudyCondition;
  currentAct: ActId;
  completedActs: ActId[];
  commissionId: string | null;
  intents: IntentCard[];
  intentVersion: number;
  selectedMaterialId: string | null;
  selectedPatternId: string | null;
  generatedPattern: GeneratedPattern | null;
  selectedToolIds: string[];
  selectedReferenceId: string | null;
  lantern: LanternState;
  stateVersion: number;
  currentDecision: DecisionResponse | null;
  acceptedPlanId: string | null;
  initialPlanId: string | null;
  acceptedRevisionPlanId: string | null;
  revisionStage: "initial" | "change_pending" | "change_active" | "complete";
  independentExplanation: string;
  shareConsent: boolean;
  aiAssistanceEnabled: boolean;
  noAiStageStarted: boolean;
  craftProgress: CraftProgress;
  events: SessionEvent[];
}

export type GameAction =
  | {
      type: "CONFIRM_WISH";
      commissionId: string;
      intents: IntentCard[];
      at?: string;
    }
  | { type: "SELECT_REFERENCE"; referenceId: string; at?: string }
  | {
      type: "SELECT_MATERIAL";
      materialId: string;
      kgNodeId: string;
      knowledgeStatus: MaterialOption["knowledgeStatus"];
      at?: string;
    }
  | { type: "SELECT_PATTERN"; patternId: string; at?: string }
  | { type: "SET_AI_PATTERN"; pattern: GeneratedPattern; at?: string }
  | { type: "SELECT_TOOLSET"; toolIds: string[]; at?: string }
  | {
      type: "UPDATE_CRAFT_PROGRESS";
      progress: CraftProgress;
      reason: "draw" | "pierce" | "phase" | "assemble" | "complete";
      at?: string;
    }
  | { type: "RECORD_EVIDENCE_VIEW"; evidenceId: string; at?: string }
  | { type: "RECEIVE_DECISION"; decision: DecisionResponse; at?: string }
  | { type: "APPLY_PLAN"; plan: PlanOption; at?: string }
  | { type: "TRIGGER_REVISION"; at?: string }
  | { type: "GO_TO_ACT"; act: ActId; at?: string }
  | { type: "SET_EXPLANATION"; value: string; at?: string }
  | { type: "SET_SHARE_CONSENT"; value: boolean; at?: string }
  | { type: "COMPLETE_SESSION"; at?: string }
  | {
      type: "LOG_EVENT";
      eventType: string;
      payload?: Record<string, unknown>;
      at?: string;
    };
