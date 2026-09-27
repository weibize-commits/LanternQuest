import { useEffect, useMemo, useState, type Dispatch as ReactDispatch } from "react";

import {
  ACTS,
  COMMISSIONS,
  EVIDENCE,
  INTENT_CHOICES,
  MATERIALS,
  PATTERNS,
  REFERENCES,
  TOOLS,
} from "./data";
import { DecisionClientError, requestDecision } from "./decisionClient";
import { Icon } from "./Icon";
import { PatternClientError, generatePattern } from "./patternClient";
import { ProcessReel } from "./ProcessReel";
import type {
  ActId,
  EvidenceCard,
  GameAction,
  GameState,
  IntentCard,
  PlanOption,
} from "./types";
import { useSession } from "./useSession";
import { Workshop3D } from "./Workshop3D";

type Dispatch = ReactDispatch<GameAction>;

const ACT_INDEX: Record<ActId, number> = {
  wish: 0,
  reference: 1,
  craft: 2,
  showcase: 3,
  handoff: 4,
};

const RESTRICTED_MEDIA_ENABLED = import.meta.env.VITE_DECISION_MODE === "proxy";

const ILLUSTRATION_PRESETS = [
  { prompt: "桂花树、月兔、层叠流云与远山", icon: "moon" as const },
  { prompt: "汴河夜色、拱桥、画舫与成排灯影", icon: "lantern" as const },
  { prompt: "锦鲤穿过荷叶、水波与金色星点", icon: "leaf" as const },
  { prompt: "喜鹊停在梅枝，细雪与月光交叠", icon: "sparkles" as const },
];

function evidenceMediaUrl(endpoint: string): string {
  const decisionEndpoint =
    import.meta.env.VITE_DECISION_API_URL ?? "/api/decision";
  const decisionUrl = new URL(decisionEndpoint, globalThis.location.href);
  return new URL(endpoint, decisionUrl).toString();
}

function ChapterProgress({ state, dispatch }: { state: GameState; dispatch: Dispatch }) {
  const currentIndex = ACT_INDEX[state.currentAct];
  return (
    <nav className="chapter-progress" aria-label="游戏章节">
      {ACTS.map((act, index) => {
        const available = index <= currentIndex || state.completedActs.includes(act.id);
        return (
          <button
            aria-label={`第 ${index + 1} 幕：${act.label}`}
            className={`chapter-step ${act.id === state.currentAct ? "is-current" : ""} ${
              state.completedActs.includes(act.id) ? "is-complete" : ""
            }`}
            disabled={!available}
            key={act.id}
            onClick={() => available && dispatch({ type: "GO_TO_ACT", act: act.id })}
            type="button"
          >
            <span className="chapter-icon" data-act={index + 1} aria-hidden="true" />
            <span className="chapter-copy">
              <small>第 {index + 1} 幕</small>
              <strong>{act.short}</strong>
            </span>
          </button>
        );
      })}
    </nav>
  );
}

function ResearchPanel({
  state,
  onExport,
  onReset,
}: {
  state: GameState;
  onExport: () => void;
  onReset: () => void;
}) {
  return (
    <details className="research-panel">
      <summary aria-label="研究记录" title="研究记录"><Icon name="evidence" /></summary>
      <dl>
        <div><dt>条件</dt><dd>{state.condition === "b4" ? "条件 A" : "条件 B"}</dd></div>
        <div><dt>研究编号</dt><dd>{state.participantCode ?? "未分配"}</dd></div>
        <div><dt>状态版本</dt><dd>{state.stateVersion}</dd></div>
        <div><dt>意图版本</dt><dd>{state.intentVersion}</dd></div>
        <div><dt>事件</dt><dd>{state.events.length}</dd></div>
        <div><dt>会话</dt><dd title={state.sessionId}>{state.sessionId.slice(-8)}</dd></div>
      </dl>
      <div className="research-actions">
        <button className="button button-secondary" onClick={onExport} type="button">导出记录</button>
        <button className="button button-quiet" onClick={onReset} type="button">重置会话</button>
      </div>
    </details>
  );
}

function EvidencePanel({
  items,
  dispatch,
}: {
  items: EvidenceCard[];
  dispatch: Dispatch;
}) {
  if (items.length === 0) return null;
  return (
    <aside className="evidence-panel evidence-drawer" aria-label="依据区">
      <details>
        <summary aria-label={`查看 ${items.length} 条知识图谱依据`} title="查看依据"><Icon name="evidence" /><span>{items.length}</span></summary>
        <div className="evidence-drawer-body">
          <div className="evidence-list">
            {items.map((item) => (
              <details
                className="evidence-card"
                key={item.evidenceId}
                onToggle={(event) => {
                  if (event.currentTarget.open) dispatch({ type: "RECORD_EVIDENCE_VIEW", evidenceId: item.evidenceId });
                }}
              >
                <summary>{item.title}</summary>
                <p>{item.summary}</p>
                {item.media && item.media.length > 0 && RESTRICTED_MEDIA_ENABLED ? (
                  <div className="evidence-media-grid">
                    {item.media.map((media) => <figure key={media.assetId}><img alt={media.alt} loading="lazy" src={evidenceMediaUrl(media.endpoint)} /><figcaption>{media.label}</figcaption></figure>)}
                  </div>
                ) : null}
                <dl><div><dt>范围</dt><dd>{item.scope}</dd></div><div><dt>来源</dt><dd>{item.sourceLabel}</dd></div></dl>
              </details>
            ))}
          </div>
        </div>
      </details>
    </aside>
  );
}

function WishScreen({ state, dispatch }: { state: GameState; dispatch: Dispatch }) {
  const [commissionId, setCommissionId] = useState(state.commissionId ?? "");
  const [selected, setSelected] = useState<Record<string, { priority: 1 | 2 | 3; locked: boolean }>>(
    Object.fromEntries(
      state.intents.map((intent) => [
        intent.intentId,
        { priority: intent.priority, locked: intent.locked },
      ]),
    ),
  );
  const selectedIds = Object.keys(selected);

  function toggleIntent(intentId: string): void {
    setSelected((current) => {
      if (current[intentId]) {
        const next = { ...current };
        delete next[intentId];
        return next;
      }
      const choice = INTENT_CHOICES.find((item) => item.id === intentId)!;
      const sameDimension = INTENT_CHOICES.find(
        (item) => item.dimension === choice.dimension && current[item.id],
      );
      const next = { ...current };
      if (sameDimension) delete next[sameDimension.id];
      return { ...next, [intentId]: { priority: 3, locked: true } };
    });
  }

  function confirm(): void {
    const confirmedAt = new Date().toISOString();
    const intents: IntentCard[] = selectedIds.map((intentId) => {
      const choice = INTENT_CHOICES.find((item) => item.id === intentId)!;
      return {
        intentId,
        label: choice.label,
        statement: choice.statement,
        priority: selected[intentId].priority,
        locked: selected[intentId].locked,
        confirmedAt,
      };
    });
    dispatch({ type: "CONFIRM_WISH", commissionId, intents });
    dispatch({ type: "GO_TO_ACT", act: "reference" });
  }

  return (
    <section className="screen-card wide-card icon-first-screen">
      <div className="section-heading compact">
        <span className="eyebrow">第一幕</span>
        <h1>许下灯愿</h1>
      </div>
      <fieldset className="choice-section visual-choice-section">
        <legend><Icon name="paper" />选任务</legend>
        <div className="commission-grid">
          {COMMISSIONS.map((commission, index) => (
            <label className={`choice-card visual-card ${commissionId === commission.id ? "is-selected" : ""}`} key={commission.id} title={commission.description}>
              <input
                checked={commissionId === commission.id}
                name="commission"
                onChange={() => setCommissionId(commission.id)}
                type="radio"
              />
              <Icon name={index === 0 ? "paper" : index === 1 ? "evidence" : "vary"} />
              <strong>{commission.title}</strong>
            </label>
          ))}
        </div>
      </fieldset>
      <fieldset className="choice-section visual-choice-section">
        <legend><Icon name="lock" />锁定两项</legend>
        <div className="intent-grid">
          {INTENT_CHOICES.map((intent, index) => {
            const value = selected[intent.id];
            return (
              <article className={`intent-card visual-card ${value ? "is-selected" : ""}`} key={intent.id} title={intent.statement}>
                <label className="intent-main">
                  <input checked={Boolean(value)} onChange={() => toggleIntent(intent.id)} type="checkbox" />
                  <Icon name={index === 0 ? "repeat" : index === 1 ? "vary" : index === 2 ? "sun" : "moon"} />
                  <span><strong>{intent.label}</strong></span>
                </label>
                {value && <span className="visual-check"><Icon name="check" /></span>}
              </article>
            );
          })}
        </div>
      </fieldset>
      <div className="screen-actions">
        <span className="selection-progress">{commissionId ? 1 : 0} + {selectedIds.length}/2</span>
        <button className="button button-primary icon-action" aria-label="确认并观看制作流程" disabled={!commissionId || selectedIds.length !== 2} onClick={confirm} title="确认并观看制作流程" type="button"><Icon name="arrow" /><span>继续</span></button>
      </div>
    </section>
  );
}

function ReferenceScreen({ state, dispatch }: { state: GameState; dispatch: Dispatch }) {
  const processSeen = state.events.some((event) => event.eventType === "process_reel_completed");
  const [showProcess, setShowProcess] = useState(!processSeen);
  const [selectedId, setSelectedId] = useState(state.selectedReferenceId ?? "");
  const [materialId, setMaterialId] = useState(state.selectedMaterialId ?? "");
  const [patternId, setPatternId] = useState(state.selectedPatternId ?? "");
  const [toolIds, setToolIds] = useState<string[]>(state.selectedToolIds);
  const [patternPrompt, setPatternPrompt] = useState("");
  const [patternLoading, setPatternLoading] = useState(false);
  const [patternError, setPatternError] = useState("");
  const selectedReference = REFERENCES.find((item) => item.referenceId === selectedId);
  const selectedMaterial = MATERIALS.find((item) => item.materialId === materialId);
  const selectedPattern = PATTERNS.find((item) => item.patternId === patternId);
  const evidenceIds = new Set([
    ...(selectedReference?.evidenceIds ?? []),
    ...(selectedMaterial?.evidenceIds ?? []),
  ]);
  const evidence = EVIDENCE.filter((item) => evidenceIds.has(item.evidenceId));
  const patternIntent = state.intents.find((intent) => intent.intentId.includes("pattern"));
  const lightIntent = state.intents.find(
    (intent) => intent.intentId.includes("brighter"),
  );
  const canConfirm = Boolean(
    selectedId &&
      selectedPattern &&
      (selectedPattern.motif !== "ai" || Boolean(state.generatedPattern?.imageDataUrl)) &&
      selectedMaterial?.selectable &&
      new Set(TOOLS.filter((tool) => toolIds.includes(tool.toolId)).map((tool) => tool.category)).size === 3,
  );

  function confirm(): void {
    if (!selectedMaterial?.selectable || !selectedPattern || !selectedReference) return;
    dispatch({
      type: "SELECT_MATERIAL",
      materialId: selectedMaterial.materialId,
      kgNodeId: selectedMaterial.kgNodeId,
      knowledgeStatus: selectedMaterial.knowledgeStatus,
    });
    if (selectedPattern.patternId !== "pattern-ai-generated") {
      dispatch({ type: "SELECT_PATTERN", patternId: selectedPattern.patternId });
    }
    dispatch({ type: "SELECT_TOOLSET", toolIds });
    dispatch({ type: "SELECT_REFERENCE", referenceId: selectedId });
    dispatch({ type: "GO_TO_ACT", act: "craft" });
  }

  function finishProcess(): void {
    setShowProcess(false);
    if (!processSeen) {
      dispatch({ type: "LOG_EVENT", eventType: "process_reel_completed", payload: { steps: 9, source: "N001_and_field06" } });
    }
  }

  async function createPattern(): Promise<void> {
    const prompt = patternPrompt.trim();
    if (prompt.length < 2 || patternLoading) return;
    setPatternLoading(true);
    setPatternError("");
    dispatch({ type: "LOG_EVENT", eventType: "ai_pattern_requested", payload: { characterCount: prompt.length } });
    try {
      const result = await generatePattern(prompt);
      dispatch({
        type: "SET_AI_PATTERN",
        pattern: {
          assetId: result.asset_id,
          imageDataUrl: result.image_data_url,
          model: result.model,
          createdAt: result.created_at,
        },
      });
      setPatternId("pattern-ai-generated");
    } catch (error) {
      setPatternError(error instanceof PatternClientError ? error.message : "生成失败，请重试");
      dispatch({ type: "LOG_EVENT", eventType: "ai_pattern_failed", payload: { code: error instanceof PatternClientError ? error.code : "unknown" } });
    } finally {
      setPatternLoading(false);
    }
  }

  if (showProcess) return <ProcessReel onFinish={finishProcess} />;

  function chooseTool(toolId: string): void {
    const option = TOOLS.find((tool) => tool.toolId === toolId);
    if (!option) return;
    setToolIds((current) => [
      ...current.filter((id) => TOOLS.find((tool) => tool.toolId === id)?.category !== option.category),
      option.toolId,
    ]);
  }

  function chooseMaterial(nextId: string): void {
    const option = MATERIALS.find((item) => item.materialId === nextId);
    if (!option) return;
    setMaterialId(nextId);
    if (!option.selectable) {
      dispatch({
        type: "LOG_EVENT",
        eventType: "kg_unknown_material_viewed",
        payload: {
          materialId: option.materialId,
          kgNodeId: option.kgNodeId,
          knowledgeStatus: option.knowledgeStatus,
        },
      });
    }
  }

  return (
    <section className="screen-layout">
      <div className="screen-card main-column icon-first-screen">
        <div className="section-heading">
          <span className="eyebrow">第二幕 · 选材</span>
          <div className="icon-heading"><h1>组一盏自己的灯</h1><button className="icon-button" aria-label="重看制作流程" onClick={() => setShowProcess(true)} title="重看九步影卷" type="button"><Icon name="play" /></button></div>
        </div>

        <section className="kg-map" aria-label="万眼萝局部样片知识图谱">
          <div className="kg-map-heading">
            <div><span className="eyebrow">可见 KG</span><h2>当前知识路径</h2></div>
            <span className="kg-count">冻结图谱 · 54 节点 / 60 关系</span>
          </div>
          <div className="kg-path">
            <div className="kg-node status-approved"><small>任务</small><strong>万眼萝局部样片</strong></div>
            <span className="kg-edge">限定材料 →</span>
            <div className={`kg-node ${selectedMaterial?.knowledgeStatus === "unknown_pending" ? "status-unknown" : "status-expression"}`}><small>数字材料试样</small><strong>{selectedMaterial?.label ?? "待选择"}</strong></div>
            <span className="kg-edge">承载 →</span>
            <div className={`kg-node ${selectedPattern ? "status-expression" : "status-unknown"}`}><small>玩家图案</small><strong>{selectedPattern?.label ?? "待选择"}</strong></div>
            <span className="kg-edge">按意图处理 →</span>
            <div className="kg-node status-approved"><small>工艺决策</small><strong>{patternIntent?.label ?? "待确认"} · {lightIntent?.label ?? "待确认"}</strong></div>
            <span className="kg-edge">证据支持 →</span>
            <div className="kg-node status-approved"><small>结果边界</small><strong>局部透光样片</strong></div>
          </div>
          <div className="kg-legend"><span><i className="legend-approved" />专家确认</span><span><i className="legend-expression" />玩家表达</span><span><i className="legend-unknown" />待验证</span></div>
        </section>

        <fieldset className="kg-choice-section">
          <legend><Icon name="paper" />纸</legend>
          <p className="field-note">选择会实际改变三维纸面的颜色、透光与操作容差。三种可选项都是数字模拟参数，不冒充真实传统纸种。</p>
          <div className="material-grid">
            {MATERIALS.map((material) => (
              <button
                aria-pressed={materialId === material.materialId}
                className={`material-card ${materialId === material.materialId ? "is-selected" : ""} ${material.knowledgeStatus === "unknown_pending" ? "is-unknown" : "is-simulation"}`}
                key={material.materialId}
                onClick={() => chooseMaterial(material.materialId)}
                type="button"
              >
                <span className="knowledge-status">{material.knowledgeStatus === "unknown_pending" ? "KG 未知 · 待专家验证" : "数字模拟 · 会改变 3D 反馈"}</span>
                <strong>{material.label}</strong>
                <p>{material.description}</p>
                {material.selectable && <span className="material-stats">透光 {Math.round(material.transmission * 100)} · 耐受 {Math.round(material.durability * 100)}</span>}
                <code>{material.kgNodeId}</code>
              </button>
            ))}
          </div>
          {selectedMaterial && !selectedMaterial.selectable && (
            <p className="kg-unknown-message" role="status">这项材料尚缺少纸种、厚度与安全边界证据。它已在 KG 中保留为未知条件，但不能用于本轮正式任务。</p>
          )}
        </fieldset>

        <fieldset className="kg-choice-section">
          <legend><Icon name="tool" />工具</legend>
          <p className="field-note">每类选一件。工具差异会改变画线方式、孔径和灯片吸附范围；这些仍是游戏模拟，不是实物规格。</p>
          <div className="tool-category-list">
            {(["drawing", "piercing", "assembly"] as const).map((category) => (
              <div className="tool-category" key={category}>
                <span className="tool-category-label">{category === "drawing" ? "绘图" : category === "piercing" ? "针刺" : "装配"}</span>
                <div className="tool-grid">
                  {TOOLS.filter((tool) => tool.category === category).map((tool) => (
                    <button
                      aria-pressed={toolIds.includes(tool.toolId)}
                      className={`tool-card tool-${tool.model} ${toolIds.includes(tool.toolId) ? "is-selected" : ""}`}
                      key={tool.toolId}
                      onClick={() => chooseTool(tool.toolId)}
                      type="button"
                    >
                      <span className="tool-model" aria-hidden="true" />
                      <span><strong>{tool.label}</strong><small>{tool.description}</small><em>{tool.simulationEffect}</em></span>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </fieldset>

        <fieldset className="kg-choice-section">
          <legend><Icon name="sparkles" />纹样</legend>
          <p className="field-note">以下是玩家表达模板，不代表经过审核的传统纹样名称或固定文化寓意。</p>
          <div className="pattern-grid">
            {PATTERNS.filter((pattern) => pattern.motif !== "ai").map((pattern) => (
              <button
                aria-pressed={patternId === pattern.patternId}
                className={`pattern-card motif-${pattern.motif} ${patternId === pattern.patternId ? "is-selected" : ""}`}
                key={pattern.patternId}
                onClick={() => setPatternId(pattern.patternId)}
                type="button"
              >
                <span className="pattern-preview" aria-hidden="true" />
                <span><small>玩家表达</small><strong>{pattern.label}</strong><p>{pattern.description}</p></span>
              </button>
            ))}
          </div>
          <div className="ai-pattern-studio">
            <div className="ai-pattern-mark"><Icon name="sparkles" /><span>AI 共创</span></div>
            <div className="ai-illustration-presets" aria-label="插画灵感">
              {ILLUSTRATION_PRESETS.map((preset, index) => (
                <button aria-label={`使用插画灵感 ${index + 1}`} disabled={patternLoading} key={preset.prompt} onClick={() => setPatternPrompt(preset.prompt)} title={preset.prompt} type="button"><Icon name={preset.icon} /></button>
              ))}
            </div>
            <input
              aria-label="描述想要的灯笼图案"
              disabled={patternLoading}
              maxLength={160}
              onChange={(event) => setPatternPrompt(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); void createPattern(); } }}
              placeholder="描述一幅插画场景"
              value={patternPrompt}
            />
            <button className="icon-button is-primary" aria-label="生成图案" disabled={patternPrompt.trim().length < 2 || patternLoading} onClick={() => void createPattern()} title="生成图案" type="button"><Icon name={patternLoading ? "pause" : "sparkles"} /></button>
            {state.generatedPattern?.imageDataUrl ? (
              <button
                aria-label="使用刚生成的图案"
                aria-pressed={patternId === "pattern-ai-generated"}
                className={`ai-pattern-result ${patternId === "pattern-ai-generated" ? "is-selected" : ""}`}
                onClick={() => setPatternId("pattern-ai-generated")}
                title="这幅 AI 插画将贴到六片灯面"
                type="button"
              >
                <img alt="刚生成的 AI 插画纹样" src={state.generatedPattern.imageDataUrl} />
                <span className="ai-pattern-location"><Icon name="lantern" /><strong>AI 插画</strong><small>→ 六片灯面</small></span>
                <i><Icon name="check" /></i>
              </button>
            ) : null}
            <small className={patternError ? "is-error" : ""}>{patternError || (patternLoading ? "正在绘制插画…" : "个人表达 · 生成后铺到六片灯面")}</small>
          </div>
        </fieldset>

        <fieldset className="kg-choice-section">
          <legend><Icon name="evidence" />依据</legend>
        <div className="reference-grid">
          {REFERENCES.map((reference, index) => (
            <button
              className={`reference-card accent-${reference.accent} ${selectedId === reference.referenceId ? "is-selected" : ""}`}
              key={reference.referenceId}
              onClick={() => setSelectedId(reference.referenceId)}
              type="button"
            >
              <span className="reference-number">0{index + 1}</span>
              <span className="reference-mark" aria-hidden="true" />
              <strong>{reference.title}</strong>
              <small>{reference.subtitle}</small>
              <p>{reference.description}</p>
            </button>
          ))}
        </div>
        </fieldset>
        <div className="screen-actions">
          <button aria-label="返回意图卡" className="button button-secondary compact-button" onClick={() => dispatch({ type: "GO_TO_ACT", act: "wish" })} title="返回" type="button"><Icon name="previous" /><span>返回</span></button>
          <span className="selection-progress" title="材料、工具、纹样与依据">{[selectedMaterial?.selectable, toolIds.length === 3, selectedPattern, selectedReference].filter(Boolean).length}/4</span>
          <button aria-label="把材料和工具带到 3D 工作台" className="button button-primary compact-button" disabled={!canConfirm} onClick={confirm} title="进入 3D 工作台" type="button"><Icon name="arrow" /><span>开始</span></button>
        </div>
      </div>
      <EvidencePanel dispatch={dispatch} items={evidence} />
    </section>
  );
}

function PlanCard({
  plan,
  onAccept,
  onReject,
  readOnly,
}: {
  plan: PlanOption;
  onAccept: () => void;
  onReject: () => void;
  readOnly: boolean;
}) {
  return (
    <article className="plan-card">
      <div className="plan-topline"><span><Icon name="sparkles" />路径</span><small>{plan.action_ids.length}</small></div>
      <h3>{plan.title}</h3>
      <dl className="plan-facts">
        <div title="保留意图"><dt><Icon name="lock" /></dt><dd>{plan.retained_intent_ids.length}</dd></div>
        <div title="需要同意"><dt><Icon name="hand" /></dt><dd>{plan.required_consent.length || 0}</dd></div>
        <div title="返工影响"><dt><Icon name="repeat" /></dt><dd>{plan.rework_cost ?? 0}</dd></div>
      </dl>
      <details className="plan-detail"><summary aria-label="查看方案说明" title="查看方案说明"><Icon name="info" /></summary><p>{plan.summary}</p><small>{plan.tradeoff}</small></details>
      <div className="plan-actions">
        <button aria-label="暂不采用" className="button button-quiet compact-button" disabled={readOnly} onClick={onReject} title="暂不采用" type="button"><Icon name="close" /><span>略过</span></button>
        <button aria-label="采用这条路径" className="button button-primary compact-button" disabled={readOnly} onClick={onAccept} title="采用" type="button"><Icon name="check" /><span>采用</span></button>
      </div>
    </article>
  );
}

function CraftScreen({ state, dispatch }: { state: GameState; dispatch: Dispatch }) {
  const [decisionLoading, setDecisionLoading] = useState(false);
  const [decisionError, setDecisionError] = useState("");

  async function refreshDecision(): Promise<void> {
    if (
      !state.aiAssistanceEnabled ||
      decisionLoading ||
      state.revisionStage === "change_pending"
    ) return;
    setDecisionLoading(true);
    setDecisionError("");
    dispatch({
      type: "LOG_EVENT",
      eventType: "decision_requested",
      payload: {
        stateVersion: state.stateVersion,
        intentVersion: state.intentVersion,
      },
    });
    try {
      const decision = await requestDecision(state);
      dispatch({ type: "RECEIVE_DECISION", decision });
    } catch (error) {
      const code = error instanceof DecisionClientError ? error.code : "unknown";
      const message =
        error instanceof Error ? error.message : "决策服务暂时不可用，请重试。";
      setDecisionError(message);
      dispatch({
        type: "LOG_EVENT",
        eventType: "decision_request_failed",
        payload: { code },
      });
    } finally {
      setDecisionLoading(false);
    }
  }

  useEffect(() => {
    if (!state.currentDecision && state.aiAssistanceEnabled) {
      void refreshDecision();
    }
    // The version tuple is the request identity; events written during the request
    // must not start a duplicate call.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.stateVersion, state.intentVersion, state.currentDecision, state.aiAssistanceEnabled]);

  const evidenceIds = state.currentDecision?.plans.flatMap((plan) => plan.evidence_ids) ?? [];
  const evidence = EVIDENCE.filter((item) => evidenceIds.includes(item.evidenceId));
  const material = MATERIALS.find((item) => item.materialId === state.selectedMaterialId);
  const pattern = PATTERNS.find((item) => item.patternId === state.selectedPatternId);

  useEffect(() => {
    if (
      state.craftProgress.phase === "complete" &&
      state.revisionStage === "initial" &&
      state.currentDecision?.plans[0]
    ) {
      dispatch({ type: "APPLY_PLAN", plan: state.currentDecision.plans[0] });
    }
  }, [dispatch, state.craftProgress.phase, state.currentDecision, state.revisionStage]);

  function accept(plan: PlanOption): void {
    const isRevision = state.revisionStage === "change_active";
    dispatch({ type: "APPLY_PLAN", plan });
    if (isRevision) dispatch({ type: "GO_TO_ACT", act: "showcase" });
  }

  return (
    <section className="craft-experience-layout">
      <Workshop3D
        generatedPatternUrl={state.generatedPattern?.imageDataUrl}
        initialProgress={state.craftProgress}
        materialId={state.selectedMaterialId ?? "material-warm-translucent"}
        onComplete={(progress) => dispatch({ type: "LOG_EVENT", eventType: "craft_3d_completed", payload: { strokes: progress.strokes, holes: progress.holes, assembledPanels: progress.assembledPanels } })}
        onProgress={(progress, reason) => dispatch({ type: "UPDATE_CRAFT_PROGRESS", progress, reason })}
        patternId={state.selectedPatternId ?? "pattern-freehand"}
        readOnly={state.noAiStageStarted}
        toolIds={state.selectedToolIds}
      />
      <div className="screen-card craft-guide-column">
        <div className="section-heading compact">
          <span className="eyebrow">第三幕</span>
          <div className="icon-heading"><h1>{state.craftProgress.phase === "complete" ? "点灯" : "制作"}</h1><Icon name={state.craftProgress.phase === "complete" ? "lantern" : "hand"} /></div>
        </div>
        {decisionLoading && <p className="decision-status">正在核对状态、意图和依据……</p>}
        {decisionError && <p className="decision-error" role="alert">{decisionError}</p>}
        <div className="intent-summary">
          {state.intents.map((intent) => (
            <span className={intent.locked ? "is-locked" : ""} key={intent.intentId}>{intent.locked ? "已锁定" : "可商量"} · {intent.label}</span>
          ))}
        </div>
        <div className="kg-trace" aria-label="本轮知识图谱推理路径">
          <span className="kg-trace-title">KG 现场依据</span>
          <span>{material?.label ?? "材料未选"}</span><b>→</b>
          <span>{pattern?.label ?? "图案未选"}</span><b>→</b>
          <span>{state.intents.find((intent) => intent.intentId.includes("pattern"))?.label ?? "图案策略"}</span><b>→</b>
          <span>{state.intents.find((intent) => intent.intentId.includes("brighter"))?.label ?? "透光区域"}</span>
        </div>
        {state.revisionStage === "initial" && state.currentDecision?.plans[0] && (
          <details className="craft-brief">
            <summary><Icon name="sparkles" /><strong>{state.currentDecision.plans[0].title}</strong><Icon name="info" /></summary>
            <p>{state.currentDecision.plans[0].summary}</p>
            <small>{state.currentDecision.plans[0].tradeoff}</small>
          </details>
        )}
        {state.revisionStage === "change_pending" && (
          <article className="plan-card change-card">
            <div className="plan-topline"><span><Icon name="repeat" />新委托</span><small>固定</small></div>
            <h3>新增一片 · 保留旧作</h3>
            <details className="plan-detail"><summary aria-label="查看变化说明" title="查看说明"><Icon name="info" /></summary><p>增加一张不同图案的灯片，并尽量保留刚才完成的纹样、针孔和装配结果。</p></details>
            <div className="plan-actions">
              <button aria-label="接受新委托并比较路径" className="button button-primary compact-button" onClick={() => dispatch({ type: "TRIGGER_REVISION" })} type="button"><Icon name="arrow" /><span>比较</span></button>
            </div>
          </article>
        )}
        {state.noAiStageStarted && (
          <div className="no-ai-banner"><span aria-hidden="true">●</span> 讲灯阶段已开始：这里只能回看，不能重生成或改动方案。</div>
        )}
        <div className="plan-list">
          {(state.revisionStage === "change_active" || (state.noAiStageStarted && state.revisionStage === "complete")) && state.currentDecision?.plans.map((plan) => (
            <PlanCard
              key={plan.plan_id}
              onAccept={() => accept(plan)}
              onReject={() => dispatch({ type: "LOG_EVENT", eventType: "plan_rejected", payload: { planId: plan.plan_id } })}
              plan={plan}
              readOnly={state.noAiStageStarted}
            />
          ))}
        </div>
        <div className="screen-actions">
          {state.craftProgress.phase === "design" && (
            <button aria-label="返回更换材料与工具" className="button button-secondary compact-button" onClick={() => dispatch({ type: "GO_TO_ACT", act: "reference" })} title="返回选材" type="button"><Icon name="previous" /><span>选材</span></button>
          )}
          <button
            aria-label="重新核对变化方案"
            className="button button-quiet compact-button"
            disabled={!state.aiAssistanceEnabled || decisionLoading || state.revisionStage !== "change_active"}
            onClick={() => void refreshDecision()}
            title="重新核对"
            type="button"
          ><Icon name="repeat" /><span>{decisionLoading ? "核对中" : "核对"}</span></button>
          {state.noAiStageStarted && (
            <button className="button button-primary" onClick={() => dispatch({ type: "GO_TO_ACT", act: "showcase" })} type="button">返回讲灯</button>
          )}
        </div>
      </div>
      <div className="craft-evidence"><EvidencePanel dispatch={dispatch} items={evidence} /></div>
    </section>
  );
}

function ShowcaseScreen({ state, dispatch }: { state: GameState; dispatch: Dispatch }) {
  const [explanation, setExplanation] = useState(state.independentExplanation);
  const enough = explanation.trim().length >= 20;
  function continueToHandoff(): void {
    dispatch({ type: "SET_EXPLANATION", value: explanation });
    dispatch({ type: "GO_TO_ACT", act: "handoff" });
  }
  return (
    <section className="screen-layout showcase-layout">
      <div className="showcase-3d-stage">
        <Workshop3D
          displayOnly
          generatedPatternUrl={state.generatedPattern?.imageDataUrl}
          initialProgress={state.craftProgress}
          materialId={state.selectedMaterialId ?? "material-warm-translucent"}
          onComplete={() => undefined}
          onProgress={() => undefined}
          patternId={state.selectedPatternId ?? "pattern-freehand"}
          readOnly
          toolIds={state.selectedToolIds}
        />
      </div>
      <div className="screen-card main-column">
        <div className="section-heading">
          <span className="eyebrow">第四幕 · 点灯与讲灯</span>
          <h1>现在由你自己讲述</h1>
          <p>这一阶段关闭 AI 帮助。请区分参照、自己的选择和仍不确定的部分。</p>
        </div>
        <div className="no-ai-banner"><span aria-hidden="true">●</span> 无 AI 讲述阶段 · 系统不会生成或改写你的文字</div>
        <label className="text-field">
          <span>我的讲述</span>
          <textarea
            maxLength={800}
            onChange={(event) => setExplanation(event.target.value)}
            onBlur={() => dispatch({ type: "SET_EXPLANATION", value: explanation })}
            placeholder="哪些来自参照？哪些是你的选择？还有什么不能确定？"
            rows={8}
            value={explanation}
          />
          <small>{explanation.length}/800 字；至少填写 20 字以继续。</small>
        </label>
        <div className="screen-actions">
          <button className="button button-secondary" onClick={() => dispatch({ type: "GO_TO_ACT", act: "craft" })} type="button">回看制作选择</button>
          <button className="button button-primary" disabled={!enough} onClick={continueToHandoff} type="button">完成讲述，进入传灯</button>
        </div>
      </div>
    </section>
  );
}

function HandoffScreen({
  state,
  dispatch,
  onExport,
}: {
  state: GameState;
  dispatch: Dispatch;
  onExport: () => void;
}) {
  const completed = Boolean(state.completedAt);
  return (
    <section className="screen-card handoff-card">
      <div className="handoff-3d-visual">
        <Workshop3D
          displayOnly
          generatedPatternUrl={state.generatedPattern?.imageDataUrl}
          initialProgress={state.craftProgress}
          materialId={state.selectedMaterialId ?? "material-warm-translucent"}
          onComplete={() => undefined}
          onProgress={() => undefined}
          patternId={state.selectedPatternId ?? "pattern-freehand"}
          readOnly
          toolIds={state.selectedToolIds}
        />
      </div>
      <div className="handoff-content">
        <div className="section-heading">
          <span className="eyebrow">第五幕 · 传灯</span>
          <h1>{completed ? "这次灯彩旅程已记录" : "展示是一项单独选择"}</h1>
          <p>完成游戏不等于同意公开。你可以不授权展示，仍然完整结束本次体验。</p>
        </div>
        <div className="session-summary">
          <div><span>已确认意图</span><strong>{state.intents.length}</strong></div>
          <div><span>采用路径</span><strong>{state.acceptedPlanId ? "1" : "0"}</strong></div>
          <div><span>记录事件</span><strong>{state.events.length}</strong></div>
        </div>
        <label className="consent-card">
          <input
            checked={state.shareConsent}
            disabled={completed}
            onChange={(event) => dispatch({ type: "SET_SHARE_CONSENT", value: event.target.checked })}
            type="checkbox"
          />
          <span><strong>我愿意展示作品与讲述</strong><small>这是可撤回的开发占位授权；正式研究将使用单独同意文件。</small></span>
        </label>
        <div className="explanation-preview"><span>我的讲述</span><p>{state.independentExplanation}</p></div>
        <div className="screen-actions">
          {!completed ? (
            <button className="button button-primary" onClick={() => dispatch({ type: "COMPLETE_SESSION" })} type="button">完成本次体验</button>
          ) : (
            <button className="button button-primary" onClick={onExport} type="button">导出匿名会话记录</button>
          )}
        </div>
      </div>
    </section>
  );
}

function Screen({ state, dispatch, onExport }: { state: GameState; dispatch: Dispatch; onExport: () => void }) {
  switch (state.currentAct) {
    case "wish": return <WishScreen dispatch={dispatch} state={state} />;
    case "reference": return <ReferenceScreen dispatch={dispatch} state={state} />;
    case "craft": return <CraftScreen dispatch={dispatch} state={state} />;
    case "showcase": return <ShowcaseScreen dispatch={dispatch} state={state} />;
    case "handoff": return <HandoffScreen dispatch={dispatch} onExport={onExport} state={state} />;
  }
}

export default function App() {
  const { state, dispatch, reset, exportSession } = useSession();
  const background = useMemo(() => {
    if (state.currentAct === "wish") return "bg-home";
    if (state.currentAct === "reference") return "bg-gallery";
    if (state.currentAct === "craft") return "bg-craft";
    return "bg-festival";
  }, [state.currentAct]);

  return (
    <div className={`app-shell ${background}`}>
      <div className="background-shade" aria-hidden="true" />
      <header className="app-header">
        <a className="brand" href="./" aria-label="LanternQuest 首页">
          <span className="brand-mark" aria-hidden="true">灯</span>
          <span><strong>LanternQuest</strong><small>灯火相传 · 本地研究原型</small></span>
        </a>
        <ChapterProgress dispatch={dispatch} state={state} />
        <ResearchPanel onExport={exportSession} onReset={reset} state={state} />
      </header>
      <main className="app-main">
        <Screen dispatch={dispatch} onExport={exportSession} state={state} />
      </main>
      <footer className="app-footer">
        <span aria-label="数字体验不等同于掌握真实工艺" title="数字体验不等同于掌握真实工艺"><Icon name="evidence" /></span>
        <span aria-label={state.aiAssistanceEnabled ? "资料助手可用" : "AI 帮助已关闭"} title={state.aiAssistanceEnabled ? "资料助手可用" : "AI 帮助已关闭"}><Icon name={state.aiAssistanceEnabled ? "sparkles" : "lock"} /></span>
      </footer>
    </div>
  );
}
