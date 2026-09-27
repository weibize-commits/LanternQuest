import type {
  EvidenceCard,
  MaterialOption,
  PatternOption,
  ReferenceOption,
  ToolOption,
} from "./types";

export const ACTS = [
  { id: "wish", label: "接灯愿", short: "意图" },
  { id: "reference", label: "寻灯样", short: "参照" },
  { id: "craft", label: "造自己的灯", short: "制作" },
  { id: "showcase", label: "点灯与讲灯", short: "讲述" },
  { id: "handoff", label: "传灯", short: "授权" },
] as const;

export const COMMISSIONS = [
  {
    id: "learning",
    title: "制作一件局部学习样片",
    description: "在已审核材料和现场监督范围内体验针刺透光关系。",
  },
  {
    id: "explanation",
    title: "为参观者解释亮暗关系",
    description: "用一件局部样片说明图案、针刺区域与点亮效果。",
  },
  {
    id: "comparison",
    title: "比较两种灯片设计路径",
    description: "先明确图案重复方式，再选择主体或背景的透光重点。",
  },
] as const;

export const INTENT_CHOICES = [
  {
    id: "intent-repeat-pattern",
    dimension: "pattern",
    label: "各灯片使用相同图案",
    statement: "保留相同图案并采用叠放定位的制作路径。",
  },
  {
    id: "intent-vary-patterns",
    dimension: "pattern",
    label: "各灯片使用不同图案",
    statement: "保留不同图案并采用逐片处理的制作路径。",
  },
  {
    id: "intent-subject-brighter",
    dimension: "light",
    label: "点亮后图案主体更亮",
    statement: "保留主体区域透光的亮暗选择。",
  },
  {
    id: "intent-background-brighter",
    dimension: "light",
    label: "点亮后背景更亮",
    statement: "保留背景透光并突出较暗主体轮廓的选择。",
  },
] as const;

export const MATERIALS: MaterialOption[] = [
  {
    materialId: "material-warm-translucent",
    label: "暖白半透纸样",
    description: "透光较强、耐受较低的数字纸面，用于观察密集针孔效果。",
    kgNodeId: "task:task_myl_needle_piercing_v0",
    knowledgeStatus: "simulation_only",
    selectable: true,
    evidenceIds: ["ev_n001_depth_control", "ev_field06_hands_on_boundary"],
    surfaceColor: "#ead8b8",
    transmission: 0.82,
    durability: 0.42,
  },
  {
    materialId: "material-natural-fiber",
    label: "原色纤维纸样",
    description: "透光与耐受均衡的数字纸面，表面保留可见纤维感。",
    kgNodeId: "task:task_myl_needle_piercing_v0",
    knowledgeStatus: "simulation_only",
    selectable: true,
    evidenceIds: ["ev_n001_depth_control", "ev_field06_hands_on_boundary"],
    surfaceColor: "#cbb991",
    transmission: 0.62,
    durability: 0.68,
  },
  {
    materialId: "material-indigo-translucent",
    label: "靛青半透纸样",
    description: "透光较低、耐受较高的数字纸面，点灯后孔洞对比更明显。",
    kgNodeId: "task:task_myl_needle_piercing_v0",
    knowledgeStatus: "simulation_only",
    selectable: true,
    evidenceIds: ["ev_n001_depth_control", "ev_field06_hands_on_boundary"],
    surfaceColor: "#547084",
    transmission: 0.38,
    durability: 0.86,
  },
  {
    materialId: "material-unverified-real-paper",
    label: "指定真实纸种（待专家补证）",
    description: "正式 KG 尚未确认纸种、厚度和安全边界，当前不能把它写成真实工艺参数。",
    kgNodeId: "unknown:task_myl_needle_piercing_v0:001",
    knowledgeStatus: "unknown_pending",
    selectable: false,
    evidenceIds: [],
    surfaceColor: "#b5aa95",
    transmission: 0.5,
    durability: 0.5,
  },
];

export const PATTERNS: PatternOption[] = [
  {
    patternId: "pattern-ai-generated",
    label: "AI 共创纹",
    description: "根据玩家的一句话生成，只作为个人表达。",
    provenance: "player_expression",
    motif: "ai",
  },
  {
    patternId: "pattern-geometric-rhythm",
    label: "几何连续纹",
    description: "用点、线与重复节奏组织自己的图案。",
    provenance: "player_expression",
    motif: "geometry",
  },
  {
    patternId: "pattern-botanical-outline",
    label: "植物轮廓纹",
    description: "以叶片或枝条轮廓构成个人图案。",
    provenance: "player_expression",
    motif: "botanical",
  },
  {
    patternId: "pattern-memory-symbol",
    label: "个人记忆符号",
    description: "用一个与自己有关的符号形成图案。",
    provenance: "player_expression",
    motif: "memory",
  },
  {
    patternId: "pattern-freehand",
    label: "从空白开始",
    description: "不载入模板，直接在三维纸样上自由绘制。",
    provenance: "player_expression",
    motif: "freeform",
  },
];

export const TOOLS: ToolOption[] = [
  {
    toolId: "tool-freehand-stylus",
    category: "drawing",
    label: "自由纹样笔",
    description: "在纸样上连续拖动，自己画出线条。",
    simulationEffect: "自由绘制，线条较细",
    model: "stylus",
  },
  {
    toolId: "tool-guided-stencil",
    category: "drawing",
    label: "定位纹样板",
    description: "保留模板轮廓，再由玩家补画和修改。",
    simulationEffect: "提供轮廓辅助",
    model: "stencil",
  },
  {
    toolId: "tool-fine-awl",
    category: "piercing",
    label: "细孔针（模拟）",
    description: "形成较小透光点，适合密集节奏。",
    simulationEffect: "孔径 5 px",
    model: "fine_awl",
  },
  {
    toolId: "tool-broad-awl",
    category: "piercing",
    label: "宽孔锥（模拟）",
    description: "形成较大透光点，亮暗对比更强。",
    simulationEffect: "孔径 10 px",
    model: "broad_awl",
  },
  {
    toolId: "tool-edge-clips",
    category: "assembly",
    label: "压边夹（模拟）",
    description: "靠近灯架时扩大吸附范围，便于定位。",
    simulationEffect: "装片吸附范围更大",
    model: "clip",
  },
  {
    toolId: "tool-direct-hands",
    category: "assembly",
    label: "徒手贴合（模拟）",
    description: "需要更准确地把灯片拖到框架位置。",
    simulationEffect: "吸附范围更小，操作更自由",
    model: "hand",
  },
];

export const REFERENCES: ReferenceOption[] = [
  {
    referenceId: "ref-pattern-branch",
    title: "参照 A · 图案安排",
    subtitle: "专家确认的流程分支",
    description: "比较相同图案叠放针刺与不同图案逐片针刺。",
    evidenceIds: ["ev_n001_pattern_branch"],
    accent: "cinnabar",
  },
  {
    referenceId: "ref-light-branch",
    title: "参照 B · 主体与背景亮暗",
    subtitle: "专家确认的光效关系",
    description: "比较主体透光与背景透光对应的视觉结果。",
    evidenceIds: ["ev_n001_light_branch"],
    accent: "celadon",
  },
  {
    referenceId: "ref-practice-boundary",
    title: "参照 C · 局部实践边界",
    subtitle: "专家确认的操作范围",
    description: "了解垫板、孔洞均匀性和数字体验不能替代实操的边界。",
    evidenceIds: ["ev_n001_depth_control", "ev_field06_hands_on_boundary"],
    accent: "indigo",
  },
];

export const EVIDENCE: EvidenceCard[] = [
  {
    evidenceId: "ev_n001_pattern_branch",
    title: "图案安排证据卡",
    summary: "图案相同的六片灯片可以叠放定位和针刺；图案不同则需要逐片针刺。",
    scope: "仅支持 N001 示范的六片万眼萝灯片，不自动推广到所有灯型和材料。",
    sourceLabel: "N001 视频 10:57–12:05；关键帧 11:44",
    reviewStatus: "domain_approved",
    media: [
      {
        assetId: "src_51c63e8ce4b1d5af",
        label: "11:44 相同图案叠片对齐",
        alt: "N001 示范中相同图案灯片叠放对齐的审核关键帧",
        endpoint: "/evidence/ev_n001_pattern_branch/keyframe-1144.jpg",
      },
    ],
  },
  {
    evidenceId: "ev_n001_light_branch",
    title: "亮暗关系证据卡",
    summary: "阳扎使图案主体区域更亮；阴扎使背景更亮并突出较暗的图案轮廓。",
    scope: "支持针刺区域与亮暗效果的关系，不支持固定审美优劣。",
    sourceLabel: "N001 视频 12:07–12:45；关键帧 12:18、12:40",
    reviewStatus: "domain_approved",
    media: [
      {
        assetId: "src_abc7513f16332e0b",
        label: "12:18 沿图案针刺",
        alt: "N001 示范中沿图案主体区域针刺的审核关键帧",
        endpoint: "/evidence/ev_n001_light_branch/keyframe-1218.jpg",
      },
      {
        assetId: "src_a8f853229e31ecce",
        label: "12:40 图案背景针刺",
        alt: "N001 示范中在图案背景区域针刺的审核关键帧",
        endpoint: "/evidence/ev_n001_light_branch/keyframe-1240.jpg",
      },
    ],
  },
  {
    evidenceId: "ev_n001_depth_control",
    title: "局部针刺证据卡",
    summary: "示范使用垫板帮助控制下针深度，并追求针孔大小与饱满程度均匀。",
    scope: "资料没有给出可普遍套用的孔径、针距或具体下针深度。",
    sourceLabel: "N001 视频 12:47–13:16；关键帧 12:58",
    reviewStatus: "domain_approved",
    media: [
      {
        assetId: "src_d01df7f40b72bec7",
        label: "12:58 垫板与下针控制",
        alt: "N001 示范中使用垫板辅助控制下针的审核关键帧",
        endpoint: "/evidence/ev_n001_depth_control/keyframe-1258.jpg",
      },
    ],
  },
  {
    evidenceId: "ev_field06_process_check",
    title: "流程范围证据卡",
    summary: "传承人对现场展示的万眼萝九步整理作了总体认可。",
    scope: "这是总体认可，不能写成九步中每一项细节均被逐项确认。",
    sourceLabel: "田野 06 音频 08:39–09:07",
    reviewStatus: "domain_approved",
  },
  {
    evidenceId: "ev_field07_complexity",
    title: "复杂度边界证据卡",
    summary: "传承人明确指出万眼萝工艺复杂，因此数字任务缩小为受控的局部样片。",
    scope: "不提供局部任务的具体教学时长，也不代表掌握完整工艺。",
    sourceLabel: "田野 07 音频 00:24–00:27",
    reviewStatus: "domain_approved",
  },
  {
    evidenceId: "ev_field06_hands_on_boundary",
    title: "动手边界证据卡",
    summary: "数字方式可以呈现制作过程，但实际操作仍需要上手。",
    scope: "数字观看或模拟不能被表述为已经掌握真实制灯技能。",
    sourceLabel: "田野 06 音频 12:27–12:46",
    reviewStatus: "domain_approved",
  },
];
