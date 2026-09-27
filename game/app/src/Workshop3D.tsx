import { useEffect, useRef, useState, type CSSProperties } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRButton } from "three/addons/webxr/VRButton.js";

import { MATERIALS, PATTERNS, TOOLS } from "./data";
import { Icon, type IconName } from "./Icon";
import type { CraftMark, CraftProgress } from "./types";

interface Workshop3DProps {
  displayOnly?: boolean;
  generatedPatternUrl?: string | null;
  initialProgress: CraftProgress;
  materialId: string;
  patternId: string;
  readOnly: boolean;
  toolIds: string[];
  onProgress: (
    progress: CraftProgress,
    reason: "draw" | "pierce" | "phase" | "assemble" | "complete",
  ) => void;
  onComplete: (progress: CraftProgress) => void;
}

const PHASE_LABELS: Record<CraftProgress["phase"], string> = {
  design: "绘制纹样",
  pierce: "针刺透光",
  assemble: "装配灯片",
  complete: "点灯检查",
};

const FRAME_RADIUS = 1.27;
const PANEL_RADIUS = FRAME_RADIUS * Math.cos(Math.PI / 6);
const PANEL_WIDTH = FRAME_RADIUS + 0.04;
const PANEL_HEIGHT = 2.72;

function placePanelOnLantern(panel: THREE.Mesh, index: number): void {
  const angle = ((index + 0.5) / 6) * Math.PI * 2;
  panel.position.set(Math.sin(angle) * PANEL_RADIUS, 2, Math.cos(angle) * PANEL_RADIUS);
  panel.rotation.set(0, angle, 0);
  panel.scale.setScalar(1);
}

function createToolModel(model: string): THREE.Group {
  const group = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ color: 0x7b4e32, roughness: 0.72 });
  const metal = new THREE.MeshStandardMaterial({ color: 0xaeb7b9, metalness: 0.72, roughness: 0.28 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x243846, roughness: 0.65 });
  if (model === "stylus" || model === "fine_awl" || model === "broad_awl") {
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.12, 1.15, 18), wood);
    handle.rotation.z = Math.PI / 2;
    const tipLength = model === "stylus" ? 0.32 : 0.5;
    const tipRadius = model === "broad_awl" ? 0.12 : 0.07;
    const tip = new THREE.Mesh(new THREE.ConeGeometry(tipRadius, tipLength, 18), metal);
    tip.rotation.z = -Math.PI / 2;
    tip.position.x = 0.82;
    group.add(handle, tip);
  } else if (model === "stencil") {
    const plate = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.08, 0.72), dark);
    const inset = new THREE.Mesh(
      new THREE.TorusGeometry(0.2, 0.035, 10, 32),
      new THREE.MeshStandardMaterial({ color: 0xd5a64a, metalness: 0.4 }),
    );
    inset.rotation.x = Math.PI / 2;
    inset.position.y = 0.06;
    group.add(plate, inset);
  } else if (model === "clip") {
    for (const offset of [-0.24, 0.24]) {
      const jaw = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.7, 0.22), metal);
      jaw.position.x = offset;
      group.add(jaw);
    }
    const bridge = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.14, 0.24), wood);
    bridge.position.y = 0.35;
    group.add(bridge);
  } else {
    const palm = new THREE.Mesh(new THREE.SphereGeometry(0.34, 20, 14), wood);
    palm.scale.set(1, 0.72, 0.35);
    group.add(palm);
  }
  return group;
}

function paintBase(
  context: CanvasRenderingContext2D,
  materialId: string,
  patternId: string,
  guided: boolean,
): void {
  const material = MATERIALS.find((item) => item.materialId === materialId);
  const pattern = PATTERNS.find((item) => item.patternId === patternId);
  const size = context.canvas.width;
  context.fillStyle = material?.surfaceColor ?? "#ead8b8";
  context.fillRect(0, 0, size, size);

  context.save();
  context.globalAlpha = 0.12;
  context.strokeStyle = materialId.includes("indigo") ? "#e8ddbd" : "#6f593d";
  context.lineWidth = 1.2;
  for (let index = 0; index < 80; index += 1) {
    const y = (index * 83) % size;
    const bend = (index % 7) * 9;
    context.beginPath();
    context.moveTo(0, y);
    context.bezierCurveTo(size * 0.3, y + bend, size * 0.7, y - bend, size, y + 4);
    context.stroke();
  }
  context.restore();

  if (!pattern || pattern.motif === "freeform") return;
  context.save();
  context.globalAlpha = guided ? 0.55 : 0.22;
  context.strokeStyle = materialId.includes("indigo") ? "#f0d8a2" : "#79483b";
  context.lineWidth = guided ? 7 : 4;
  context.setLineDash(guided ? [16, 9] : [9, 14]);
  if (pattern.motif === "geometry") {
    for (let y = 90; y < size; y += 130) {
      context.beginPath();
      for (let x = 40; x <= size - 40; x += 65) {
        const yy = y + ((x / 65) % 2 === 0 ? -32 : 32);
        if (x === 40) context.moveTo(x, yy);
        else context.lineTo(x, yy);
      }
      context.stroke();
    }
  } else if (pattern.motif === "botanical") {
    context.beginPath();
    context.moveTo(size * 0.18, size * 0.86);
    context.bezierCurveTo(size * 0.32, size * 0.58, size * 0.46, size * 0.46, size * 0.72, size * 0.14);
    context.stroke();
    for (let index = 0; index < 5; index += 1) {
      const x = size * (0.29 + index * 0.095);
      const y = size * (0.7 - index * 0.115);
      context.beginPath();
      context.ellipse(x, y, 48, 22, index % 2 ? -0.55 : 0.55, 0, Math.PI * 2);
      context.stroke();
    }
  } else {
    for (const [x, y] of [[0.3, 0.34], [0.67, 0.34], [0.3, 0.7], [0.67, 0.7]]) {
      context.beginPath();
      context.arc(size * x, size * y, 62, 0, Math.PI * 2);
      context.stroke();
      context.beginPath();
      context.arc(size * x, size * y, 28, 0, Math.PI * 2);
      context.stroke();
    }
  }
  context.restore();
}

function paintMarks(context: CanvasRenderingContext2D, marks: CraftMark[]): void {
  const size = context.canvas.width;
  const strokeGroups = new Map<number, CraftMark[]>();
  for (const mark of marks) {
    if (mark.kind === "hole") {
      const gradient = context.createRadialGradient(
        mark.x * size,
        mark.y * size,
        0,
        mark.x * size,
        mark.y * size,
        mark.size * 2.3,
      );
      gradient.addColorStop(0, "#fffbe9");
      gradient.addColorStop(0.45, "#ffd47a");
      gradient.addColorStop(1, "rgba(213,166,74,0)");
      context.fillStyle = gradient;
      context.beginPath();
      context.arc(mark.x * size, mark.y * size, mark.size * 2.3, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#fffdf5";
      context.beginPath();
      context.arc(mark.x * size, mark.y * size, mark.size, 0, Math.PI * 2);
      context.fill();
      continue;
    }
    const group = strokeGroups.get(mark.strokeId) ?? [];
    group.push(mark);
    strokeGroups.set(mark.strokeId, group);
  }
  context.strokeStyle = "#5e2d28";
  context.lineCap = "round";
  context.lineJoin = "round";
  for (const group of strokeGroups.values()) {
    if (group.length < 2) continue;
    context.lineWidth = group[0].size;
    context.beginPath();
    context.moveTo(group[0].x * size, group[0].y * size);
    for (const point of group.slice(1)) context.lineTo(point.x * size, point.y * size);
    context.stroke();
  }
}

function cloneProgress(progress: CraftProgress): CraftProgress {
  return { ...progress, marks: [...progress.marks] };
}

function playIgnitionChime(): void {
  try {
    const AudioContextClass = globalThis.AudioContext;
    if (!AudioContextClass) return;
    const audio = new AudioContextClass();
    const master = audio.createGain();
    master.gain.setValueAtTime(0.0001, audio.currentTime);
    master.gain.exponentialRampToValueAtTime(0.075, audio.currentTime + 0.05);
    master.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 1.55);
    master.connect(audio.destination);
    [392, 523.25, 659.25].forEach((frequency, index) => {
      const oscillator = audio.createOscillator();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      oscillator.connect(master);
      oscillator.start(audio.currentTime + index * 0.12);
      oscillator.stop(audio.currentTime + 1.45);
    });
    globalThis.setTimeout(() => void audio.close(), 1700);
  } catch {
    // The visual lighting ritual remains available when audio is blocked.
  }
}

export function Workshop3D({
  displayOnly = false,
  generatedPatternUrl = null,
  initialProgress,
  materialId,
  patternId,
  readOnly,
  toolIds,
  onProgress,
  onComplete,
}: Workshop3DProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<CraftProgress>(cloneProgress(initialProgress));
  const phaseRef = useRef<CraftProgress["phase"]>(initialProgress.phase);
  const [phase, setPhase] = useState(initialProgress.phase);
  const [strokes, setStrokes] = useState(initialProgress.strokes);
  const [holes, setHoles] = useState(initialProgress.holes);
  const [assembledPanels, setAssembledPanels] = useState(initialProgress.assembledPanels);
  const [message, setMessage] = useState("拖动空白处旋转视角；滚轮缩放。先在纸样上画两笔。 ");
  const [igniting, setIgniting] = useState(false);

  const drawingTool = TOOLS.find(
    (tool) => tool.category === "drawing" && toolIds.includes(tool.toolId),
  );
  const piercingTool = TOOLS.find(
    (tool) => tool.category === "piercing" && toolIds.includes(tool.toolId),
  );
  const assemblyTool = TOOLS.find(
    (tool) => tool.category === "assembly" && toolIds.includes(tool.toolId),
  );
  const requiredStrokes = drawingTool?.model === "stencil" ? 1 : 2;

  function emit(
    next: CraftProgress,
    reason: "draw" | "pierce" | "phase" | "assemble" | "complete",
  ): void {
    progressRef.current = cloneProgress(next);
    phaseRef.current = next.phase;
    setPhase(next.phase);
    setStrokes(next.strokes);
    setHoles(next.holes);
    setAssembledPanels(next.assembledPanels);
    onProgress(cloneProgress(next), reason);
  }

  function changePhase(nextPhase: CraftProgress["phase"]): void {
    const next = { ...progressRef.current, phase: nextPhase };
    emit(next, "phase");
    setMessage(
      nextPhase === "pierce"
        ? "拿起针刺工具，在图案区域点击至少 8 个透光孔。"
        : nextPhase === "assemble"
          ? "抓住下方灯片，拖到六边形灯架中央；靠近时会吸附。"
          : "作品已点亮。拖动空白处，从不同角度检查自己的灯。",
    );
  }

  function igniteLantern(): void {
    if (readOnly || progressRef.current.assembledPanels < 6 || phaseRef.current !== "assemble") return;
    const complete: CraftProgress = {
      ...progressRef.current,
      phase: "complete",
      completedAt: new Date().toISOString(),
    };
    setIgniting(true);
    emit(complete, "complete");
    setMessage("灯火正在从针孔间亮起。拖动视角，完成最后的点灯检查。 ");
    onComplete(cloneProgress(complete));
    playIgnitionChime();
    navigator.vibrate?.(35);
    globalThis.setTimeout(() => setIgniting(false), 2800);
  }

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x102734);
    scene.fog = new THREE.Fog(0x102734, 9, 18);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 60);
    camera.position.set(0, 4.3, 9.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio, 1.75));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.shadowMap.enabled = true;
    renderer.xr.enabled = true;
    renderer.domElement.className = "workshop-canvas";
    mount.appendChild(renderer.domElement);

    let vrButton: HTMLElement | null = null;
    const xr = (
      navigator as Navigator & {
        xr?: { isSessionSupported: (mode: string) => Promise<boolean> };
      }
    ).xr;
    void xr?.isSessionSupported("immersive-vr").then((supported) => {
      if (!supported || !renderer.domElement.isConnected) return;
      vrButton = VRButton.createButton(renderer);
      vrButton.classList.add("workshop-vr-button");
      vrButton.textContent = "进入 VR 查看";
      mount.appendChild(vrButton);
    });

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 1.8, 0);
    controls.minDistance = 5;
    controls.maxDistance = 13;
    controls.maxPolarAngle = Math.PI * 0.49;

    scene.add(new THREE.HemisphereLight(0xfff3d6, 0x28475b, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffe1a6, 3.2);
    keyLight.position.set(-3.5, 7, 5);
    keyLight.castShadow = true;
    scene.add(keyLight);
    const lanternLight = new THREE.PointLight(0xffbe5c, 0, 8, 1.6);
    lanternLight.position.set(0, 2, 0);
    scene.add(lanternLight);
    const glowCanvas = document.createElement("canvas");
    glowCanvas.width = 128;
    glowCanvas.height = 128;
    const glowContext = glowCanvas.getContext("2d");
    if (glowContext) {
      const gradient = glowContext.createRadialGradient(64, 64, 2, 64, 64, 62);
      gradient.addColorStop(0, "rgba(255,245,183,0.95)");
      gradient.addColorStop(0.28, "rgba(255,195,78,0.48)");
      gradient.addColorStop(1, "rgba(255,142,31,0)");
      glowContext.fillStyle = gradient;
      glowContext.fillRect(0, 0, 128, 128);
    }
    const glowTexture = new THREE.CanvasTexture(glowCanvas);
    const glowMaterial = new THREE.SpriteMaterial({
      map: glowTexture,
      color: 0xffca69,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const glowSprite = new THREE.Sprite(glowMaterial);
    glowSprite.position.set(0, 2, 0);
    glowSprite.scale.set(5.2, 5.2, 1);
    scene.add(glowSprite);

    const motePositions = new Float32Array(72 * 3);
    for (let index = 0; index < 72; index += 1) {
      const angle = (index / 72) * Math.PI * 8;
      const radius = 1.5 + ((index * 37) % 100) / 100;
      motePositions[index * 3] = Math.sin(angle) * radius;
      motePositions[index * 3 + 1] = 0.35 + ((index * 53) % 100) / 24;
      motePositions[index * 3 + 2] = Math.cos(angle) * radius;
    }
    const moteGeometry = new THREE.BufferGeometry();
    moteGeometry.setAttribute("position", new THREE.BufferAttribute(motePositions, 3));
    const moteMaterial = new THREE.PointsMaterial({ color: 0xffd47a, size: 0.055, transparent: true, opacity: 0, depthWrite: false });
    const motes = new THREE.Points(moteGeometry, moteMaterial);
    scene.add(motes);

    const table = new THREE.Mesh(
      new THREE.BoxGeometry(12, 0.45, 7),
      new THREE.MeshStandardMaterial({ color: 0x4f3124, roughness: 0.86 }),
    );
    table.position.y = -0.35;
    table.receiveShadow = true;
    scene.add(table);
    const tableLines = new THREE.GridHelper(10, 20, 0x76533c, 0x5f402d);
    tableLines.position.y = -0.1;
    scene.add(tableLines);

    const textureCanvas = document.createElement("canvas");
    textureCanvas.width = 768;
    textureCanvas.height = 768;
    const textureContext = textureCanvas.getContext("2d");
    if (!textureContext) return;
    let generatedPatternImage: HTMLImageElement | null = null;
    const redrawTexture = (): void => {
      paintBase(
        textureContext,
        materialId,
        patternId,
        drawingTool?.model === "stencil",
      );
      if (generatedPatternImage) {
        textureContext.save();
        textureContext.globalAlpha = 0.88;
        textureContext.drawImage(generatedPatternImage, 0, 0, textureCanvas.width, textureCanvas.height);
        textureContext.restore();
      }
      paintMarks(textureContext, progressRef.current.marks);
    };
    redrawTexture();
    const paperTexture = new THREE.CanvasTexture(textureCanvas);
    paperTexture.colorSpace = THREE.SRGBColorSpace;
    paperTexture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    if (generatedPatternUrl) {
      const image = new Image();
      image.onload = () => {
        generatedPatternImage = image;
        redrawTexture();
        paperTexture.needsUpdate = true;
      };
      image.src = generatedPatternUrl;
    }
    const selectedMaterial = MATERIALS.find((item) => item.materialId === materialId);
    const paperMaterial = new THREE.MeshPhysicalMaterial({
      map: paperTexture,
      color: 0xffffff,
      roughness: 0.78,
      transmission: (selectedMaterial?.transmission ?? 0.6) * 0.12,
      transparent: true,
      opacity: 0.97,
      side: THREE.DoubleSide,
      emissive: new THREE.Color(0x6d421f),
      emissiveIntensity: 0.08,
    });

    const designGroup = new THREE.Group();
    const backing = new THREE.Mesh(
      new THREE.BoxGeometry(3.7, 4.35, 0.18),
      new THREE.MeshStandardMaterial({ color: 0x8b6749, roughness: 0.9 }),
    );
    backing.position.set(-0.75, 2.25, -0.17);
    backing.castShadow = true;
    designGroup.add(backing);
    const designPanel = new THREE.Mesh(new THREE.PlaneGeometry(3.25, 3.8), paperMaterial);
    designPanel.position.set(-0.75, 2.25, -0.05);
    designPanel.userData.kind = "design-panel";
    designGroup.add(designPanel);
    scene.add(designGroup);

    const toolStand = new THREE.Group();
    const selectedTools = TOOLS.filter((tool) => toolIds.includes(tool.toolId));
    selectedTools.forEach((tool, index) => {
      const model = createToolModel(tool.model);
      model.position.set(2.2, 0.35 + index * 1.35, 0.5);
      model.rotation.z = index % 2 ? -0.2 : 0.2;
      model.userData.toolId = tool.toolId;
      toolStand.add(model);
    });
    scene.add(toolStand);

    const frameGroup = new THREE.Group();
    frameGroup.visible = false;
    const bamboo = new THREE.MeshStandardMaterial({ color: 0x8a552d, roughness: 0.74 });
    const vertices = Array.from({ length: 6 }, (_, index) => {
      const angle = (index / 6) * Math.PI * 2;
      return new THREE.Vector3(Math.sin(angle) * FRAME_RADIUS, 0, Math.cos(angle) * FRAME_RADIUS);
    });
    for (const y of [0.65, 3.35]) {
      for (let index = 0; index < 6; index += 1) {
        const start = vertices[index];
        const end = vertices[(index + 1) % 6];
        const edge = new THREE.Mesh(new THREE.BoxGeometry(start.distanceTo(end), 0.16, 0.16), bamboo);
        edge.position.set((start.x + end.x) / 2, y, (start.z + end.z) / 2);
        edge.rotation.y = -Math.atan2(end.z - start.z, end.x - start.x);
        edge.castShadow = true;
        frameGroup.add(edge);
      }
    }
    for (let index = 0; index < 6; index += 1) {
      const angle = (index / 6) * Math.PI * 2;
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 2.7, 10), bamboo);
      post.position.set(Math.sin(angle) * FRAME_RADIUS, 2, Math.cos(angle) * FRAME_RADIUS);
      post.castShadow = true;
      frameGroup.add(post);
    }
    const capsGroup = new THREE.Group();
    capsGroup.visible = false;
    const capMaterial = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(selectedMaterial?.surfaceColor ?? "#ead8b8"),
      roughness: 0.82,
      transparent: true,
      opacity: 0.94,
      side: THREE.DoubleSide,
    });
    const bottomCap = new THREE.Mesh(new THREE.CircleGeometry(FRAME_RADIUS - 0.05, 6), capMaterial);
    bottomCap.rotation.x = -Math.PI / 2;
    bottomCap.rotation.z = Math.PI / 6;
    bottomCap.position.y = 0.64;
    const topCap = new THREE.Mesh(new THREE.RingGeometry(0.2, FRAME_RADIUS - 0.05, 6), capMaterial.clone());
    topCap.rotation.x = -Math.PI / 2;
    topCap.rotation.z = Math.PI / 6;
    topCap.position.y = 3.36;
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.58, 0.045, 10, 40, Math.PI), bamboo);
    handle.position.y = 3.38;
    capsGroup.add(bottomCap, topCap, handle);
    frameGroup.add(capsGroup);
    scene.add(frameGroup);

    const panels: THREE.Mesh[] = [];
    const occupiedSlots = new Map<number, THREE.Mesh>();
    for (let index = 0; index < 6; index += 1) {
      const panel = new THREE.Mesh(new THREE.PlaneGeometry(PANEL_WIDTH, PANEL_HEIGHT), paperMaterial.clone());
      panel.position.set(-3 + index * 1.2, 0.6, 1.7);
      panel.scale.setScalar(0.72);
      panel.visible = false;
      panel.castShadow = true;
      panel.userData.kind = "assembly-panel";
      panel.userData.index = index;
      panel.userData.home = panel.position.clone();
      panel.userData.slot = null;
      panels.push(panel);
      frameGroup.add(panel);
    }
    for (let index = 0; index < Math.min(initialProgress.assembledPanels, 6); index += 1) {
      const panel = panels[index];
      placePanelOnLantern(panel, index);
      panel.userData.slot = index;
      occupiedSlots.set(index, panel);
    }

    const updatePhaseScene = (): void => {
      const current = phaseRef.current;
      designGroup.visible = current === "design" || current === "pierce";
      toolStand.visible = current === "design" || current === "pierce";
      frameGroup.visible = current === "assemble" || current === "complete";
      capsGroup.visible = current === "complete" || (current === "assemble" && occupiedSlots.size === 6);
      panels.forEach((panel, index) => {
        panel.visible = current === "assemble" || current === "complete";
        if (current === "complete" && panel.userData.slot === null) {
          placePanelOnLantern(panel, index);
          panel.userData.slot = index;
        }
      });
      const litAt = progressRef.current.completedAt ? Date.parse(progressRef.current.completedAt) : 0;
      const lightProgress = current === "complete"
        ? THREE.MathUtils.clamp((Date.now() - litAt) / 2200, 0, 1)
        : 0;
      const lightEase = 1 - Math.pow(1 - lightProgress, 3);
      lanternLight.intensity = current === "pierce" ? 2.4 : current === "complete" ? 0.35 + lightEase * 7.4 : 0;
      glowMaterial.opacity = current === "complete" ? lightEase * 0.48 : 0;
      moteMaterial.opacity = current === "complete" ? lightEase * 0.82 : 0;
      motes.visible = current === "complete";
      panels.forEach((panel) => {
        const material = panel.material as THREE.MeshPhysicalMaterial;
        material.emissiveIntensity = current === "complete" ? 0.08 + lightEase * 0.72 : 0.08;
      });
      renderer.toneMappingExposure = current === "complete" ? 1.15 + lightEase * 0.2 : 1.15;
      controls.target.set(0, current === "assemble" || current === "complete" ? 1.9 : 2.05, 0);
    };
    updatePhaseScene();

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const dragPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -1.7);
    let drawing = false;
    let strokeId = 0;
    let strokeStartCount = 0;
    let draggedPanel: THREE.Mesh | null = null;
    let dragOffset = new THREE.Vector3();

    const setPointer = (event: PointerEvent): void => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
    };

    const playTone = (frequency: number, duration = 0.06): void => {
      try {
        const AudioContextClass = globalThis.AudioContext;
        if (!AudioContextClass) return;
        const audio = new AudioContextClass();
        const oscillator = audio.createOscillator();
        const gain = audio.createGain();
        oscillator.frequency.value = frequency;
        gain.gain.setValueAtTime(0.035, audio.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + duration);
        oscillator.connect(gain).connect(audio.destination);
        oscillator.start();
        oscillator.stop(audio.currentTime + duration);
        oscillator.addEventListener("ended", () => void audio.close(), { once: true });
      } catch {
        // Audio feedback is optional; visual interaction remains available.
      }
    };

    const addDrawMark = (intersection: THREE.Intersection): void => {
      if (!intersection.uv) return;
      const marks = progressRef.current.marks;
      const nextMark: CraftMark = {
        x: THREE.MathUtils.clamp(intersection.uv.x, 0, 1),
        y: THREE.MathUtils.clamp(1 - intersection.uv.y, 0, 1),
        kind: "draw",
        size: drawingTool?.model === "stencil" ? 7 : 5,
        strokeId,
      };
      const previous = marks.at(-1);
      if (
        previous?.strokeId === strokeId &&
        Math.hypot(previous.x - nextMark.x, previous.y - nextMark.y) < 0.006
      ) return;
      marks.push(nextMark);
      if (marks.length > 1500) marks.splice(0, marks.length - 1500);
      redrawTexture();
      paperTexture.needsUpdate = true;
    };

    const addHole = (intersection: THREE.Intersection): void => {
      if (!intersection.uv) return;
      const size = piercingTool?.model === "broad_awl" ? 9 : 5;
      const next = cloneProgress(progressRef.current);
      next.marks.push({
        x: THREE.MathUtils.clamp(intersection.uv.x, 0, 1),
        y: THREE.MathUtils.clamp(1 - intersection.uv.y, 0, 1),
        kind: "hole",
        size,
        strokeId: Date.now(),
      });
      next.holes += 1;
      progressRef.current = next;
      setHoles(next.holes);
      redrawTexture();
      paperTexture.needsUpdate = true;
      onProgress(cloneProgress(next), "pierce");
      playTone(520 + Math.min(next.holes, 12) * 12);
      setMessage(next.holes >= 8 ? "透光孔已达到装配要求，可以进入装片。" : `已完成 ${next.holes}/8 个透光孔。`);
    };

    const handlePointerDown = (event: PointerEvent): void => {
      if (readOnly) return;
      setPointer(event);
      const currentPhase = phaseRef.current;
      if (currentPhase === "design" || currentPhase === "pierce") {
        const hit = raycaster.intersectObject(designPanel, false)[0];
        if (!hit) return;
        controls.enabled = false;
        renderer.domElement.setPointerCapture(event.pointerId);
        if (currentPhase === "pierce") {
          addHole(hit);
          return;
        }
        drawing = true;
        strokeId = Date.now();
        strokeStartCount = progressRef.current.marks.length;
        addDrawMark(hit);
        playTone(250, 0.045);
        return;
      }
      if (currentPhase === "assemble") {
        const hit = raycaster.intersectObjects(panels.filter((panel) => panel.visible), false)[0];
        if (!hit) return;
        draggedPanel = hit.object as THREE.Mesh;
        const oldSlot = draggedPanel.userData.slot as number | null;
        if (oldSlot !== null) {
          occupiedSlots.delete(oldSlot);
          draggedPanel.userData.slot = null;
        }
        draggedPanel.rotation.set(0, 0, 0);
        draggedPanel.scale.setScalar(0.82);
        draggedPanel.position.z = 1.7;
        const point = new THREE.Vector3();
        raycaster.ray.intersectPlane(dragPlane, point);
        dragOffset = draggedPanel.position.clone().sub(point);
        controls.enabled = false;
        renderer.domElement.setPointerCapture(event.pointerId);
        setMessage("把灯片拖到灯架中央的亮区，松手后吸附。 ");
      }
    };

    const handlePointerMove = (event: PointerEvent): void => {
      setPointer(event);
      if (drawing && phaseRef.current === "design") {
        const hit = raycaster.intersectObject(designPanel, false)[0];
        if (hit) addDrawMark(hit);
      }
      if (draggedPanel) {
        const point = new THREE.Vector3();
        if (raycaster.ray.intersectPlane(dragPlane, point)) {
          draggedPanel.position.copy(point.add(dragOffset));
          draggedPanel.position.x = THREE.MathUtils.clamp(draggedPanel.position.x, -3.8, 3.8);
          draggedPanel.position.y = THREE.MathUtils.clamp(draggedPanel.position.y, 0.3, 3.8);
        }
      }
    };

    const handlePointerUp = (event: PointerEvent): void => {
      if (drawing) {
        drawing = false;
        const added = progressRef.current.marks.length - strokeStartCount;
        if (added >= 2) {
          const next = cloneProgress(progressRef.current);
          next.strokes += 1;
          progressRef.current = next;
          setStrokes(next.strokes);
          onProgress(cloneProgress(next), "draw");
          setMessage(next.strokes >= requiredStrokes ? "纹样已经可以进入针刺；你也可以继续补画。" : `再画 ${requiredStrokes - next.strokes} 笔即可进入针刺。`);
        }
      }
      if (draggedPanel) {
        const panel = draggedPanel;
        const snapRadius = assemblyTool?.model === "clip" ? 2.05 : 1.55;
        const nearFrame = Math.hypot(panel.position.x, panel.position.y - 2) <= snapRadius;
        if (nearFrame) {
          const freeSlot = Array.from({ length: 6 }, (_, index) => index).find(
            (index) => !occupiedSlots.has(index),
          );
          if (freeSlot !== undefined) {
            placePanelOnLantern(panel, freeSlot);
            panel.userData.slot = freeSlot;
            occupiedSlots.set(freeSlot, panel);
            const next = cloneProgress(progressRef.current);
            next.assembledPanels = occupiedSlots.size;
            progressRef.current = next;
            setAssembledPanels(next.assembledPanels);
            onProgress(cloneProgress(next), "assemble");
            playTone(320 + freeSlot * 55, 0.12);
            setMessage(`灯片已吸附 ${next.assembledPanels}/6。${next.assembledPanels < 6 ? "继续装配。" : "灯体已经闭合。现在由你亲手点灯。"}`);
            if (next.assembledPanels === 6) playTone(520, 0.22);
          }
        } else {
          panel.position.copy(panel.userData.home as THREE.Vector3);
          panel.rotation.set(0, 0, 0);
          panel.scale.setScalar(0.72);
          const next = cloneProgress(progressRef.current);
          next.assembledPanels = occupiedSlots.size;
          progressRef.current = next;
          setAssembledPanels(next.assembledPanels);
          setMessage("灯片没有靠近灯架，已放回材料托盘。 ");
        }
        draggedPanel = null;
      }
      controls.enabled = true;
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
    };

    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerup", handlePointerUp);
    renderer.domElement.addEventListener("pointercancel", handlePointerUp);

    const resize = (): void => {
      const width = Math.max(320, mount.clientWidth);
      const height = Math.max(480, mount.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    resize();

    renderer.setAnimationLoop(() => {
      updatePhaseScene();
      if (phaseRef.current === "complete") {
        frameGroup.rotation.y += 0.0025;
        motes.rotation.y -= 0.0018;
        motes.position.y = Math.sin(Date.now() / 700) * 0.08;
      }
      controls.update();
      renderer.render(scene, camera);
    });

    return () => {
      resizeObserver.disconnect();
      renderer.setAnimationLoop(null);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("pointerup", handlePointerUp);
      renderer.domElement.removeEventListener("pointercancel", handlePointerUp);
      controls.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      paperTexture.dispose();
      glowTexture.dispose();
      glowMaterial.dispose();
      moteGeometry.dispose();
      moteMaterial.dispose();
      if (generatedPatternImage) generatedPatternImage.onload = null;
      renderer.dispose();
      vrButton?.remove();
      renderer.domElement.remove();
    };
    // Craft state is synchronized at checkpoints. Recreating the WebGL scene for
    // every mark would interrupt direct manipulation, so initialization inputs are frozen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generatedPatternUrl, materialId, patternId, toolIds.join("|"), readOnly]);

  const phaseIcons: Record<CraftProgress["phase"], IconName> = {
    design: "brush",
    pierce: "sparkles",
    assemble: "clip",
    complete: "lantern",
  };

  return (
    <section className={`workshop-shell ${displayOnly ? "is-display-only" : ""}`} aria-label={displayOnly ? "三维灯彩作品预览" : "三维灯彩制作工作台"}>
      {!displayOnly && (
      <div className="workshop-topbar">
        <div>
          <span className="eyebrow">3D 直接制作</span>
          <h2>{PHASE_LABELS[phase]}</h2>
        </div>
        <div className="craft-meter-row">
          <span className={strokes >= requiredStrokes ? "is-done" : ""} title="笔画"><Icon name="brush" />{strokes}/{requiredStrokes}</span>
          <span className={holes >= 8 ? "is-done" : ""} title="透光孔"><Icon name="sparkles" />{holes}/8</span>
          <span className={assembledPanels >= 6 ? "is-done" : ""} title="装片"><Icon name="clip" />{assembledPanels}/6</span>
        </div>
      </div>
      )}
      <div className="workshop-viewport" ref={mountRef}>
        <div className="workshop-help">
          <strong>{displayOnly ? "拖动旋转 · 滚轮缩放" : message}</strong>
        </div>
        {generatedPatternUrl && (
          <div className="ai-surface-badge" title="AI 生成的插画已成为全部六片灯面的底图"><Icon name="sparkles" /><span>AI 插画</span><b>6</b><Icon name="lantern" /></div>
        )}
        <div className={`lantern-ritual ${igniting ? "is-active" : ""}`} aria-hidden="true">
          {Array.from({ length: 12 }, (_, index) => <i key={index} style={{ "--ritual-index": index } as CSSProperties} />)}
        </div>
        {!displayOnly && (
        <div className="workshop-phase-track" aria-label="制作阶段">
          {(["design", "pierce", "assemble", "complete"] as const).map((item, index) => (
            <span className={item === phase ? "is-current" : index < ["design", "pierce", "assemble", "complete"].indexOf(phase) ? "is-done" : ""} key={item} title={PHASE_LABELS[item]}><Icon name={phaseIcons[item]} /><i>{index + 1}</i></span>
          ))}
        </div>
        )}
      </div>
      {!displayOnly && (
      <div className="workshop-actions">
        <span className="simulation-info" title="纸面性能、孔径和吸附范围为游戏模拟参数"><Icon name="info" /></span>
        {phase === "design" && (
          <button className="button button-primary icon-action" disabled={readOnly || strokes < requiredStrokes} onClick={() => changePhase("pierce")} type="button"><Icon name="sparkles" /><span>针刺</span></button>
        )}
        {phase === "pierce" && (
          <button className="button button-primary icon-action" disabled={readOnly || holes < 8} onClick={() => changePhase("assemble")} type="button"><Icon name="clip" /><span>装片</span></button>
        )}
        {phase === "assemble" && assembledPanels >= 6 && (
          <button aria-label="亲手点亮灯笼" className="button ignition-button" disabled={readOnly} onClick={igniteLantern} type="button"><Icon name="sun" /><span>点灯</span></button>
        )}
        {phase === "complete" && <span className="craft-complete-badge"><Icon name="lantern" />完成</span>}
      </div>
      )}
    </section>
  );
}
