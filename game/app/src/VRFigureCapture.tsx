import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRButton } from "three/addons/webxr/VRButton.js";

type Stage = "overview" | "materials" | "assembly" | "pattern" | "process" | "light" | "kg";

const STAGES: Stage[] = ["overview", "materials", "assembly", "pattern", "process", "light", "kg"];
const STAGE_LABELS: Record<Stage, string> = {
  overview: "VR workshop",
  materials: "Material and tool selection",
  assembly: "Hand-scale 3D assembly",
  pattern: "AI pattern mapped to the lantern",
  process: "Craft-process reel",
  light: "Lantern-lighting ritual",
  kg: "KG evidence and state guidance",
};

function roundedRect(context: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number): void {
  context.beginPath();
  context.roundRect(x, y, width, height, radius);
}

function textTexture(title: string, subtitle: string, accent = "#C95E72"): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 1400;
  canvas.height = 330;
  const context = canvas.getContext("2d")!;
  context.clearRect(0, 0, canvas.width, canvas.height);
  roundedRect(context, 12, 12, canvas.width - 24, canvas.height - 24, 38);
  context.fillStyle = "rgba(8, 26, 38, 0.90)";
  context.fill();
  context.strokeStyle = "rgba(255,255,255,0.18)";
  context.lineWidth = 4;
  context.stroke();
  context.fillStyle = accent;
  context.fillRect(58, 58, 12, 210);
  context.fillStyle = "#FBF5E8";
  context.font = "700 62px Arial, sans-serif";
  context.fillText(title, 105, 145);
  context.fillStyle = "rgba(251,245,232,0.72)";
  context.font = "400 34px Arial, sans-serif";
  context.fillText(subtitle, 105, 215);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function patternTexture(patterned: boolean): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 900;
  const context = canvas.getContext("2d")!;
  context.fillStyle = patterned ? "#EADDB8" : "#F3E5BE";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.globalAlpha = 0.15;
  context.strokeStyle = "#7D6040";
  context.lineWidth = 2;
  for (let y = 20; y < canvas.height; y += 28) {
    context.beginPath();
    context.moveTo(0, y);
    context.bezierCurveTo(180, y + 8, 430, y - 8, canvas.width, y + 3);
    context.stroke();
  }
  context.globalAlpha = 1;
  context.strokeStyle = patterned ? "#9F493E" : "#80574A";
  context.fillStyle = patterned ? "#C9A45C" : "#8C6658";
  context.lineWidth = 13;
  context.lineCap = "round";
  if (patterned) {
    context.beginPath();
    context.moveTo(110, 790);
    context.bezierCurveTo(210, 600, 310, 430, 500, 120);
    context.stroke();
    for (let index = 0; index < 7; index += 1) {
      const x = 180 + index * 45;
      const y = 650 - index * 74;
      context.save();
      context.translate(x, y);
      context.rotate(index % 2 ? -0.55 : 0.55);
      context.beginPath();
      context.ellipse(0, 0, 55, 24, 0, 0, Math.PI * 2);
      context.fill();
      context.restore();
    }
    for (const [x, y] of [[150, 290], [310, 250], [420, 510], [240, 710]]) {
      context.beginPath();
      context.arc(x, y, 38, 0, Math.PI * 2);
      context.stroke();
      context.beginPath();
      context.arc(x, y, 13, 0, Math.PI * 2);
      context.fill();
    }
  } else {
    for (let y = 170; y < 790; y += 190) {
      context.beginPath();
      context.moveTo(110, y);
      context.bezierCurveTo(230, y - 45, 365, y + 45, 525, y);
      context.stroke();
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  return texture;
}

function createLantern(patterned: boolean, lit: boolean, exploded: boolean): THREE.Group {
  const group = new THREE.Group();
  const radius = 1.05;
  const panelWidth = radius + 0.03;
  const panelHeight = 2.45;
  const bamboo = new THREE.MeshStandardMaterial({ color: 0x9a5a2c, roughness: 0.66 });
  const paper = new THREE.MeshPhysicalMaterial({
    map: patternTexture(patterned),
    color: 0xffffff,
    roughness: 0.7,
    transparent: true,
    opacity: 0.96,
    transmission: lit ? 0.18 : 0.03,
    emissive: new THREE.Color(lit ? 0xffb84e : 0x6d421f),
    emissiveIntensity: lit ? 1.05 : 0.07,
    side: THREE.DoubleSide,
  });
  for (const y of [0.4, 2.95]) {
    for (let index = 0; index < 6; index += 1) {
      const a0 = (index / 6) * Math.PI * 2;
      const a1 = ((index + 1) / 6) * Math.PI * 2;
      const x0 = Math.sin(a0) * radius;
      const z0 = Math.cos(a0) * radius;
      const x1 = Math.sin(a1) * radius;
      const z1 = Math.cos(a1) * radius;
      const edge = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(x1 - x0, z1 - z0), 0.13, 0.13), bamboo);
      edge.position.set((x0 + x1) / 2, y, (z0 + z1) / 2);
      edge.rotation.y = -Math.atan2(z1 - z0, x1 - x0);
      edge.castShadow = true;
      group.add(edge);
    }
  }
  for (let index = 0; index < 6; index += 1) {
    const angle = (index / 6) * Math.PI * 2;
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 2.55, 12), bamboo);
    post.position.set(Math.sin(angle) * radius, 1.68, Math.cos(angle) * radius);
    post.castShadow = true;
    group.add(post);
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(panelWidth, panelHeight), paper.clone());
    const faceAngle = ((index + 0.5) / 6) * Math.PI * 2;
    const panelRadius = radius * Math.cos(Math.PI / 6);
    const spread = exploded ? 0.72 : 0;
    panel.position.set(
      Math.sin(faceAngle) * (panelRadius + spread),
      1.68 + (exploded ? Math.sin(index * 1.7) * 0.18 : 0),
      Math.cos(faceAngle) * (panelRadius + spread),
    );
    panel.rotation.y = faceAngle;
    panel.castShadow = true;
    group.add(panel);
  }
  const handle = new THREE.Mesh(new THREE.TorusGeometry(0.48, 0.045, 10, 44, Math.PI), bamboo);
  handle.position.y = 3.0;
  group.add(handle);
  const top = new THREE.Mesh(new THREE.RingGeometry(0.16, radius - 0.05, 6), paper.clone());
  top.rotation.x = -Math.PI / 2;
  top.rotation.z = Math.PI / 6;
  top.position.y = 3.02;
  group.add(top);
  if (lit) {
    const light = new THREE.PointLight(0xffbd55, 9.5, 9, 1.6);
    light.position.set(0, 1.7, 0);
    group.add(light);
  }
  return group;
}

function applyAiIllustration(lantern: THREE.Group): void {
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 720;
    canvas.height = 720;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#F0E2BD";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 118, 1295, 226, 180, 0, 0, canvas.width, canvas.height);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    lantern.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      const material = object.material;
      const materials = Array.isArray(material) ? material : [material];
      materials.forEach((candidate) => {
        if (!(candidate instanceof THREE.MeshPhysicalMaterial) || !candidate.map) return;
        candidate.map = texture;
        candidate.color.set(0xffffff);
        candidate.needsUpdate = true;
      });
    });
  };
  image.src = "/vr-figure/ai-pattern-source.png";
}

function createController(color: number): THREE.Group {
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.25 });
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.12, 0.45, 8, 18), bodyMaterial);
  body.rotation.x = Math.PI / 2.8;
  group.add(body);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.028, 8, 30), bodyMaterial);
  ring.rotation.x = Math.PI / 2;
  ring.position.set(0, 0.12, -0.26);
  group.add(ring);
  const rayGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.12, -0.2), new THREE.Vector3(0, 0.12, -4.8)]);
  const ray = new THREE.Line(rayGeometry, new THREE.LineBasicMaterial({ color: 0xffdda0, transparent: true, opacity: 0.68 }));
  group.add(ray);
  return group;
}

function addImagePanel(scene: THREE.Scene, stage: Stage): void {
  const texture = new THREE.TextureLoader().load(`/vr-figure/${stage}.png`);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  const frame = new THREE.Mesh(
    new THREE.BoxGeometry(6.35, 3.62, 0.12),
    new THREE.MeshStandardMaterial({ color: 0x8f5d35, roughness: 0.65, metalness: 0.08 }),
  );
  frame.position.set(-0.45, 3.0, -1.1);
  frame.rotation.y = 0.045;
  scene.add(frame);
  const screen = new THREE.Mesh(
    new THREE.PlaneGeometry(6.16, 3.43),
    new THREE.MeshBasicMaterial({ map: texture, toneMapped: false }),
  );
  screen.position.set(-0.45, 3.0, -1.028);
  screen.rotation.y = 0.045;
  scene.add(screen);
}

function addMaterialObjects(scene: THREE.Scene): void {
  const colors = [0xead8b8, 0xcdbb86, 0x6f8ea0];
  colors.forEach((color, index) => {
    const roll = new THREE.Mesh(
      new THREE.CylinderGeometry(0.24, 0.24, 1.05, 30),
      new THREE.MeshPhysicalMaterial({ color, roughness: 0.76 }),
    );
    roll.rotation.z = Math.PI / 2;
    roll.position.set(2.0 + index * 0.38, 0.38 + index * 0.1, 0.5 - index * 0.22);
    roll.rotation.y = index * 0.38;
    scene.add(roll);
  });
  const awl = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.08, 1.2, 18), new THREE.MeshStandardMaterial({ color: 0x9a5a2c }));
  awl.rotation.z = Math.PI / 2;
  awl.position.set(2.55, 0.5, 1.15);
  scene.add(awl);
}

function addKgObjects(scene: THREE.Scene): void {
  const nodes: THREE.Mesh[] = [];
  const positions = [[2.2, 3.2, 0.2], [2.9, 2.6, 0], [2.15, 2.1, 0.3], [3.25, 1.7, 0.1], [2.45, 1.25, 0.4]];
  positions.forEach((position, index) => {
    const node = new THREE.Mesh(
      new THREE.SphereGeometry(index === 0 ? 0.2 : 0.14, 24, 16),
      new THREE.MeshStandardMaterial({ color: [0xc95e72, 0x6576a5, 0xb78336, 0x5f8f82, 0x9b93b9][index], emissive: 0x132334, emissiveIntensity: 0.25 }),
    );
    node.position.set(position[0], position[1], position[2]);
    nodes.push(node);
    scene.add(node);
  });
  for (let index = 0; index < nodes.length - 1; index += 1) {
    const geometry = new THREE.BufferGeometry().setFromPoints([nodes[index].position, nodes[index + 1].position]);
    scene.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xe8ebef, transparent: true, opacity: 0.75 })));
  }
}

export default function VRFigureCapture() {
  const mountRef = useRef<HTMLDivElement>(null);
  const query = new URLSearchParams(globalThis.location.search);
  const requested = query.get("stage") as Stage | null;
  const stage: Stage = requested && STAGES.includes(requested) ? requested : "overview";
  const paperMode = query.get("paper") === "1";

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(stage === "light" ? 0x07131e : 0x0c2633);
    scene.fog = new THREE.Fog(stage === "light" ? 0x07131e : 0x0c2633, 9, 20);
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 80);
    camera.position.set(0, 2.7, 8.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = stage === "light" ? 1.25 : 1.08;
    renderer.shadowMap.enabled = true;
    renderer.xr.enabled = true;
    renderer.domElement.className = "vr-figure-canvas";
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 2.15, 0);
    controls.minDistance = 6.8;
    controls.maxDistance = 11;

    scene.add(new THREE.HemisphereLight(0xffefd4, 0x183949, stage === "light" ? 1.15 : 2.15));
    const key = new THREE.DirectionalLight(0xffd7a0, stage === "light" ? 1.5 : 3.1);
    key.position.set(-4, 8, 6);
    key.castShadow = true;
    scene.add(key);
    const rim = new THREE.PointLight(0x8bb9cf, 2.4, 14);
    rim.position.set(4.2, 4.5, 3.8);
    scene.add(rim);

    const roomTexture = new THREE.TextureLoader().load(stage === "light" ? "/assets/bg-festival-night.png" : "/assets/bg-crafting-table.png");
    roomTexture.colorSpace = THREE.SRGBColorSpace;
    const room = new THREE.Mesh(
      new THREE.PlaneGeometry(15.5, 8.7),
      new THREE.MeshBasicMaterial({
        map: roomTexture,
        color: stage === "light" ? 0x786b72 : 0x71828a,
        transparent: true,
        opacity: stage === "light" ? 0.56 : 0.42,
        toneMapped: false,
      }),
    );
    room.position.set(0, 3.55, -4.1);
    scene.add(room);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(18, 14),
      new THREE.MeshStandardMaterial({ color: 0x39251d, roughness: 0.9 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.05;
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(16, 32, 0x8e6544, 0x5c4030);
    grid.position.y = -0.035;
    scene.add(grid);

    const table = new THREE.Mesh(new THREE.BoxGeometry(10, 0.42, 4.7), new THREE.MeshStandardMaterial({ color: 0x5a3826, roughness: 0.82 }));
    table.position.set(0, 0.02, 0.55);
    table.receiveShadow = true;
    scene.add(table);

    if (!paperMode) {
      const label = new THREE.Mesh(
        new THREE.PlaneGeometry(5.55, 1.31),
        new THREE.MeshBasicMaterial({ map: textTexture(STAGE_LABELS[stage], "LanternQuest · WebXR interaction view"), transparent: true, toneMapped: false }),
      );
      label.position.set(-1.28, 4.92, -0.7);
      label.rotation.y = 0.035;
      scene.add(label);
    }

    if (stage !== "overview" && stage !== "light") addImagePanel(scene, stage);
    if (stage === "overview") {
      const lantern = createLantern(true, true, false);
      lantern.position.set(0, 0.15, -0.2);
      lantern.rotation.y = 0.35;
      lantern.scale.setScalar(1.18);
      scene.add(lantern);
      addKgObjects(scene);
      addMaterialObjects(scene);
    } else if (stage === "materials") {
      addMaterialObjects(scene);
    } else if (stage === "assembly") {
      const lantern = createLantern(false, false, true);
      lantern.position.set(2.55, 0.18, 0.15);
      lantern.rotation.y = -0.4;
      lantern.scale.setScalar(0.82);
      scene.add(lantern);
    } else if (stage === "pattern") {
      const lantern = createLantern(true, false, false);
      applyAiIllustration(lantern);
      lantern.position.set(2.65, 0.14, 0.16);
      lantern.rotation.y = -0.3;
      lantern.scale.setScalar(0.82);
      scene.add(lantern);
    } else if (stage === "process") {
      const platePaths = ["/assets/process-plate-01-plan-v1.webp", "/assets/process-plate-02-pierce-v1.webp", "/assets/process-plate-03-light-v1.webp"];
      platePaths.forEach((path, index) => {
        const texture = new THREE.TextureLoader().load(path);
        texture.colorSpace = THREE.SRGBColorSpace;
        const plate = new THREE.Mesh(new THREE.PlaneGeometry(1.15, 0.8), new THREE.MeshBasicMaterial({ map: texture, toneMapped: false }));
        plate.position.set(1.72 + index * 0.72, 0.85 + index * 0.28, 1.0 - index * 0.18);
        plate.rotation.set(-0.75, -0.25 + index * 0.16, 0.03 * (index - 1));
        scene.add(plate);
      });
    } else if (stage === "light") {
      const lantern = createLantern(false, true, false);
      lantern.position.set(0, 0.13, -0.25);
      lantern.rotation.y = 0.4;
      lantern.scale.setScalar(1.28);
      scene.add(lantern);
      for (let index = 0; index < 46; index += 1) {
        const mote = new THREE.Mesh(new THREE.SphereGeometry(0.018 + (index % 3) * 0.006, 10, 7), new THREE.MeshBasicMaterial({ color: 0xffd27c, transparent: true, opacity: 0.72 }));
        const angle = index * 1.93;
        const radius = 1.4 + (index % 11) * 0.12;
        mote.position.set(Math.sin(angle) * radius, 0.4 + (index % 17) * 0.24, Math.cos(angle) * radius);
        scene.add(mote);
      }
    } else if (stage === "kg") {
      addKgObjects(scene);
    }

    const leftController = createController(0x435b70);
    leftController.position.set(-1.25, 0.75, 4.45);
    leftController.rotation.set(-0.12, 0.16, -0.18);
    scene.add(leftController);
    const rightController = createController(0x9f493e);
    rightController.position.set(1.05, 0.68, 4.38);
    rightController.rotation.set(-0.18, -0.12, 0.2);
    scene.add(rightController);

    for (let index = 0; index < 2; index += 1) {
      const controller = renderer.xr.getController(index);
      controller.add(createController(index === 0 ? 0x435b70 : 0x9f493e));
      scene.add(controller);
    }

    let vrButton: HTMLElement | null = null;
    const xr = (navigator as Navigator & { xr?: { isSessionSupported: (mode: string) => Promise<boolean> } }).xr;
    void xr?.isSessionSupported("immersive-vr").then((supported) => {
      if (!supported || !renderer.domElement.isConnected) return;
      vrButton = VRButton.createButton(renderer);
      vrButton.classList.add("vr-figure-enter");
      mount.appendChild(vrButton);
    });

    const resize = (): void => {
      const width = Math.max(640, mount.clientWidth);
      const height = Math.max(480, mount.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();

    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    return () => {
      observer.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      vrButton?.remove();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [paperMode, stage]);

  return (
    <main className="vr-figure-shell">
      <div ref={mountRef} className="vr-figure-stage" data-stage={stage} />
    </main>
  );
}
