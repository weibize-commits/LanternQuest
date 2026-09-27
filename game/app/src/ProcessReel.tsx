import { useEffect, useState, type CSSProperties } from "react";

import { Icon, type IconName } from "./Icon";

const PLATE_1 = "/assets/process-plate-01-plan-v1.webp";
const PLATE_2 = "/assets/process-plate-02-pierce-v1.webp";
const PLATE_3 = "/assets/process-plate-03-light-v1.webp";

const STEPS: Array<{ label: string; detail: string; icon: IconName; art: string; image: string; focus: string }> = [
  { label: "裁纸", detail: "裁出六张同等大小的灯片", icon: "paper", art: "cut", image: PLATE_1, focus: "12%" },
  { label: "结构图", detail: "把凹凸、弧线与折线转成平面结构图", icon: "brush", art: "draft", image: PLATE_1, focus: "50%" },
  { label: "复制六片", detail: "叠放固定，用针孔定位复制结构", icon: "repeat", art: "stack", image: PLATE_1, focus: "88%" },
  { label: "上色", detail: "涂底色并绘制装饰纹样", icon: "brush", art: "paint", image: PLATE_2, focus: "12%" },
  { label: "定位", detail: "核对六片图案并确定针刺位置", icon: "tool", art: "locate", image: PLATE_2, focus: "50%" },
  { label: "针刺", detail: "置于蜡板，以锥针形成透光孔", icon: "sparkles", art: "pierce", image: PLATE_2, focus: "88%" },
  { label: "裁剪折形", detail: "留接边，按结构线正折与反折", icon: "paper", art: "fold", image: PLATE_3, focus: "12%" },
  { label: "六片合拢", detail: "分段接片、首尾合拢并整形", icon: "clip", art: "join", image: PLATE_3, focus: "50%" },
  { label: "封装点灯", detail: "完成顶底与装饰，放置光源", icon: "lantern", art: "light", image: PLATE_3, focus: "88%" },
];

function ProcessScene({ art, image, focus }: { art: string; image: string; focus: string }) {
  return (
    <div className="process-illustration" style={{ "--focus-x": focus } as CSSProperties}>
      <img alt="" aria-hidden="true" src={image} />
    <svg className={`process-scene art-${art}`} viewBox="0 0 760 420" aria-hidden="true">
      <title>万眼萝灯制作流程动画：{STEPS.find((step) => step.art === art)?.label}</title>
      <defs>
        <linearGradient id="paperGlow" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#f7e9c7" />
          <stop offset="1" stopColor="#d7b77b" />
        </linearGradient>
        <radialGradient id="lampGlow">
          <stop offset="0" stopColor="#fff3a8" stopOpacity=".95" />
          <stop offset="1" stopColor="#e18438" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect className="reel-table" x="0" y="315" width="760" height="105" rx="28" />
      <circle className="reel-glow" cx="380" cy="205" r="170" fill="url(#lampGlow)" />
      <g className="reel-paper">
        <rect x="250" y="75" width="260" height="260" rx="12" fill="url(#paperGlow)" />
        <path className="reel-fiber" d="M270 120c70 16 130-18 220 5M270 185c95-20 135 20 220 0M270 255c80 18 145-14 220 6" />
      </g>
      <g className="reel-cuts">
        <path d="M315 82v245M380 82v245M445 82v245" />
        <g className="reel-scissors" transform="translate(300 62)">
          <circle cx="0" cy="0" r="15" /><circle cx="34" cy="0" r="15" />
          <path d="m10 9 58 64M24 9-34 73" />
        </g>
      </g>
      <g className="reel-draft">
        <path d="M290 270 338 120l46 150 46-150 46 150" />
        <path d="M290 198h186M290 235h186" />
        <g className="reel-pencil"><path d="m270 92 18-18 105 105-18 18Z" /><path d="m270 92-8 26 26-8Z" /></g>
      </g>
      <g className="reel-stack">
        {[0, 1, 2, 3, 4, 5].map((index) => <rect key={index} x={275 + index * 7} y={92 - index * 4} width="210" height="224" rx="9" />)}
        <path className="stack-pin" d="M290 110h180M290 155h180M290 200h180M290 245h180" />
      </g>
      <g className="reel-paint">
        <path d="M286 250c45-90 92 25 135-70 24-53 52-54 70-74" />
        <path d="M300 153c22-24 43-24 64 0-22 24-43 24-64 0ZM402 236c24-25 47-25 70 0-24 25-47 25-70 0Z" />
        <g className="reel-brush"><path d="m500 76 18-18 35 35-18 18Z"/><path d="m500 76-45 72 28 15 52-52Z"/></g>
      </g>
      <g className="reel-locate">
        <path d="M300 110h160v170H300Z" />
        {[0,1,2,3,4].flatMap((row) => [0,1,2,3,4].map((column) => <circle key={`${row}-${column}`} cx={316 + column * 32} cy={126 + row * 32} r="4" />))}
        <g className="reel-awl"><path d="M525 70 410 205"/><path d="m530 62 24 20-22 27-24-20Z"/></g>
      </g>
      <g className="reel-pierce">
        {Array.from({ length: 42 }, (_, index) => {
          const angle = index * 0.62;
          const radius = 14 + index * 2.8;
          return <circle key={index} cx={380 + Math.cos(angle) * radius} cy={205 + Math.sin(angle) * radius * .72} r="4" />;
        })}
        <g className="reel-awl"><path d="M525 70 410 205"/><path d="m530 62 24 20-22 27-24-20Z"/></g>
      </g>
      <g className="reel-fold">
        <path d="M245 285 285 100l40 185 40-185 40 185 40-185 40 185" />
        <path className="fold-arrow" d="m225 190 42 0M535 190h-42" />
      </g>
      <g className="reel-join">
        {Array.from({ length: 6 }, (_, index) => {
          const angle = index * Math.PI / 3;
          return <rect key={index} x="350" y="112" width="60" height="185" rx="7" transform={`rotate(${index * 60} 380 205) translate(0 -78)`} />;
        })}
        <circle cx="380" cy="205" r="58" />
      </g>
      <g className="reel-lantern">
        <ellipse cx="380" cy="100" rx="105" ry="35" />
        <ellipse cx="380" cy="300" rx="105" ry="35" />
        {Array.from({ length: 6 }, (_, index) => <path key={index} d={`M${290 + index * 36} 108v184`} />)}
        <path d="M335 82q45-75 90 0M380 335v52M360 387h40" />
        {Array.from({ length: 22 }, (_, index) => <circle key={index} cx={310 + (index % 6) * 28} cy={138 + Math.floor(index / 6) * 42} r="3" />)}
      </g>
    </svg>
    </div>
  );
}

export function ProcessReel({ onFinish }: { onFinish: () => void }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return;
    const timer = globalThis.setTimeout(() => {
      if (step === STEPS.length - 1) setPlaying(false);
      else setStep((current) => current + 1);
    }, 1800);
    return () => globalThis.clearTimeout(timer);
  }, [playing, step]);

  const current = STEPS[step];
  return (
    <section className="process-reel" aria-label="万眼萝灯制作流程动画">
      <header className="process-reel-head">
        <div className="process-reel-title"><Icon name="lantern" /><strong>万眼萝 · 工艺影卷</strong><small>24 道工序归并</small></div>
        <button className="icon-button" aria-label="关闭流程动画" onClick={onFinish} title="关闭" type="button"><Icon name="close" /></button>
      </header>
      <div className="process-stage" data-step={step + 1}>
        <ProcessScene art={current.art} focus={current.focus} image={current.image} />
        <div className="process-step-name"><span>{String(step + 1).padStart(2, "0")}</span><strong>{current.label}</strong></div>
      </div>
      <div className="process-timeline" aria-label="制作步骤">
        {STEPS.map((item, index) => (
          <button
            aria-label={`第 ${index + 1} 步：${item.label}`}
            aria-pressed={index === step}
            className={index === step ? "is-current" : index < step ? "is-seen" : ""}
            key={item.label}
            onClick={() => { setStep(index); setPlaying(false); }}
            title={item.detail}
            type="button"
          ><Icon name={item.icon} /><span>{item.label}</span></button>
        ))}
      </div>
      <footer className="process-controls">
        <button className="icon-button" aria-label="上一步" disabled={step === 0} onClick={() => { setStep(step - 1); setPlaying(false); }} title="上一步" type="button"><Icon name="previous" /></button>
        <button className="icon-button is-primary" aria-label={playing ? "暂停" : "播放"} onClick={() => setPlaying(!playing)} title={playing ? "暂停" : "播放"} type="button"><Icon name={playing ? "pause" : "play"} /></button>
        <button className="icon-button" aria-label="下一步" disabled={step === STEPS.length - 1} onClick={() => { setStep(step + 1); setPlaying(false); }} title="下一步" type="button"><Icon name="next" /></button>
        <details className="process-scope"><summary aria-label="查看流程依据"><Icon name="info" /></summary><p>依据 N001 制作影像、关键帧及其 24 道工序清单归并为九段动画。动画用于认识流程；尺寸、胶量与安全参数仍须由传承人逐项确认。</p></details>
        <button className="reel-finish" onClick={onFinish} type="button"><Icon name="arrow" />开始创作</button>
      </footer>
    </section>
  );
}
