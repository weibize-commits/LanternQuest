import type { SVGProps } from "react";

export type IconName =
  | "arrow"
  | "brush"
  | "check"
  | "clip"
  | "close"
  | "evidence"
  | "hand"
  | "info"
  | "lantern"
  | "leaf"
  | "lock"
  | "moon"
  | "next"
  | "paper"
  | "pause"
  | "play"
  | "previous"
  | "repeat"
  | "sparkles"
  | "sun"
  | "tool"
  | "vary";

interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
}

export function Icon({ name, ...props }: IconProps) {
  const common = {
    fill: "none",
    stroke: "currentColor",
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    strokeWidth: 1.8,
  };
  const paths: Record<IconName, React.ReactNode> = {
    arrow: <><path d="M5 12h14"/><path d="m14 7 5 5-5 5"/></>,
    brush: <><path d="m14.5 4.5 5 5L10 19H5v-5Z"/><path d="m12 7 5 5"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    clip: <><path d="M7 4v12a5 5 0 0 0 10 0V6a3 3 0 0 0-6 0v10a1 1 0 0 0 2 0V7"/></>,
    close: <><path d="m6 6 12 12"/><path d="m18 6-12 12"/></>,
    evidence: <><path d="M5 3h11l3 3v15H5Z"/><path d="M16 3v4h4"/><path d="M8 11h8M8 15h6"/></>,
    hand: <path d="M7 11V6a1.5 1.5 0 0 1 3 0v4-6a1.5 1.5 0 0 1 3 0v6-5a1.5 1.5 0 0 1 3 0v6-3a1.5 1.5 0 0 1 3 0v5c0 5-3 8-7 8h-1c-3 0-5-2-7-5l-2-3a1.7 1.7 0 0 1 3-2l2 2Z"/>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></>,
    lantern: <><path d="M8 4h8M7 7h10l1 3-1 8H7L6 10Z"/><path d="M9 21h6M9 18v3M15 18v3M10 4V2h4v2"/></>,
    leaf: <><path d="M5 20c0-8 5-14 14-15-1 9-6 14-14 15Z"/><path d="M5 20c4-5 7-8 12-11"/></>,
    lock: <><rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></>,
    moon: <path d="M20 15.5A8.5 8.5 0 0 1 8.5 4 8.5 8.5 0 1 0 20 15.5Z"/>,
    next: <><path d="m9 6 6 6-6 6"/><path d="M18 6v12"/></>,
    paper: <><path d="M6 2h9l4 4v16H6Z"/><path d="M15 2v5h5M9 12h6M9 16h6"/></>,
    pause: <><path d="M9 7v10M15 7v10"/></>,
    play: <path d="m9 6 9 6-9 6Z"/>,
    previous: <><path d="m15 6-6 6 6 6"/><path d="M6 6v12"/></>,
    repeat: <><path d="M17 2l4 4-4 4"/><path d="M3 11V9a3 3 0 0 1 3-3h15"/><path d="m7 22-4-4 4-4"/><path d="M21 13v2a3 3 0 0 1-3 3H3"/></>,
    sparkles: <><path d="m12 3 1.2 3.8L17 8l-3.8 1.2L12 13l-1.2-3.8L7 8l3.8-1.2Z"/><path d="m18 14 .8 2.2L21 17l-2.2.8L18 20l-.8-2.2L15 17l2.2-.8Z"/><path d="m5 13 .7 1.8 1.8.7-1.8.7L5 18l-.7-1.8-1.8-.7 1.8-.7Z"/></>,
    sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></>,
    tool: <><path d="M14 6a4 4 0 0 0-5 5L3 17l4 4 6-6a4 4 0 0 0 5-5l-3 3-4-4Z"/></>,
    vary: <><path d="M4 7h5l6 10h5"/><path d="m17 14 3 3-3 3"/><path d="M4 17h5l2-3M15 7h5"/><path d="m17 4 3 3-3 3"/></>,
  };
  return <svg aria-hidden="true" viewBox="0 0 24 24" {...common} {...props}>{paths[name]}</svg>;
}
