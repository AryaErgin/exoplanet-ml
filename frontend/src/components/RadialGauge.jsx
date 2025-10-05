import React from "react";
import { formatPct, polar } from "../lib/utils";

export default function RadialGauge({ value = 0 }) {
  const pct = Math.max(0, Math.min(0.99, Number(value) || 0));
  const angle = 270 * pct;
  const r = 60,
    cx = 70,
    cy = 70;
  const startAngle = -225;
  const endAngle = startAngle + angle;
  const start = polar(cx, cy, r, startAngle);
  const end = polar(cx, cy, r, endAngle);
  const largeArc = angle > 180 ? 1 : 0;
  const d = `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  const color = pct > 0.8 ? "#22c55e" : pct > 0.5 ? "#eab308" : "#f97316";

  return (
    <div className="inline-flex items-center gap-4">
      <svg width="160" height="110" viewBox="0 0 160 110">
        <path
          d="M 10 70 A 60 60 0 1 1 130 70"
          stroke="rgba(255,255,255,0.15)"
          strokeWidth="10"
          fill="none"
          strokeLinecap="round"
        />

        <path
          d={d}
          stroke={color}
          strokeWidth="10"
          fill="none"
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r="3" fill="#fff" />
      </svg>

      <div>
        <div className="text-3xl font-extrabold text-white">
          {formatPct(pct)}
        </div>
        <div className="text-white/60 text-sm">Planet candidate likelihood</div>
      </div>
    </div>
  );
}
