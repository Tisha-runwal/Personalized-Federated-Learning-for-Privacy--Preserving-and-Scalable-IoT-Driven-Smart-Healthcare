interface PrivacyGaugeProps {
  epsilonSpent: number;
  epsilonTarget?: number;
}

export function PrivacyGauge({ epsilonSpent, epsilonTarget = 10 }: PrivacyGaugeProps) {
  const ratio = Math.min(epsilonSpent / epsilonTarget, 1);
  const pct = ratio * 100;

  // Color transitions: green → yellow → red
  const getColor = (r: number) => {
    if (r < 0.5) {
      // Green to yellow
      const t = r / 0.5;
      const g = Math.round(160 + (255 - 160) * t);
      const r2 = Math.round(74 + (234 - 74) * t);
      return `rgb(${r2}, ${g}, 72)`;
    } else {
      // Yellow to red
      const t = (r - 0.5) / 0.5;
      const g = Math.round(255 * (1 - t));
      return `rgb(239, ${g}, 68)`;
    }
  };

  const color = getColor(ratio);

  // SVG arc parameters
  const cx = 100;
  const cy = 90;
  const r = 70;
  const startAngle = -Math.PI * 0.8;
  const endAngle = Math.PI * 0.8;
  const totalAngle = endAngle - startAngle;
  const currentAngle = startAngle + totalAngle * ratio;

  const polarToCart = (angle: number, radius: number) => ({
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  });

  const start = polarToCart(startAngle, r);
  const end = polarToCart(endAngle, r);
  const current = polarToCart(currentAngle, r);

  const arcPath = (from: typeof start, to: typeof end, largeArc: boolean) => {
    const la = largeArc ? 1 : 0;
    return `M ${from.x} ${from.y} A ${r} ${r} 0 ${la} 1 ${to.x} ${to.y}`;
  };

  const bgPath = arcPath(start, end, true);
  const fgPath = ratio > 0
    ? arcPath(start, current, totalAngle * ratio > Math.PI)
    : '';

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 flex flex-col items-center">
      <h3 className="text-sm font-medium text-slate-300 mb-4 self-start">Privacy Budget</h3>

      <svg width="200" height="140" viewBox="0 0 200 140">
        {/* Background arc */}
        <path
          d={bgPath}
          fill="none"
          stroke="#334155"
          strokeWidth="14"
          strokeLinecap="round"
        />

        {/* Foreground arc (spent) */}
        {fgPath && (
          <path
            d={fgPath}
            fill="none"
            stroke={color}
            strokeWidth="14"
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${color}60)` }}
          />
        )}

        {/* Center text */}
        <text x={cx} y={cy - 10} textAnchor="middle" fill="white" fontSize="20" fontWeight="bold">
          ε={epsilonSpent.toFixed(2)}
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle" fill="#94a3b8" fontSize="11">
          of {epsilonTarget} budget
        </text>
        <text x={cx} y={cy + 28} textAnchor="middle" fill={color} fontSize="13" fontWeight="600">
          {pct.toFixed(0)}% used
        </text>

        {/* Min/max labels */}
        <text x={start.x - 8} y={start.y + 4} textAnchor="end" fill="#64748b" fontSize="10">0</text>
        <text x={end.x + 8} y={end.y + 4} textAnchor="start" fill="#64748b" fontSize="10">{epsilonTarget}</text>
      </svg>

      {/* Status indicator */}
      <div
        className="mt-2 text-xs font-medium px-3 py-1 rounded-full"
        style={{ backgroundColor: `${color}20`, color }}
      >
        {pct < 50 ? 'Low privacy expenditure' : pct < 80 ? 'Moderate expenditure' : 'High privacy expenditure'}
      </div>
    </div>
  );
}
