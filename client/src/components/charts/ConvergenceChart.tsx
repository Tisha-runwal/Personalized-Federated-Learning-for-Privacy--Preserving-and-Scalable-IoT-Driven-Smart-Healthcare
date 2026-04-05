import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { RoundUpdate, MethodName } from '../../types/metrics';
import { METHOD_COLORS, METHOD_LABELS } from '../../types/metrics';

interface ConvergenceChartProps {
  updates: RoundUpdate[];
}

interface DataPoint {
  round: number;
  [method: string]: number | undefined;
}

export function ConvergenceChart({ updates }: ConvergenceChartProps) {
  // Group data by round for each method
  const dataMap = new Map<number, DataPoint>();

  for (const update of updates) {
    const acc = update.metrics.global_accuracy;
    if (acc === undefined) continue;

    if (!dataMap.has(update.round)) {
      dataMap.set(update.round, { round: update.round });
    }
    const point = dataMap.get(update.round)!;
    point[update.method] = acc * 100; // Convert to percentage
  }

  const data = Array.from(dataMap.values()).sort((a, b) => a.round - b.round);

  // Get unique methods
  const methods = [...new Set(updates.map((u) => u.method))];

  if (data.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 flex flex-col items-center justify-center h-80">
        <p className="text-slate-500 text-sm">No convergence data yet</p>
        <p className="text-slate-600 text-xs mt-1">Start training to see accuracy curves</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
      <h3 className="text-sm font-medium text-slate-300 mb-4">Accuracy vs. Rounds</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="round"
            stroke="#64748b"
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            label={{ value: 'Round', position: 'insideBottomRight', offset: -5, fill: '#94a3b8', fontSize: 11 }}
          />
          <YAxis
            stroke="#64748b"
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
            labelStyle={{ color: '#94a3b8', fontSize: 11 }}
            formatter={(value) => [`${Number(value).toFixed(1)}%`, '']}
          />
          <Legend
            wrapperStyle={{ fontSize: '11px', paddingTop: '16px' }}
          />
          {methods.map((method) => {
            const color = METHOD_COLORS[method as MethodName] ?? '#64748b';
            const isPflHcare = method === 'pfl_hcare';
            return (
              <Line
                key={method}
                type="monotone"
                dataKey={method}
                name={METHOD_LABELS[method as MethodName] ?? method}
                stroke={color}
                strokeWidth={isPflHcare ? 3 : 1.5}
                strokeDasharray={isPflHcare ? undefined : '5 3'}
                dot={false}
                activeDot={{ r: 4, fill: color }}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
