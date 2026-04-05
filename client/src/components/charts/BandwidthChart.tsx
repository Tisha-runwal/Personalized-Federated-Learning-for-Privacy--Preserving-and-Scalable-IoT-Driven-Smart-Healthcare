import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { RoundUpdate } from '../../types/metrics';

interface BandwidthChartProps {
  updates: RoundUpdate[];
  maxRounds?: number;
}

interface DataPoint {
  round: number;
  original: number;
  quantized: number;
}

function formatYAxis(bytes: number): string {
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)}M`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)}K`;
  return `${bytes}`;
}

export function BandwidthChart({ updates, maxRounds = 20 }: BandwidthChartProps) {
  const data: DataPoint[] = updates
    .filter((u) => u.metrics.bytes_original !== undefined || u.metrics.bytes_quantized !== undefined)
    .slice(-maxRounds)
    .map((u) => ({
      round: u.round,
      original: u.metrics.bytes_original ?? 0,
      quantized: u.metrics.bytes_quantized ?? 0,
    }));

  if (data.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 flex flex-col items-center justify-center h-64">
        <p className="text-slate-500 text-sm">No bandwidth data yet</p>
        <p className="text-slate-600 text-xs mt-1">Start training to see communication stats</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
      <h3 className="text-sm font-medium text-slate-300 mb-4">Communication Overhead per Round</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
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
            tickFormatter={formatYAxis}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
            labelStyle={{ color: '#94a3b8', fontSize: 11 }}
            formatter={(value) => [formatYAxis(Number(value)) + ' bytes', '']}
          />
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '16px' }} />
          <Bar dataKey="original" name="Original" fill="#6b7280" radius={[2, 2, 0, 0]} />
          <Bar dataKey="quantized" name="Quantized" fill="#3b82f6" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
