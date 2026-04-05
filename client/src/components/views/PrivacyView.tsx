import type { RoundUpdate } from '../../types/metrics';
import { PrivacyGauge } from '../charts/PrivacyGauge';
import { formatEpsilon, formatDuration } from '../../utils/format';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface PrivacyViewProps {
  updates: RoundUpdate[];
}

export function PrivacyView({ updates }: PrivacyViewProps) {
  const latest = updates.length > 0 ? updates[updates.length - 1] : null;
  const epsilonHistory = updates
    .filter((u) => u.metrics.epsilon_spent !== undefined)
    .map((u) => ({ round: u.round, epsilon: u.metrics.epsilon_spent! }));

  const currentEpsilon = latest?.metrics.epsilon_spent ?? 0;
  const encStatus = latest?.metrics.encryption_status;
  const encLatency = latest?.metrics.encryption_latency_ms;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white">Privacy Analysis</h2>
        <p className="text-sm text-slate-400 mt-1">Differential privacy budget and encryption status</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Privacy gauge */}
        <PrivacyGauge epsilonSpent={currentEpsilon} epsilonTarget={10} />

        {/* Encryption status */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 space-y-4">
          <h3 className="text-sm font-medium text-slate-300">Encryption & DP Status</h3>

          <div className="space-y-3">
            {/* Encryption status */}
            <div className="flex items-center justify-between py-2 border-b border-slate-700">
              <span className="text-sm text-slate-400">Encryption Status</span>
              <span
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  encStatus === 'active'
                    ? 'bg-green-900/50 text-green-400'
                    : encStatus
                    ? 'bg-yellow-900/50 text-yellow-400'
                    : 'bg-slate-700 text-slate-400'
                }`}
              >
                {encStatus ?? 'Unknown'}
              </span>
            </div>

            {/* Encryption latency */}
            <div className="flex items-center justify-between py-2 border-b border-slate-700">
              <span className="text-sm text-slate-400">Encryption Latency</span>
              <span className="text-sm text-white font-mono">
                {formatDuration(encLatency)}
              </span>
            </div>

            {/* Current epsilon */}
            <div className="flex items-center justify-between py-2 border-b border-slate-700">
              <span className="text-sm text-slate-400">Epsilon Spent</span>
              <span className="text-sm text-purple-300 font-mono">
                ε = {formatEpsilon(currentEpsilon)}
              </span>
            </div>

            {/* Total rounds with DP */}
            <div className="flex items-center justify-between py-2 border-b border-slate-700">
              <span className="text-sm text-slate-400">Rounds with DP</span>
              <span className="text-sm text-white">{epsilonHistory.length}</span>
            </div>

            {/* Privacy method */}
            <div className="flex items-center justify-between py-2">
              <span className="text-sm text-slate-400">Method</span>
              <span className="text-sm text-white">{latest?.method ?? '—'}</span>
            </div>
          </div>

          {/* DP info box */}
          <div className="bg-purple-900/20 rounded-lg border border-purple-800/40 p-3">
            <p className="text-xs text-purple-300 leading-relaxed">
              Differential privacy is enforced via Gaussian noise mechanism.
              The privacy budget (ε) accumulates across rounds via Rényi DP composition.
              Lower ε means stronger privacy guarantees.
            </p>
          </div>
        </div>
      </div>

      {/* Epsilon over time chart */}
      {epsilonHistory.length > 0 && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Privacy Budget Over Time</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={epsilonHistory} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
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
                tickFormatter={(v) => `ε=${v.toFixed(1)}`}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                labelStyle={{ color: '#94a3b8', fontSize: 11 }}
                formatter={(value) => [`ε=${Number(value).toFixed(3)}`, 'Privacy Budget']}
              />
              <Line
                type="monotone"
                dataKey="epsilon"
                stroke="#a855f7"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#a855f7' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
