import type { RoundUpdate } from '../../types/metrics';
import { ConvergenceChart } from '../charts/ConvergenceChart';
import { formatPercent } from '../../utils/format';

interface ConvergenceViewProps {
  updates: RoundUpdate[];
}

export function ConvergenceView({ updates }: ConvergenceViewProps) {
  const latest = updates.length > 0 ? updates[updates.length - 1] : null;
  const methods = [...new Set(updates.map((u) => u.method))];

  // Per-method best accuracy
  const bestAccByMethod: Record<string, number> = {};
  for (const update of updates) {
    const acc = update.metrics.global_accuracy ?? 0;
    if (acc > (bestAccByMethod[update.method] ?? 0)) {
      bestAccByMethod[update.method] = acc;
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white">Convergence Analysis</h2>
        <p className="text-sm text-slate-400 mt-1">Accuracy vs. training rounds across methods</p>
      </div>

      {/* Large convergence chart */}
      <ConvergenceChart updates={updates} />

      {/* Per-method summary */}
      {methods.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {methods.map((method) => (
            <div key={method} className="bg-slate-800 rounded-lg border border-slate-700 p-4 text-center">
              <p className="text-xs text-slate-400 mb-1">{method}</p>
              <p className="text-lg font-bold text-white">
                {bestAccByMethod[method] !== undefined
                  ? formatPercent(bestAccByMethod[method])
                  : '—'}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">Best accuracy</p>
            </div>
          ))}
        </div>
      )}

      {/* Per-client accuracy */}
      {latest?.metrics.per_client_accuracy && latest.metrics.per_client_accuracy.length > 0 && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
          <h3 className="text-sm font-medium text-slate-300 mb-4">
            Per-Client Accuracy — Round {latest.round}
          </h3>
          <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-10 gap-2">
            {latest.metrics.per_client_accuracy.map((acc, idx) => (
              <div key={idx} className="flex flex-col items-center">
                <div
                  className="w-full h-12 rounded flex items-end justify-center pb-1 text-xs font-medium"
                  style={{
                    background: `linear-gradient(to top, ${acc >= 0.8 ? '#22c55e' : acc >= 0.6 ? '#eab308' : '#ef4444'}40, transparent)`,
                    border: `1px solid ${acc >= 0.8 ? '#22c55e' : acc >= 0.6 ? '#eab308' : '#ef4444'}60`,
                  }}
                >
                  <span style={{ color: acc >= 0.8 ? '#4ade80' : acc >= 0.6 ? '#facc15' : '#f87171' }}>
                    {(acc * 100).toFixed(0)}%
                  </span>
                </div>
                <span className="text-xs text-slate-500 mt-1">C{idx}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
