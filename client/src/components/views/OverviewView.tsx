import type { RoundUpdate, TrainingStatus } from '../../types/metrics';
import { KpiCard } from '../widgets/KpiCard';
import { ActivityFeed } from '../widgets/ActivityFeed';
import { formatPercent, formatBytes, formatEpsilon } from '../../utils/format';

interface OverviewViewProps {
  updates: RoundUpdate[];
  status: TrainingStatus;
}

export function OverviewView({ updates, status }: OverviewViewProps) {
  const latest = updates.length > 0 ? updates[updates.length - 1] : null;
  const prev = updates.length > 1 ? updates[updates.length - 2] : null;

  const accuracy = latest?.metrics.global_accuracy;
  const loss = latest?.metrics.global_loss;
  const epsilon = latest?.metrics.epsilon_spent;
  const savingsPct = latest?.metrics.savings_percent;

  // Calculate accuracy trend
  const prevAccuracy = prev?.metrics.global_accuracy;
  const accuracyTrend =
    accuracy !== undefined && prevAccuracy !== undefined
      ? (accuracy - prevAccuracy) * 100
      : undefined;

  const prevLoss = prev?.metrics.global_loss;
  const lossTrend =
    loss !== undefined && prevLoss !== undefined
      ? -((loss - prevLoss) / Math.max(prevLoss, 1e-6)) * 100
      : undefined;

  return (
    <div className="space-y-6">
      {/* Page title */}
      <div>
        <h2 className="text-xl font-semibold text-white">Overview</h2>
        <p className="text-sm text-slate-400 mt-1">Real-time federated learning metrics</p>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard
          label="Global Accuracy"
          value={accuracy !== undefined ? formatPercent(accuracy) : '—'}
          subValue={`Round ${status.round}`}
          trend={accuracyTrend}
          accentColor="#3b82f6"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          }
        />
        <KpiCard
          label="Global Loss"
          value={loss !== undefined ? loss.toFixed(4) : '—'}
          subValue="Cross-entropy"
          trend={lossTrend}
          accentColor="#f59e0b"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
              <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
            </svg>
          }
        />
        <KpiCard
          label="Privacy Budget"
          value={epsilon !== undefined ? `ε=${formatEpsilon(epsilon)}` : '—'}
          subValue="Epsilon spent (DP)"
          accentColor="#a855f7"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          }
        />
        <KpiCard
          label="Bandwidth Saved"
          value={savingsPct !== undefined ? `${savingsPct.toFixed(1)}%` : '—'}
          subValue={
            latest?.metrics.bytes_original !== undefined
              ? `${formatBytes(latest.metrics.bytes_original)} → ${formatBytes(latest.metrics.bytes_quantized)}`
              : 'Quantization savings'
          }
          accentColor="#06b6d4"
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          }
        />
      </div>

      {/* Activity feed + network topology */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ActivityFeed updates={updates} />

        {/* Network topology placeholder */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Network Topology</h3>
          <div className="flex flex-col items-center justify-center h-48 text-slate-500">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-3 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
            </svg>
            <p className="text-sm">Network topology visualization</p>
            <p className="text-xs mt-1 text-slate-600">
              {updates.length > 0
                ? `${new Set(updates.map(u => u.metrics.clients_selected?.length ?? 0)).size} client configurations observed`
                : 'Start training to see client topology'}
            </p>
            {latest?.metrics.clients_selected && (
              <div className="mt-3 flex flex-wrap gap-1 justify-center max-w-xs">
                {latest.metrics.clients_selected.slice(0, 10).map((cid) => (
                  <span key={cid} className="text-xs bg-blue-900/50 text-blue-300 rounded px-1.5 py-0.5">
                    C{cid}
                  </span>
                ))}
                {(latest.metrics.clients_selected.length > 10) && (
                  <span className="text-xs text-slate-500">+{latest.metrics.clients_selected.length - 10} more</span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
