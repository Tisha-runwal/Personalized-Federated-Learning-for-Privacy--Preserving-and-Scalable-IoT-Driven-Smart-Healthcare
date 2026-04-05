import type { RoundUpdate } from '../../types/metrics';
import { BandwidthChart } from '../charts/BandwidthChart';
import { formatBytes } from '../../utils/format';

interface CommunicationViewProps {
  updates: RoundUpdate[];
}

export function CommunicationView({ updates }: CommunicationViewProps) {
  const latest = updates.length > 0 ? updates[updates.length - 1] : null;

  // Aggregate totals
  let totalOriginal = 0;
  let totalQuantized = 0;
  for (const u of updates) {
    totalOriginal += u.metrics.bytes_original ?? 0;
    totalQuantized += u.metrics.bytes_quantized ?? 0;
  }
  const totalSaved = totalOriginal - totalQuantized;
  const overallSavings = totalOriginal > 0 ? (totalSaved / totalOriginal) * 100 : 0;

  const currentSavings = latest?.metrics.savings_percent ?? 0;
  const compressionRatio = latest?.metrics.compression_ratio;

  const stats = [
    {
      label: 'Total Original',
      value: formatBytes(totalOriginal),
      subValue: 'Cumulative uncompressed',
      color: '#6b7280',
    },
    {
      label: 'Total Quantized',
      value: formatBytes(totalQuantized),
      subValue: 'Cumulative compressed',
      color: '#3b82f6',
    },
    {
      label: 'Total Saved',
      value: formatBytes(totalSaved),
      subValue: `${overallSavings.toFixed(1)}% overall reduction`,
      color: '#22c55e',
    },
    {
      label: 'Latest Savings',
      value: `${currentSavings.toFixed(1)}%`,
      subValue: compressionRatio ? `${compressionRatio.toFixed(2)}x compression` : 'Per-round reduction',
      color: '#06b6d4',
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white">Communication Efficiency</h2>
        <p className="text-sm text-slate-400 mt-1">Bandwidth reduction via gradient quantization</p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="bg-slate-800 rounded-xl border border-slate-700 p-5"
          >
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">{stat.label}</p>
            <p className="text-2xl font-bold text-white">{stat.value}</p>
            <p className="text-xs text-slate-500 mt-1">{stat.subValue}</p>
            <div className="mt-3 h-0.5 rounded-full bg-slate-700">
              <div className="h-0.5 rounded-full w-1/2" style={{ backgroundColor: stat.color }} />
            </div>
          </div>
        ))}
      </div>

      {/* Bandwidth chart */}
      <BandwidthChart updates={updates} />

      {/* Per-round details table */}
      {updates.length > 0 && (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Per-Round Details</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-400 border-b border-slate-700">
                  <th className="text-left py-2 pr-4">Round</th>
                  <th className="text-left py-2 pr-4">Method</th>
                  <th className="text-right py-2 pr-4">Original</th>
                  <th className="text-right py-2 pr-4">Quantized</th>
                  <th className="text-right py-2 pr-4">Saved</th>
                  <th className="text-right py-2">Ratio</th>
                </tr>
              </thead>
              <tbody>
                {[...updates].reverse().slice(0, 10).map((u, idx) => (
                  <tr key={idx} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="py-2 pr-4 font-mono text-slate-300">R{u.round}</td>
                    <td className="py-2 pr-4 text-slate-300">{u.method}</td>
                    <td className="py-2 pr-4 text-right text-slate-400">
                      {formatBytes(u.metrics.bytes_original)}
                    </td>
                    <td className="py-2 pr-4 text-right text-blue-400">
                      {formatBytes(u.metrics.bytes_quantized)}
                    </td>
                    <td className="py-2 pr-4 text-right text-green-400">
                      {u.metrics.savings_percent !== undefined
                        ? `${u.metrics.savings_percent.toFixed(1)}%`
                        : '—'}
                    </td>
                    <td className="py-2 text-right text-cyan-400">
                      {u.metrics.compression_ratio !== undefined
                        ? `${u.metrics.compression_ratio.toFixed(2)}x`
                        : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
