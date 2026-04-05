import type { TrainingStatus } from '../../types/metrics';

interface HeaderProps {
  connected: boolean;
  status: TrainingStatus;
}

export function Header({ connected, status }: HeaderProps) {
  const isRunning = status.status === 'running';
  const progress =
    isRunning && status.total_rounds > 0
      ? (status.round / status.total_rounds) * 100
      : 0;

  return (
    <header className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6 gap-4 flex-shrink-0">
      {/* Title */}
      <div className="flex-1">
        <h1 className="text-lg font-semibold text-white">PFL-HCare Dashboard</h1>
        {isRunning && (
          <p className="text-xs text-slate-400 mt-0.5">
            Round {status.round} / {status.total_rounds}
            {status.method && ` — ${status.method}`}
          </p>
        )}
      </div>

      {/* Progress bar during training */}
      {isRunning && (
        <div className="flex-1 max-w-xs">
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span>Training Progress</span>
            <span>{progress.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Status badge */}
      {status.status !== 'idle' && (
        <span
          className={`text-xs px-2 py-1 rounded-full font-medium ${
            isRunning
              ? 'bg-blue-900 text-blue-300'
              : status.status === 'completed'
              ? 'bg-green-900 text-green-300'
              : status.status === 'stopped'
              ? 'bg-red-900 text-red-300'
              : 'bg-slate-700 text-slate-300'
          }`}
        >
          {status.status.charAt(0).toUpperCase() + status.status.slice(1)}
        </span>
      )}

      {/* Connection indicator */}
      <div className="flex items-center gap-2">
        <span
          className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
            connected ? 'bg-green-400 shadow-[0_0_6px_#4ade80]' : 'bg-red-400'
          }`}
        />
        <span className="text-xs text-slate-400 hidden sm:block">
          {connected ? 'Live' : 'Disconnected'}
        </span>
      </div>
    </header>
  );
}
