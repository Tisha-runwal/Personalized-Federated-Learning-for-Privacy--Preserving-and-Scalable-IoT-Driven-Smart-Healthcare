import { useState } from 'react';
import type { TrainingConfig, TrainingStatus } from '../../types/metrics';

interface ControlRibbonProps {
  status: TrainingStatus;
  onStart: (config: TrainingConfig) => void;
  onStop: () => void;
}

const DATASETS = ['synthetic_health', 'uci_har', 'mnist'] as const;
const METHODS = [
  { value: 'pfl_hcare', label: 'PFL-HCare' },
  { value: 'pfedme', label: 'pFedMe' },
  { value: 'per_fedavg', label: 'Per-FedAvg' },
  { value: 'fedprox', label: 'FedProx' },
  { value: 'fedavg', label: 'FedAvg' },
];
const K_BITS_OPTIONS = [2, 4, 8, 16, 32] as const;

export function ControlRibbon({ status, onStart, onStop }: ControlRibbonProps) {
  const [config, setConfig] = useState<TrainingConfig>({
    method: 'pfl_hcare',
    dataset: 'synthetic_health',
    num_clients: 10,
    num_rounds: 20,
    noise_multiplier: 1.0,
    k_bits: 8,
    partition_alpha: 0.5,
    learning_rate: 0.01,
  });

  const isRunning = status.status === 'running';

  const handleChange = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleStart = () => {
    onStart(config);
  };

  return (
    <div className="bg-slate-800 border-b border-slate-700 px-6 py-3 flex flex-wrap items-center gap-4">
      {/* Dataset */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-slate-400 whitespace-nowrap">Dataset</label>
        <select
          className="bg-slate-700 text-slate-100 text-xs rounded px-2 py-1.5 border border-slate-600 focus:outline-none focus:border-blue-500"
          value={config.dataset}
          onChange={(e) => handleChange('dataset', e.target.value)}
          disabled={isRunning}
        >
          {DATASETS.map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
      </div>

      {/* Method */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-slate-400 whitespace-nowrap">Method</label>
        <select
          className="bg-slate-700 text-slate-100 text-xs rounded px-2 py-1.5 border border-slate-600 focus:outline-none focus:border-blue-500"
          value={config.method}
          onChange={(e) => handleChange('method', e.target.value)}
          disabled={isRunning}
        >
          {METHODS.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
      </div>

      {/* Clients */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-slate-400 whitespace-nowrap">
          Clients: <span className="text-slate-200">{config.num_clients}</span>
        </label>
        <input
          type="range"
          min={2}
          max={50}
          step={1}
          value={config.num_clients}
          onChange={(e) => handleChange('num_clients', Number(e.target.value))}
          disabled={isRunning}
          className="w-20 accent-blue-500"
        />
      </div>

      {/* Rounds */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-slate-400 whitespace-nowrap">Rounds</label>
        <input
          type="number"
          min={1}
          max={200}
          value={config.num_rounds}
          onChange={(e) => handleChange('num_rounds', Number(e.target.value))}
          disabled={isRunning}
          className="bg-slate-700 text-slate-100 text-xs rounded px-2 py-1.5 border border-slate-600 w-16 focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* Sigma (noise multiplier) */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-slate-400 whitespace-nowrap">
          Sigma: <span className="text-slate-200">{config.noise_multiplier.toFixed(1)}</span>
        </label>
        <input
          type="range"
          min={0}
          max={3}
          step={0.1}
          value={config.noise_multiplier}
          onChange={(e) => handleChange('noise_multiplier', Number(e.target.value))}
          disabled={isRunning}
          className="w-20 accent-blue-500"
        />
      </div>

      {/* K-bits */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-slate-400 whitespace-nowrap">k-bits</label>
        <select
          className="bg-slate-700 text-slate-100 text-xs rounded px-2 py-1.5 border border-slate-600 focus:outline-none focus:border-blue-500"
          value={config.k_bits}
          onChange={(e) => handleChange('k_bits', Number(e.target.value))}
          disabled={isRunning}
        >
          {K_BITS_OPTIONS.map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Start / Stop buttons */}
      {!isRunning ? (
        <button
          onClick={handleStart}
          className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium px-4 py-2 rounded transition-colors"
        >
          Start Training
        </button>
      ) : (
        <button
          onClick={onStop}
          className="bg-red-700 hover:bg-red-600 text-white text-xs font-medium px-4 py-2 rounded transition-colors"
        >
          Stop
        </button>
      )}
    </div>
  );
}
