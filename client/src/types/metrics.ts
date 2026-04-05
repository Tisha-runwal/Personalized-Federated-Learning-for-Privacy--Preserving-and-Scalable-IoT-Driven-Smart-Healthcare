export interface RoundMetrics {
  global_accuracy?: number;
  global_loss?: number;
  epsilon_spent?: number;
  bytes_original?: number;
  bytes_quantized?: number;
  compression_ratio?: number;
  savings_percent?: number;
  clients_selected?: number[];
  per_client_accuracy?: number[];
  per_client_gradient_norm?: number[];
  encryption_latency_ms?: number;
  encryption_status?: string;
  round_time_ms?: number;
}

export interface RoundUpdate {
  type: 'round_update';
  round: number;
  method: string;
  metrics: RoundMetrics;
}

export interface TrainingConfig {
  method: string;
  dataset: string;
  num_clients: number;
  num_rounds: number;
  noise_multiplier: number;
  k_bits: number;
  partition_alpha: number;
  learning_rate: number;
}

export interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'stopped' | string;
  method: string | null;
  round: number;
  total_rounds: number;
}

export type MethodName = 'fedavg' | 'fedprox' | 'per_fedavg' | 'pfedme' | 'pfl_hcare';

export const METHOD_COLORS: Record<MethodName, string> = {
  pfl_hcare: '#3b82f6', pfedme: '#a855f7', per_fedavg: '#fb923c', fedprox: '#9ca3af', fedavg: '#6b7280',
};

export const METHOD_LABELS: Record<MethodName, string> = {
  pfl_hcare: 'PFL-HCare', pfedme: 'pFedMe', per_fedavg: 'Per-FedAvg', fedprox: 'FedProx', fedavg: 'FedAvg',
};
