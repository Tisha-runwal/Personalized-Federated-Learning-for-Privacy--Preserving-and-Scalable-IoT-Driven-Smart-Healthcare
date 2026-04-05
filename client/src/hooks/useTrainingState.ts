import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import type { RoundUpdate, TrainingConfig, TrainingStatus } from '../types/metrics';

interface TrainingStateResult {
  connected: boolean;
  updates: RoundUpdate[];
  status: TrainingStatus;
  startTraining: (config: TrainingConfig) => Promise<void>;
  stopTraining: () => Promise<void>;
}

const defaultStatus: TrainingStatus = {
  status: 'idle',
  method: null,
  round: 0,
  total_rounds: 0,
};

export function useTrainingState(): TrainingStateResult {
  const { connected, lastMessage } = useWebSocket('/ws/live');
  const [updates, setUpdates] = useState<RoundUpdate[]>([]);
  const [status, setStatus] = useState<TrainingStatus>(defaultStatus);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;
    try {
      const data = JSON.parse(lastMessage.data);
      if (data.type === 'round_update') {
        setUpdates((prev) => [...prev, data as RoundUpdate]);
      } else if (data.type === 'status') {
        setStatus(data as TrainingStatus);
      }
    } catch {
      // Ignore parse errors
    }
  }, [lastMessage]);

  // Poll /api/training/status every 5 seconds
  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/training/status');
      if (res.ok) {
        const data: TrainingStatus = await res.json();
        setStatus(data);
      }
    } catch {
      // Ignore network errors during polling
    }
  }, []);

  useEffect(() => {
    pollStatus();
    pollingRef.current = setInterval(pollStatus, 5000);
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [pollStatus]);

  const startTraining = useCallback(async (config: TrainingConfig) => {
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (res.ok) {
        setUpdates([]);
        await pollStatus();
      }
    } catch (err) {
      console.error('Failed to start training:', err);
    }
  }, [pollStatus]);

  const stopTraining = useCallback(async () => {
    try {
      const res = await fetch('/api/training/stop', { method: 'POST' });
      if (res.ok) {
        await pollStatus();
      }
    } catch (err) {
      console.error('Failed to stop training:', err);
    }
  }, [pollStatus]);

  return { connected, updates, status, startTraining, stopTraining };
}
