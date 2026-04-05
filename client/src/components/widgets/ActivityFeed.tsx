import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { RoundUpdate } from '../../types/metrics';
import { formatPercent, formatEpsilon } from '../../utils/format';

interface ActivityFeedProps {
  updates: RoundUpdate[];
  maxItems?: number;
}

export function ActivityFeed({ updates, maxItems = 20 }: ActivityFeedProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const recent = [...updates].reverse().slice(0, maxItems);

  // Auto-scroll to bottom when new items arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [updates.length]);

  if (updates.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
        <h3 className="text-sm font-medium text-slate-300 mb-4">Activity Feed</h3>
        <div className="flex flex-col items-center justify-center h-32 text-slate-500">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-sm">No training events yet</p>
          <p className="text-xs mt-1">Start training to see live updates</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 flex flex-col">
      <h3 className="text-sm font-medium text-slate-300 mb-4">Activity Feed</h3>
      <div className="flex-1 overflow-y-auto max-h-64 space-y-2 pr-1">
        <AnimatePresence initial={false}>
          {recent.map((update, idx) => (
            <motion.div
              key={`${update.round}-${idx}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="flex items-center gap-3 py-2 px-3 rounded-lg bg-slate-700/50 text-xs"
            >
              {/* Round badge */}
              <span className="flex-shrink-0 w-14 text-center bg-blue-900/60 text-blue-300 rounded px-1.5 py-0.5 font-mono">
                R{update.round}
              </span>

              {/* Method */}
              <span className="text-slate-300 flex-shrink-0 w-20 truncate">{update.method}</span>

              {/* Accuracy */}
              <div className="flex items-center gap-1">
                <span className="text-slate-500">Acc:</span>
                <span className={
                  (update.metrics.global_accuracy ?? 0) >= 0.8
                    ? 'text-green-400'
                    : (update.metrics.global_accuracy ?? 0) >= 0.6
                    ? 'text-yellow-400'
                    : 'text-red-400'
                }>
                  {formatPercent(update.metrics.global_accuracy)}
                </span>
              </div>

              {/* Epsilon */}
              {update.metrics.epsilon_spent !== undefined && (
                <div className="flex items-center gap-1 ml-auto">
                  <span className="text-slate-500">ε:</span>
                  <span className="text-purple-300">{formatEpsilon(update.metrics.epsilon_spent)}</span>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
