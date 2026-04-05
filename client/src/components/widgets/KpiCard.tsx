import { useEffect, useRef } from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';

interface KpiCardProps {
  label: string;
  value: string;
  subValue?: string;
  trend?: number; // positive = good, negative = bad
  accentColor?: string;
  icon?: React.ReactNode;
}

export function KpiCard({ label, value, subValue, trend, accentColor = '#3b82f6', icon }: KpiCardProps) {
  return (
    <motion.div
      className="bg-slate-800 rounded-xl p-5 border border-slate-700 flex flex-col gap-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
        {icon && (
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ backgroundColor: `${accentColor}20` }}>
            <span style={{ color: accentColor }}>{icon}</span>
          </div>
        )}
      </div>

      {/* Value */}
      <div className="flex items-end justify-between">
        <motion.div
          key={value}
          initial={{ scale: 1.05, opacity: 0.7 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.2 }}
        >
          <span className="text-2xl font-bold text-white">{value}</span>
          {subValue && <p className="text-xs text-slate-400 mt-0.5">{subValue}</p>}
        </motion.div>

        {/* Trend indicator */}
        {trend !== undefined && (
          <div
            className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full ${
              trend >= 0 ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'
            }`}
          >
            <span>{trend >= 0 ? '▲' : '▼'}</span>
            <span>{Math.abs(trend).toFixed(1)}%</span>
          </div>
        )}
      </div>

      {/* Accent bar */}
      <div className="h-0.5 rounded-full w-full bg-slate-700">
        <div className="h-0.5 rounded-full w-1/3" style={{ backgroundColor: accentColor }} />
      </div>
    </motion.div>
  );
}

// Animated numeric value component
interface AnimatedValueProps {
  value: number;
  format?: (n: number) => string;
}

export function AnimatedValue({ value, format }: AnimatedValueProps) {
  const spring = useSpring(value, { stiffness: 100, damping: 30 });
  const display = useTransform(spring, (n) => (format ? format(n) : n.toFixed(2)));
  const prevRef = useRef(value);

  useEffect(() => {
    if (prevRef.current !== value) {
      spring.set(value);
      prevRef.current = value;
    }
  }, [value, spring]);

  return <motion.span>{display}</motion.span>;
}
