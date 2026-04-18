import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

/**
 * PulseChart — Live scrolling rPPG pulse waveform.
 *
 * Normalizes incoming green-channel spatial-mean values and renders them
 * as a smooth cyan line chart using recharts.  The chart auto-scrolls
 * to show the most recent 8 seconds of data.
 */
function PulseChart({ data = [] }) {
  // Normalize the data for display (zero-mean, unit-variance).
  const chartData = useMemo(() => {
    if (!data || data.length < 4) return [];

    // Only show last 8 seconds.
    const tMax = data[data.length - 1].t;
    const tMin = Math.max(0, tMax - 8);
    const visible = data.filter((d) => d.t >= tMin);

    if (visible.length < 2) return [];

    const values = visible.map((d) => d.v);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std =
      Math.sqrt(
        values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length
      ) || 1;

    return visible.map((d) => ({
      t: +(d.t).toFixed(2),
      v: +((d.v - mean) / std).toFixed(3),
    }));
  }, [data]);

  if (chartData.length < 4) {
    return (
      <div className="h-[120px] bg-slate-800/30 rounded-lg border border-slate-700/30 flex items-center justify-center">
        <span className="text-[11px] text-slate-600 font-medium">
          ⏳ Acquiring rPPG signal…
        </span>
      </div>
    );
  }

  return (
    <div className="h-[120px] bg-slate-800/30 rounded-lg border border-slate-700/30 px-2 pt-1">
      <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em] font-medium pl-1 mb-0.5">
        Live rPPG Pulse Waveform
      </div>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <XAxis dataKey="t" hide />
          <YAxis hide domain={['auto', 'auto']} />
          <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="v"
            stroke="#22d3ee"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PulseChart;
