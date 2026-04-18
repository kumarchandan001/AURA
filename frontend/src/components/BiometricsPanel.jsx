import { Heart, Activity, Brain } from 'lucide-react';
import MetricCard from './MetricCard';
import PulseChart from './PulseChart';

/**
 * BiometricsPanel — Top-right telemetry dashboard.
 *
 * Displays BPM, HRV (RMSSD), and Cognitive State in bento-box cards,
 * a dynamic heart icon pulsing at the detected BPM rate, and an
 * embedded live pulse waveform chart.
 *
 * The panel border dynamically transitions from emerald (Calm) to
 * rose (Stressed) based on the cognitive state classification.
 */
function BiometricsPanel({ bpm, rmssd, state, pulseData }) {
  // ── Dynamic border based on cognitive state ──────────────
  const borderClass =
    state === 'Stressed'
      ? 'border-rose-500/50 glow-rose'
      : state === 'Calm'
        ? 'border-emerald-500/20 glow-emerald'
        : 'border-slate-700/50';

  // ── Heart animation speed synced to BPM ──────────────────
  const heartDuration = bpm && bpm > 0 ? `${60 / bpm}s` : '1s';

  // ── State config ─────────────────────────────────────────
  const stateConfig = {
    Calm:     { text: 'CALM',        color: 'emerald', icon: '✅' },
    Stressed: { text: 'STRESSED',    color: 'rose',    icon: '⚠️' },
    Unknown:  { text: 'Calibrating', color: 'amber',   icon: '⏳' },
  };
  const s = stateConfig[state] || stateConfig.Unknown;

  // ── BPM / HRV colour coding ──────────────────────────────
  const bpmColor = bpm !== null ? (bpm > 85 ? 'rose' : 'cyan') : 'slate';
  const hrvColor = rmssd !== null ? (rmssd < 30 ? 'rose' : 'cyan') : 'slate';

  return (
    <div
      className={`bg-slate-900/70 backdrop-blur-sm rounded-xl border-2 transition-all duration-700 ease-in-out p-4 ${borderClass}`}
    >
      {/* Panel header */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold text-cyan-400 flex items-center gap-2 uppercase tracking-wider">
          <Activity className="w-3.5 h-3.5" />
          Physiological Telemetry
        </h2>
        <div className="flex items-center gap-1.5">
          <Heart
            className="w-4 h-4 text-rose-500 animate-pulse-heart"
            style={{ animationDuration: heartDuration }}
            fill="currentColor"
          />
        </div>
      </div>

      {/* Metric cards grid */}
      <div className="grid grid-cols-3 gap-2.5 mb-3">
        <MetricCard
          label="Heart Rate"
          value={bpm !== null ? Math.round(bpm) : null}
          unit="BPM"
          icon={Heart}
          color={bpmColor}
        />
        <MetricCard
          label="HRV (RMSSD)"
          value={rmssd !== null ? Math.round(rmssd) : null}
          unit="ms"
          icon={Activity}
          color={hrvColor}
        />
        <MetricCard
          label="Cognitive State"
          value={s.text}
          unit={s.icon}
          icon={Brain}
          color={s.color}
        />
      </div>

      {/* Live pulse graph */}
      <PulseChart data={pulseData} />
    </div>
  );
}

export default BiometricsPanel;
