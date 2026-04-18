/**
 * MetricCard — A "bento box" style metric display card.
 *
 * Displays a label, large numeric value, unit, and optional icon.
 * Colour-coded based on the `color` prop.
 */
function MetricCard({ label, value, unit, icon: Icon, color = 'cyan', children }) {
  const display = value !== null && value !== undefined ? value : '—';

  const colorMap = {
    cyan:    { text: 'text-cyan-400',    glow: 'shadow-cyan-500/5' },
    emerald: { text: 'text-emerald-400', glow: 'shadow-emerald-500/5' },
    rose:    { text: 'text-rose-400',    glow: 'shadow-rose-500/5' },
    amber:   { text: 'text-amber-400',   glow: 'shadow-amber-500/5' },
    slate:   { text: 'text-slate-400',   glow: '' },
  };

  const c = colorMap[color] || colorMap.cyan;

  return (
    <div className={`relative bg-slate-800/40 rounded-xl border border-slate-700/40 px-4 py-4 flex flex-col items-center justify-center shadow-lg ${c.glow} transition-all duration-500`}>
      {Icon && (
        <Icon className={`w-4 h-4 mb-1.5 ${c.text} opacity-60`} />
      )}
      <span className="text-[9px] text-slate-500 uppercase tracking-[0.15em] font-medium">
        {label}
      </span>
      <span className={`text-2xl xl:text-3xl font-bold font-mono leading-tight mt-0.5 ${c.text}`}>
        {display}
      </span>
      <span className="text-[10px] text-slate-500 mt-0.5">
        {unit}
      </span>
      {children}
    </div>
  );
}

export default MetricCard;
