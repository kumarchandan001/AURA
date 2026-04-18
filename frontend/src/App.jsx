import { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, Radio, Wifi, WifiOff } from 'lucide-react';
import VideoFeed from './components/VideoFeed';
import BiometricsPanel from './components/BiometricsPanel';
import ChatPanel from './components/ChatPanel';

function App() {
  // ── Telemetry state ──────────────────────────────────────
  const [telemetry, setTelemetry] = useState({
    bpm: null,
    rmssd: null,
    state: 'Unknown',
    fps: 0,
  });
  const [pulseData, setPulseData] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  // ── WebSocket connection with auto-reconnect ─────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`);

    ws.onopen = () => {
      setIsConnected(true);
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;
      // Auto-reconnect after 2 seconds.
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setTelemetry({
          bpm: data.bpm,
          rmssd: data.rmssd,
          state: data.state || 'Unknown',
          fps: data.fps || 0,
        });

        // Accumulate pulse waveform data from batched green values.
        if (data.pulse && data.pulse.length > 0) {
          setPulseData((prev) => {
            const updated = [...prev, ...data.pulse];
            // Keep last 300 points (~10s at 30fps).
            return updated.length > 300 ? updated.slice(-300) : updated;
          });
        }
      } catch {
        // Ignore parse errors.
      }
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // ══════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════
  return (
    <div className="h-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden">
      {/* ── Header ────────────────────────────────────────── */}
      <header className="h-14 min-h-[56px] bg-slate-900/80 backdrop-blur-md border-b border-slate-800/60 flex items-center justify-between px-5 z-10">
        {/* Left: Branding */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Activity className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight">
              <span className="text-cyan-400">SENTIO</span>
              <span className="text-slate-400 font-normal ml-2 hidden sm:inline">
                Contactless Affective Computing Framework
              </span>
            </h1>
          </div>
        </div>

        {/* Right: Status */}
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-1.5 text-xs font-medium ${
            isConnected ? 'text-emerald-400' : 'text-rose-400'
          }`}>
            {isConnected ? (
              <Wifi className="w-3.5 h-3.5" />
            ) : (
              <WifiOff className="w-3.5 h-3.5" />
            )}
            {isConnected ? 'Live' : 'Reconnecting…'}
          </div>
          <div className="w-px h-5 bg-slate-700" />
          <div className="flex items-center gap-1.5">
            <Radio className={`w-3 h-3 ${isConnected ? 'text-emerald-400 animate-pulse' : 'text-slate-600'}`} />
            <span className="text-[11px] text-slate-500 font-mono">
              v1.0
            </span>
          </div>
        </div>
      </header>

      {/* ── Main Dashboard Grid ───────────────────────────── */}
      <main className="flex-1 grid grid-cols-5 gap-3 p-3 min-h-0">
        {/* Left: Video Feed (3 columns) */}
        <div className="col-span-3 min-h-0">
          <VideoFeed fps={telemetry.fps} isConnected={isConnected} />
        </div>

        {/* Right Column (2 columns) */}
        <div className="col-span-2 flex flex-col gap-3 min-h-0 overflow-hidden">
          {/* Top: Biometrics */}
          <BiometricsPanel
            bpm={telemetry.bpm}
            rmssd={telemetry.rmssd}
            state={telemetry.state}
            pulseData={pulseData}
          />
          {/* Bottom: Chat */}
          <ChatPanel state={telemetry.state} />
        </div>
      </main>
    </div>
  );
}

export default App;
