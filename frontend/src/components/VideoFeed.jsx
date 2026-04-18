import { Camera, CameraOff } from 'lucide-react';

/**
 * VideoFeed — Displays the live MJPEG stream from the FastAPI backend.
 *
 * The <img> tag consumes the multipart/x-mixed-replace stream directly,
 * providing native browser-level efficiency with zero JavaScript overhead.
 */
function VideoFeed({ fps = 0, isConnected = false }) {
  const fpsColor =
    fps >= 25 ? 'text-emerald-400' : fps >= 15 ? 'text-amber-400' : 'text-rose-400';

  return (
    <div className="relative h-full bg-slate-900/60 rounded-xl border border-slate-800/60 overflow-hidden shadow-2xl shadow-black/40 glow-cyan">
      {/* Video stream */}
      {isConnected ? (
        <img
          id="sentio-video-feed"
          src="/video_feed"
          alt="SENTIO Live Sensor Feed"
          className="w-full h-full object-contain bg-black"
        />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center bg-black/80 gap-3">
          <CameraOff className="w-12 h-12 text-slate-600" />
          <span className="text-sm text-slate-500">Connecting to sensor…</span>
        </div>
      )}

      {/* Top-left: Feed label */}
      <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-slate-700/40 flex items-center gap-2">
        <Camera className="w-3.5 h-3.5 text-cyan-400" />
        <span className="text-[11px] text-cyan-400 font-medium">
          Live Sensor — ROI Mesh
        </span>
      </div>

      {/* Top-right: FPS counter */}
      <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-lg border border-slate-700/40">
        <span className={`text-xs font-mono font-semibold ${fpsColor}`}>
          {fps > 0 ? `${Math.round(fps)} FPS` : '— FPS'}
        </span>
      </div>

      {/* Bottom gradient overlay */}
      <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-black/40 to-transparent pointer-events-none" />
    </div>
  );
}

export default VideoFeed;
