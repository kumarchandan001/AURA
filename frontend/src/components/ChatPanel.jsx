import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Zap } from 'lucide-react';

/**
 * ChatPanel — Messenger-style Affective AI chat interface.
 *
 * User messages align right (blue), AURA AI messages align left (dark).
 * Includes a typing indicator (animated dots) while the FastAPI /api/chat
 * endpoint processes the request.
 */
function ChatPanel({ state }) {
  const [messages, setMessages] = useState([
    {
      sender: 'ai',
      text: `Welcome to Project AURA.\n\nI'm your adaptive AI assistant. My communication style adjusts in real-time based on your physiological state:\n\n• CALM → Detailed, technical responses\n• STRESSED → Concise, supportive guidance\n\nType a message below to begin.`,
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll on new messages.
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isTyping) return;

    setInput('');
    inputRef.current?.focus();

    // Append user message.
    setMessages((prev) => [...prev, { sender: 'user', text }]);

    // Append state indicator.
    setMessages((prev) => [
      ...prev,
      { sender: 'state', text: `Responding in ${state} mode` },
    ]);

    setIsTyping(true);

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await resp.json();
      setMessages((prev) => [
        ...prev,
        { sender: 'ai', text: data.response },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          sender: 'ai',
          text: '⚠️ Connection error. Ensure the backend server is running on port 8000.',
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex-1 min-h-0 bg-slate-900/70 backdrop-blur-sm rounded-xl border border-slate-700/50 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 pt-3 pb-2 border-b border-slate-800/60">
        <h2 className="text-xs font-semibold text-cyan-400 flex items-center gap-2 uppercase tracking-wider">
          <Bot className="w-3.5 h-3.5" />
          AURA AI — Adaptive Assistant
        </h2>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {messages.map((msg, i) => (
          <ChatBubble key={i} {...msg} />
        ))}
        {isTyping && <TypingIndicator />}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-slate-800/60">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            id="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask AURA anything…"
            className="flex-1 bg-slate-800/80 text-slate-200 placeholder-slate-500 rounded-lg px-4 py-2.5 text-sm outline-none border border-slate-700/60 focus:border-cyan-500/40 transition-colors duration-200"
            disabled={isTyping}
          />
          <button
            id="chat-send-btn"
            onClick={handleSend}
            disabled={isTyping || !input.trim()}
            className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-4 py-2.5 rounded-lg transition-all duration-200 flex items-center gap-1.5 text-sm font-medium"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Chat Bubble ─────────────────────────────────────────────────────

function ChatBubble({ sender, text }) {
  // State indicator (centered pill).
  if (sender === 'state') {
    return (
      <div className="flex justify-center">
        <span className="inline-flex items-center gap-1.5 text-[10px] text-cyan-400/60 bg-cyan-500/5 border border-cyan-500/10 px-3 py-1 rounded-full">
          <Zap className="w-2.5 h-2.5" />
          {text}
        </span>
      </div>
    );
  }

  const isUser = sender === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      {/* Avatar for AI */}
      {!isUser && (
        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center mr-2 mt-1 flex-shrink-0 shadow-lg shadow-cyan-500/10">
          <Bot className="w-3 h-3 text-white" />
        </div>
      )}

      <div
        className={`max-w-[85%] rounded-xl px-4 py-2.5 text-[13px] leading-relaxed ${
          isUser
            ? 'bg-blue-600/70 text-white rounded-br-sm shadow-lg shadow-blue-600/10'
            : 'bg-slate-800/70 text-slate-200 rounded-bl-sm border border-slate-700/30'
        }`}
      >
        {!isUser && (
          <div className="text-[9px] text-cyan-400/50 font-semibold uppercase tracking-wider mb-1">
            AURA AI
          </div>
        )}
        <p className="whitespace-pre-wrap">{text}</p>
      </div>

      {/* Avatar for user */}
      {isUser && (
        <div className="w-6 h-6 rounded-full bg-blue-600/30 flex items-center justify-center ml-2 mt-1 flex-shrink-0">
          <User className="w-3 h-3 text-blue-400" />
        </div>
      )}
    </div>
  );
}

// ── Typing Indicator ────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="w-6 h-6 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center mr-2 mt-1 flex-shrink-0">
        <Bot className="w-3 h-3 text-white" />
      </div>
      <div className="bg-slate-800/70 rounded-xl rounded-bl-sm px-4 py-3 border border-slate-700/30">
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-cyan-400/50 font-semibold uppercase tracking-wider">
            AURA AI
          </span>
          <div className="flex gap-1 ml-1">
            <div className="typing-dot" />
            <div className="typing-dot" />
            <div className="typing-dot" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatPanel;
