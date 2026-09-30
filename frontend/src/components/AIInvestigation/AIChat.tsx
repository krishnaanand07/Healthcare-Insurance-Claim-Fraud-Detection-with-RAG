import React, { useState } from 'react';
import { Send, Bot, User, Sparkles, Loader2, MessageSquare } from 'lucide-react';

interface AIChatProps {
  claimData: any;
  investigationId?: string;
}

interface ChatMessage {
  sender: 'user' | 'ai';
  text: string;
  sources?: any[];
}

export const AIChat: React.FC<AIChatProps> = ({ claimData, investigationId }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      sender: 'ai',
      text: "Hello! I am your AI Claim Investigation Assistant. Ask me anything about this claim, flagged anomalies, or relevant healthcare fraud guidelines."
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [activeId, setActiveId] = useState<string | undefined>(investigationId);

  const sampleQuestions = [
    "Why was this claim flagged?",
    "What evidence supports this assessment?",
    "What should an investigator verify?",
    "Could this be a false positive?"
  ];

  const handleSend = async (queryText?: string) => {
    const query = queryText || inputQuery;
    if (!query.trim() || isSending) return;

    const userMsg: ChatMessage = { sender: 'user', text: query };
    setMessages(prev => [...prev, userMsg]);
    if (!queryText) setInputQuery('');
    setIsSending(true);

    try {
      const response = await fetch('http://localhost:8000/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          claim: claimData,
          investigation_id: activeId
        })
      });

      if (!response.ok) {
        throw new Error('Chat service unavailable');
      }

      const data = await response.json();
      if (data.investigation_id) {
        setActiveId(data.investigation_id);
      }

      const aiMsg: ChatMessage = {
        sender: 'ai',
        text: data.reply,
        sources: data.retrieved_sources
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err: any) {
      setMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: `Notice: ${err.message || 'Error reaching AI Assistant'}. Please verify backend status.`
        }
      ]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="neo-surface p-6 rounded-2xl flex flex-col h-[520px]">
      <div className="flex items-center gap-3 pb-4 border-b border-slate-200 mb-4">
        <div className="w-10 h-10 rounded-xl bg-[#172554] flex items-center justify-center text-[#FACC15]">
          <Bot size={22} />
        </div>
        <div>
          <h3 className="text-base font-bold text-[#172554] flex items-center gap-2">
            AI Investigation Assistant <Sparkles size={16} className="text-[#FACC15]" />
          </h3>
          <p className="text-xs text-slate-500 font-medium">Grounded in Healthcare Fraud Rules & RAG Retrieval</p>
        </div>
      </div>

      {/* Suggested Quick Questions */}
      <div className="flex flex-wrap gap-2 mb-4">
        {sampleQuestions.map((q, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => handleSend(q)}
            disabled={isSending}
            className="text-[11px] font-semibold text-[#172554] bg-[#EEF3FA] hover:bg-[#FACC15]/30 border border-[#172554]/10 px-3 py-1 rounded-full transition-all flex items-center gap-1"
          >
            <MessageSquare size={10} className="text-[#F59E0B]" /> {q}
          </button>
        ))}
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar mb-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-3 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.sender === 'ai' && (
              <div className="w-7 h-7 rounded-lg bg-[#172554] text-[#FACC15] flex items-center justify-center shrink-0 mt-1">
                <Bot size={14} />
              </div>
            )}
            <div
              className={`max-w-[85%] p-3.5 rounded-2xl text-xs leading-relaxed font-medium ${
                msg.sender === 'user'
                  ? 'bg-[#172554] text-white rounded-br-none shadow-md'
                  : 'bg-white border border-slate-200 text-slate-800 rounded-bl-none shadow-sm'
              }`}
            >
              {msg.text}

              {/* RAG Sources Citations */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-2 border-t border-slate-100 space-y-1">
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    CITED RAG SOURCES:
                  </p>
                  {msg.sources.map((s: any, sIdx: number) => (
                    <div key={sIdx} className="text-[10px] text-[#172554] bg-slate-50 px-2 py-1 rounded border border-slate-200">
                      📄 <strong>{s.filename}</strong> ({s.section})
                    </div>
                  ))}
                </div>
              )}
            </div>
            {msg.sender === 'user' && (
              <div className="w-7 h-7 rounded-lg bg-[#F59E0B] text-white flex items-center justify-center shrink-0 mt-1 font-bold text-xs">
                <User size={14} />
              </div>
            )}
          </div>
        ))}
        {isSending && (
          <div className="flex gap-3 justify-start">
            <div className="w-7 h-7 rounded-lg bg-[#172554] text-[#FACC15] flex items-center justify-center shrink-0">
              <Bot size={14} />
            </div>
            <div className="bg-white border border-slate-200 p-3 rounded-2xl flex items-center gap-2 text-xs text-slate-500">
              <Loader2 size={14} className="animate-spin text-[#F59E0B]" />
              Retrieving context & reasoning...
            </div>
          </div>
        )}
      </div>

      {/* Input Box */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          placeholder="Ask AI assistant about this claim..."
          className="neo-input text-xs flex-1"
          disabled={isSending}
        />
        <button
          type="submit"
          disabled={isSending || !inputQuery.trim()}
          className="neo-btn-primary px-4 py-2 text-xs font-bold flex items-center gap-1.5"
        >
          <Send size={14} /> Send
        </button>
      </form>
    </div>
  );
};
