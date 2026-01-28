import { useState, useRef, useEffect } from 'react';
import {
  ingestDocument,
  askQuestion,
  type ChunkInfo,
  type RoutingInfo,
} from './utils/api';

// Types
interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: ChunkInfo[];
  routing?: RoutingInfo;
}

// Theme Toggle Component
function ThemeToggle() {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== 'undefined') {
      return document.documentElement.classList.contains('dark');
    }
    return false;
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  return (
    <button
      onClick={() => setIsDark(!isDark)}
      className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      aria-label="Toggle theme"
    >
      {isDark ? (
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
        </svg>
      ) : (
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
        </svg>
      )}
    </button>
  );
}

// Citation Panel Component
function CitationPanel({ 
  citation, 
  onClose 
}: { 
  citation: ChunkInfo | null; 
  onClose: () => void;
}) {
  if (!citation) return null;

  const highlightText = () => {
    const text = citation.text;
    const start = citation.highlight_start ?? 0;
    const end = citation.highlight_end ?? Math.min(100, text.length);
    
    return (
      <>
        {text.slice(0, start)}
        <mark className="bg-mistral-orange/30 px-0.5 rounded">{text.slice(start, end)}</mark>
        {text.slice(end)}
      </>
    );
  };

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-white dark:bg-mistral-dark border-l border-gray-200 dark:border-gray-700 shadow-xl slide-in z-50">
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="font-semibold">Source [Chunk {citation.chunk_id + 1}]</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-1 bg-mistral-orange/10 text-mistral-orange rounded">
            {(citation.relevance_score * 100).toFixed(0)}% relevant
          </span>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
      <div className="p-4 overflow-auto max-h-[calc(100vh-80px)]">
        <p className="text-sm leading-relaxed whitespace-pre-wrap">
          {highlightText()}
        </p>
      </div>
    </div>
  );
}

// Routing Dashboard Component
function RoutingDashboard({
  routing,
  totalCost,
}: {
  routing: RoutingInfo | null;
  totalCost: number;
}) {
  if (!routing) return null;

  const isSmall = routing.model_used.includes('8b') || routing.model_used.includes('ministral');
  const riskColors = {
    low: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    high: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-mistral-darker border-t border-gray-200 dark:border-gray-700 p-3 z-40">
      <div className="max-w-6xl mx-auto flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-4">
          {/* Model Badge */}
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            isSmall 
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
              : 'bg-mistral-orange/20 text-mistral-orange'
          }`}>
            {isSmall ? 'Ministral 8B' : 'Mistral Large'}
          </span>

          {/* Confidence */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">Confidence</span>
            <div className="w-20 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-mistral-orange transition-all"
                style={{ width: `${routing.confidence * 100}%` }}
              />
            </div>
            <span className="text-xs font-medium">{(routing.confidence * 100).toFixed(0)}%</span>
          </div>

          {/* Risk Level */}
          <span className={`px-2 py-1 rounded text-xs font-medium ${riskColors[routing.risk_level]}`}>
            {routing.risk_level.toUpperCase()}
          </span>
        </div>

        <div className="flex items-center gap-6 text-sm">
          {/* Cost */}
          <div className="text-right">
            <div className="text-gray-500 dark:text-gray-400 text-xs">This query</div>
            <div className="font-mono">${routing.cost_estimate_usd.toFixed(6)}</div>
          </div>
          <div className="text-right">
            <div className="text-gray-500 dark:text-gray-400 text-xs">Session total</div>
            <div className="font-mono font-medium">${totalCost.toFixed(6)}</div>
          </div>

          {/* Latency */}
          <div className="text-right">
            <div className="text-gray-500 dark:text-gray-400 text-xs">Latency</div>
            <div className="font-mono">{routing.latency_ms}ms</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Parse citations from answer text
function parseAnswer(answer: string, onCitationClick: (id: number) => void) {
  const parts = answer.split(/(\[\d+\])/g);
  
  return parts.map((part, i) => {
    const match = part.match(/\[(\d+)\]/);
    if (match) {
      const citationId = parseInt(match[1]) - 1;
      return (
        <button
          key={i}
          onClick={() => onCitationClick(citationId)}
          className="citation-badge mx-0.5"
        >
          {match[1]}
        </button>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// Main App Component
export default function App() {
  // State
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [docText, setDocText] = useState('');
  const [docType, setDocType] = useState<'policy' | 'contract' | 'legal' | 'technical' | 'general'>('general');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<ChunkInfo | null>(null);
  const [lastRouting, setLastRouting] = useState<RoutingInfo | null>(null);
  const [totalCost, setTotalCost] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load theme on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      document.documentElement.classList.add('dark');
    }
  }, []);

  // Ingest document
  const handleIngest = async () => {
    if (!docText.trim()) {
      setError('Please enter document text');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await ingestDocument({
        text: docText,
        doc_type: docType,
      });
      setSessionId(response.session_id);
      setMessages([{
        role: 'assistant',
        content: `📄 Document loaded successfully!\n\n**${response.chunks_created}** chunks created from **${response.total_tokens}** tokens.\nDocument type: **${response.doc_type}**\n\nYou can now ask questions about the document.`,
      }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to ingest document');
    } finally {
      setIsLoading(false);
    }
  };

  // Ask question
  const handleAsk = async () => {
    if (!input.trim() || !sessionId) return;

    const question = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: question }]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await askQuestion({
        session_id: sessionId,
        question,
      });

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        routing: response.routing,
      }]);
      setLastRouting(response.routing);
      setTotalCost(prev => prev + response.routing.cost_estimate_usd);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get answer');
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Sorry, I encountered an error processing your question.',
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle citation click
  const handleCitationClick = (citationId: number, citations?: ChunkInfo[]) => {
    if (!citations) return;
    const citation = citations.find(c => c.chunk_id === citationId);
    if (citation) {
      setSelectedCitation(citation);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-mistral-darker">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white dark:bg-mistral-dark border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold">
              <span className="text-mistral-orange">Le</span>-Route
            </h1>
            {sessionId && (
              <span className="text-xs px-2 py-1 bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded">
                Session: {sessionId}
              </span>
            )}
          </div>
          <ThemeToggle />
        </div>
      </header>

      {/* Main Content */}
      <main className={`max-w-6xl mx-auto p-4 ${lastRouting ? 'pb-24' : ''}`}>
        {!sessionId ? (
          /* Document Input */
          <div className="max-w-2xl mx-auto fade-in">
            <div className="bg-white dark:bg-mistral-dark rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
              <h2 className="text-lg font-semibold mb-4">Load a Document</h2>
              
              {error && (
                <div className="mb-4 p-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg text-sm">
                  {error}
                </div>
              )}

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Document Type</label>
                  <select
                    value={docType}
                    onChange={(e) => setDocType(e.target.value as typeof docType)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-mistral-darker focus:ring-2 focus:ring-mistral-orange focus:border-transparent"
                  >
                    <option value="general">General</option>
                    <option value="policy">Policy</option>
                    <option value="contract">Contract</option>
                    <option value="legal">Legal</option>
                    <option value="technical">Technical</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Document Text</label>
                  <textarea
                    value={docText}
                    onChange={(e) => setDocText(e.target.value)}
                    placeholder="Paste your document here..."
                    className="w-full h-64 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-mistral-darker focus:ring-2 focus:ring-mistral-orange focus:border-transparent resize-none"
                  />
                </div>

                <button
                  onClick={handleIngest}
                  disabled={isLoading || !docText.trim()}
                  className="w-full py-3 bg-mistral-orange hover:bg-mistral-orange-light disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <div className="spinner" />
                      Processing...
                    </>
                  ) : (
                    'Load Document'
                  )}
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* Chat Interface */
          <div className="flex gap-4">
            <div className={`flex-1 transition-all ${selectedCitation ? 'mr-96' : ''}`}>
              {/* Messages */}
              <div className="space-y-4 mb-4">
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] px-4 py-3 rounded-2xl ${
                        msg.role === 'user'
                          ? 'bg-mistral-orange text-white'
                          : 'bg-white dark:bg-mistral-dark border border-gray-200 dark:border-gray-700'
                      }`}
                    >
                      <div className="text-sm whitespace-pre-wrap">
                        {msg.role === 'assistant' && msg.citations
                          ? parseAnswer(msg.content, (id) => handleCitationClick(id, msg.citations))
                          : msg.content}
                      </div>
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white dark:bg-mistral-dark border border-gray-200 dark:border-gray-700 px-4 py-3 rounded-2xl">
                      <div className="flex items-center gap-2">
                        <div className="spinner" />
                        <span className="text-sm text-gray-500">Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="sticky bottom-20 bg-gray-50 dark:bg-mistral-darker pt-4">
                {error && (
                  <div className="mb-2 p-2 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded text-sm">
                    {error}
                  </div>
                )}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleAsk()}
                    placeholder="Ask a question about the document..."
                    disabled={isLoading}
                    className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-mistral-dark focus:ring-2 focus:ring-mistral-orange focus:border-transparent"
                  />
                  <button
                    onClick={handleAsk}
                    disabled={isLoading || !input.trim()}
                    className="px-6 py-3 bg-mistral-orange hover:bg-mistral-orange-light disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-xl transition-colors"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Citation Panel */}
      <CitationPanel
        citation={selectedCitation}
        onClose={() => setSelectedCitation(null)}
      />

      {/* Routing Dashboard */}
      <RoutingDashboard routing={lastRouting} totalCost={totalCost} />
    </div>
  );
}
