"use client";

import { useState, useMemo, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import type { Message, Citation } from "@/types/chat";
import { CitationBadge } from "./CitationCard";
import { Skeleton } from "@/components/ui/skeleton";
import { Database } from "lucide-react";

interface MessageBubbleProps {
  message: Message;
  onCitationClick?: (citation: Citation) => void;
}

/**
 * Source card for horizontal scrolling display.
 */
function SourceCard({
  citation,
  onClick,
}: {
  citation: Citation;
  onClick: () => void;
}) {
  const relevancePercent = citation.source?.relevance_score
    ? Math.round(citation.source.relevance_score * 100)
    : null;

  return (
    <button
      onClick={onClick}
      className="flex flex-col text-left min-w-[180px] max-w-[180px] p-3 border border-zinc-200 bg-white hover:bg-zinc-50 hover:border-red-500/50 transition-all group rounded-sm shadow-sm hover:shadow-md"
    >
      <div className="flex justify-between items-start w-full mb-2">
        <div className="bg-zinc-100 text-zinc-500 text-[9px] px-1 py-0.5 font-mono border border-zinc-200">
          [{citation.ref}]
        </div>
        {relevancePercent !== null && (
          <span className="text-red-600 text-[10px] font-mono font-bold opacity-0 group-hover:opacity-100 transition-opacity">
            {relevancePercent}% MATCH
          </span>
        )}
      </div>
      <h4 className="text-xs font-bold text-zinc-900 line-clamp-2 mb-1 leading-tight group-hover:text-red-600 transition-colors">
        {citation.source?.title || `Source ${citation.ref}`}
      </h4>
      {citation.claim && (
        <div className="text-[10px] text-zinc-500 line-clamp-2">
          {citation.claim}
        </div>
      )}
    </button>
  );
}

/**
 * Renders text with citation badges [1], [2], etc.
 */
function TextWithCitations({
  text,
  citationMap,
  activeCitation,
  onCitationClick,
}: {
  text: string;
  citationMap: Map<number, Citation>;
  activeCitation: number | null;
  onCitationClick: (citation: Citation) => void;
}) {
  // Split text by citation references [1], [2], etc.
  const parts = text.split(/(\[\d+\])/g);

  return (
    <>
      {parts.map((part, index) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          const refNum = parseInt(match[1], 10);
          const citation = citationMap.get(refNum);
          if (citation) {
            return (
              <CitationBadge
                key={index}
                refNum={refNum}
                isActive={activeCitation === refNum}
                onClick={() => onCitationClick(citation)}
                citation={citation}
              />
            );
          }
        }
        return <span key={index}>{part}</span>;
      })}
    </>
  );
}

/**
 * Renders a single message with citations and markdown.
 */
export function MessageBubble({ message, onCitationClick }: MessageBubbleProps) {
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  const isUser = message.role === "user";
  const isStreaming = message.isStreaming;

  const citationMap = useMemo(() => {
    const citations = message.citations || [];
    return new Map(citations.map((c) => [c.ref, c]));
  }, [message.citations]);

  const handleCitationClick = useCallback((citation: Citation) => {
    if (onCitationClick) {
      onCitationClick(citation);
    } else {
      setActiveCitation((prev) => (prev === citation.ref ? null : citation.ref));
    }
  }, [onCitationClick]);

  // Helper function to process children and replace citation references
  const processChildren = useCallback((children: React.ReactNode): React.ReactNode => {
    if (!children) return children;

    if (typeof children === "string") {
      return (
        <TextWithCitations
          text={children}
          citationMap={citationMap}
          activeCitation={activeCitation}
          onCitationClick={handleCitationClick}
        />
      );
    }

    if (Array.isArray(children)) {
      return children.map((child, i) => {
        if (typeof child === "string") {
          return (
            <TextWithCitations
              key={i}
              text={child}
              citationMap={citationMap}
              activeCitation={activeCitation}
              onCitationClick={handleCitationClick}
            />
          );
        }
        return child;
      });
    }

    return children;
  }, [citationMap, activeCitation, handleCitationClick]);

  // Custom components for ReactMarkdown that handle citations
  const markdownComponents = useMemo(() => ({
    // Paragraphs with citation handling
    p: ({ children }: { children?: React.ReactNode }) => {
      const processedChildren = processChildren(children);
      return <p className="mb-4 last:mb-0">{processedChildren}</p>;
    },
    // Strong/bold text
    strong: ({ children }: { children?: React.ReactNode }) => (
      <strong className="font-bold text-zinc-900">{children}</strong>
    ),
    // Emphasis/italic
    em: ({ children }: { children?: React.ReactNode }) => (
      <em className="italic">{children}</em>
    ),
    // Lists
    ul: ({ children }: { children?: React.ReactNode }) => (
      <ul className="list-disc list-inside mb-4 space-y-1">{children}</ul>
    ),
    ol: ({ children }: { children?: React.ReactNode }) => (
      <ol className="list-decimal list-inside mb-4 space-y-1">{children}</ol>
    ),
    li: ({ children }: { children?: React.ReactNode }) => {
      const processedChildren = processChildren(children);
      return <li className="text-sm">{processedChildren}</li>;
    },
    // Code
    code: ({ children }: { children?: React.ReactNode }) => (
      <code className="bg-zinc-100 px-1.5 py-0.5 rounded text-sm font-mono text-red-600">
        {children}
      </code>
    ),
    // Headings
    h1: ({ children }: { children?: React.ReactNode }) => (
      <h1 className="text-lg font-bold mb-2 mt-4">{children}</h1>
    ),
    h2: ({ children }: { children?: React.ReactNode }) => (
      <h2 className="text-base font-bold mb-2 mt-3">{children}</h2>
    ),
    h3: ({ children }: { children?: React.ReactNode }) => (
      <h3 className="text-sm font-bold mb-1 mt-2">{children}</h3>
    ),
    // Tables
    table: ({ children }: { children?: React.ReactNode }) => (
      <div className="overflow-x-auto my-4">
        <table className="min-w-full text-sm border-collapse border border-zinc-200 rounded">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }: { children?: React.ReactNode }) => (
      <thead className="bg-zinc-100">{children}</thead>
    ),
    tbody: ({ children }: { children?: React.ReactNode }) => (
      <tbody>{children}</tbody>
    ),
    tr: ({ children }: { children?: React.ReactNode }) => (
      <tr className="border-b border-zinc-200">{children}</tr>
    ),
    th: ({ children }: { children?: React.ReactNode }) => (
      <th className="px-3 py-2 text-left font-semibold text-zinc-700 border-r border-zinc-200 last:border-r-0">
        {children}
      </th>
    ),
    td: ({ children }: { children?: React.ReactNode }) => (
      <td className="px-3 py-2 text-zinc-600 border-r border-zinc-200 last:border-r-0">
        {children}
      </td>
    ),
  }), [processChildren]);

  return (
    <div className={`flex flex-col max-w-4xl mx-auto ${isUser ? "items-end" : "items-start"}`}>
      {/* Timestamp label */}
      <div className="flex items-center gap-2 mb-2 opacity-50 text-[10px] font-mono tracking-widest text-zinc-400">
        {isUser ? "QUERY" : "SYNTHESIS"} :: {message.timestamp.toLocaleTimeString([], { hour12: false })}
      </div>

      {isUser ? (
        /* User message - large quote style */
        <div className="text-xl md:text-2xl text-zinc-900 font-medium tracking-tight border-b border-zinc-200 pb-2 w-full">
          &ldquo;{message.content}&rdquo;
        </div>
      ) : (
        <div className="w-full">
          {/* Sources Grid - horizontal scrolling */}
          {message.citations && message.citations.length > 0 && !isStreaming && (
            <div className="mb-6">
              <div className="text-[10px] font-mono text-zinc-500 mb-2 flex items-center gap-2">
                <Database size={12} /> RETRIEVED_CONTEXT
              </div>
              <div className="flex gap-3 overflow-x-auto pb-4 scrollbar-hide">
                {message.citations.map((citation) => (
                  <SourceCard
                    key={citation.ref}
                    citation={citation}
                    onClick={() => handleCitationClick(citation)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Message content with Markdown */}
          <div className="text-sm leading-7 text-zinc-700 font-mono">
            {isStreaming && !message.content ? (
              <div className="space-y-2 p-4 border border-zinc-200 bg-white rounded-sm animate-pulse shadow-sm">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            ) : (
              <div className="markdown-content">
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={markdownComponents}
                >
                  {message.content || ""}
                </ReactMarkdown>
                {isStreaming && (
                  <span className="inline-block w-1.5 h-4 ml-0.5 bg-zinc-900 cursor-blink" />
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
