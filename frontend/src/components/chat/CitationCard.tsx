"use client";

import { useState } from "react";
import type { Citation } from "@/types/chat";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink, ChevronDown, ChevronUp, Quote } from "lucide-react";

interface CitationCardProps {
  citation: Citation;
  isExpanded?: boolean;
  onToggle?: () => void;
}

/**
 * Expandable card showing citation details.
 */
export function CitationCard({ citation, isExpanded, onToggle }: CitationCardProps) {
  const [localExpanded, setLocalExpanded] = useState(false);
  const expanded = isExpanded ?? localExpanded;
  const handleToggle = onToggle ?? (() => setLocalExpanded(!localExpanded));

  const source = citation.source;
  const relevancePercent = source?.relevance_score
    ? Math.round(source.relevance_score * 100)
    : null;

  return (
    <Card className="overflow-hidden">
      <CardHeader
        className="p-3 cursor-pointer hover:bg-muted/50 transition-colors"
        onClick={handleToggle}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Badge variant="outline" className="shrink-0">
              [{citation.ref}]
            </Badge>
            <CardTitle className="text-sm font-medium truncate">
              {source?.title || `Source ${citation.ref}`}
            </CardTitle>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {relevancePercent !== null && (
              <Badge
                variant={relevancePercent > 80 ? "default" : "secondary"}
                className="text-xs"
              >
                {relevancePercent}%
              </Badge>
            )}
            {expanded ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        </div>
      </CardHeader>

      {expanded && (
        <CardContent className="p-3 pt-0 space-y-3 border-t">
          {/* Claim */}
          {citation.claim && (
            <div className="text-sm">
              <span className="text-muted-foreground">Claim: </span>
              {citation.claim}
            </div>
          )}

          {/* Quote */}
          {citation.quote && (
            <div className="flex gap-2 text-sm bg-muted/50 p-2 rounded">
              <Quote className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
              <span className="italic text-muted-foreground">
                &ldquo;{citation.quote}&rdquo;
              </span>
            </div>
          )}

          {/* Section */}
          {source?.section && (
            <div className="text-sm">
              <span className="text-muted-foreground">Section: </span>
              {source.section}
            </div>
          )}

          {/* URL */}
          {source?.url && (
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
            >
              <ExternalLink className="h-3 w-3" />
              View source
            </a>
          )}
        </CardContent>
      )}
    </Card>
  );
}

interface CitationBadgeProps {
  refNum: number;
  onClick?: () => void;
  isActive?: boolean;
  citation?: Citation;
}

/**
 * Inline clickable citation badge with hover preview.
 */
export function CitationBadge({ refNum, onClick, isActive, citation }: CitationBadgeProps) {
  const [showPreview, setShowPreview] = useState(false);

  return (
    <span
      className="relative inline-block"
      onMouseEnter={() => setShowPreview(true)}
      onMouseLeave={() => setShowPreview(false)}
    >
      <button
        onClick={onClick}
        className={`
          inline-flex items-center justify-center
          text-xs font-medium
          px-1.5 py-0.5 rounded
          transition-colors
          ${
            isActive
              ? "bg-primary text-primary-foreground"
              : "bg-primary/10 text-primary hover:bg-primary/20"
          }
        `}
      >
        [{refNum}]
      </button>
      {showPreview && citation?.source && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-white border border-zinc-200 rounded-sm shadow-lg z-50 pointer-events-none">
          <div className="text-[10px] font-mono text-zinc-400 mb-1">
            [{refNum}]
            {citation.source.relevance_score ? (
              <span className="text-emerald-600 ml-2">
                {Math.round(citation.source.relevance_score * 100)}% match
              </span>
            ) : null}
          </div>
          <div className="text-xs font-bold text-zinc-900 line-clamp-2 leading-tight">
            {citation.source.title || `Source ${refNum}`}
          </div>
          {citation.claim && (
            <div className="text-[11px] text-zinc-500 line-clamp-2 mt-1">
              {citation.claim}
            </div>
          )}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-zinc-200" />
        </div>
      )}
    </span>
  );
}
