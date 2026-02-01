"use client";

import { Database, PlusCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeaderProps {
  onNewChat?: () => void;
  paperCount?: number;
}

/**
 * Minimal retro-styled header bar.
 */
export function Header({ onNewChat, paperCount = 6142 }: HeaderProps) {
  return (
    <header className="h-12 border-b border-zinc-200 flex items-center justify-between px-4 bg-white select-none z-40 relative shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="font-pixel text-red-600 text-[14px] tracking-tighter pt-1">
          Paper<span className="text-zinc-900">RAG</span>
        </div>
      </div>
      
      {/* Right side */}
      <div className="flex items-center gap-4">
        {/* Paper count */}
        <div className="flex items-center gap-1.5 text-xs text-zinc-500">
          <Database size={12} />
          <span className="font-mono">knowledge_base: 173,990 records</span>
        </div>
        
        {/* New Chat button */}
        {onNewChat && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onNewChat}
            className="gap-1.5 text-xs h-8"
          >
            <PlusCircle size={14} />
            New
          </Button>
        )}
      </div>
    </header>
  );
}
