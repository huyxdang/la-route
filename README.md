<p align="center">
  <img src="public/Paper-Rag-logo.png" alt="PaperRAG Logo" width="280"/>
</p>

<p align="center">
  <strong>A RAG system to explore research papers in AI</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#corrective-rag-flow">Corrective RAG Flow</a> •
  <a href="#pipeline-flow">Pipeline Flow</a> •
  <a href="#tech-stack">Tech Stack</a>
</p>

---

## Overview

PaperRAG is a production-ready Retrieval-Augmented Generation (RAG) system that enables conversational exploration of ~6,000 NeurIPS 2025 research papers. 

Implements [**Adaptive RAG**](https://arxiv.org/pdf/2403.14403), [**Corrective RAG**](https://arxiv.org/pdf/2401.15884), and [**Self-RAG**](https://arxiv.org/pdf/2310.11511), orchestrated with LangGraph. 

Powered by **Mistral models** (Mistral Large, Mistral Small, Ministral 3B, Mistral Embed), Cohere (Re-ranker) and Tavily (Web-search).

## RAG Flow

<p>
  <img src="public/RAG-flow.jpg" alt="Rag-workflow"/>
</p>

### Pipeline Flow

1. **Query Rewriting** — Transforms users questions into search queries using conversation history + user's intent, to make downstream retrieval more accurate.

2. **Query Routing** — Classifies queries into three paths:
   - `conversational`: Greetings, off-topic → Direct LLM response
   - `vectorstore`: Research questions (that are available in VectorDB) → Full RAG pipeline
   - `web_search`: Other → Web search fallback

3. **Hybrid Retrieval** — Combines semantic (dense) and keyword (BM25 sparse) search via Pinecone

4. **Reranking** — Cohere Rerank v3.5 reorders documents by relevance

5. **Document Grading** — LLM evaluates each document's relevance; triggers web search if insufficient

6. **Answer Generation** — Streams response with inline citations `[1]`, `[2]`, etc.

7. **Citation Extraction** — Parses structured citation metadata from LLM output

8. **Generation Grading** — Validates answer for grounding (hallucination check) and usefulness

## Features

- **Streaming Responses** — Real-time token streaming via Server-Sent Events (SSE)
- **Structured Citations** — Click-to-inspect source cards with relevance scores
- **Intelligent Routing** — Automatic classification of query intent
- **Hybrid Search** — Best of semantic understanding + keyword matching
- **Corrective RAG** — Self-healing pipeline with fallback strategies
- **Latency Tracking** — Per-step timing metrics in the RAG console

## Tech Stack

### Backend
| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| Streaming | SSE (sse-starlette) |
| Orchestration | LangGraph |
| Vector DB | Pinecone (hybrid sparse-dense) |
| Embeddings | Mistral Embed |
| Reranking | Cohere Rerank v3.5 |
| LLMs | Mistral Large, Mistral Small, Ministral 3B |
| Sessions | Redis |
| Web Search | Tavily |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS |
| Components | shadcn/ui |
| Fonts | JetBrains Mono, Press Start 2P |
| Streaming | EventSource API |
| Markdown | react-markdown |


## Project Structure

```
la-route/
├── crag/                    # Backend package
│   ├── api.py               # FastAPI endpoints
│   ├── streaming.py         # SSE streaming pipeline
│   ├── graders.py           # Router, rewriter, graders
│   ├── retrieval.py         # Pinecone hybrid retriever
│   ├── citations.py         # Citation extraction
│   ├── session.py           # Session management
│   ├── config.py            # Model configurations
│   └── graph.py             # LangGraph workflow
├── frontend/                # Next.js frontend
│   ├── src/
│   │   ├── app/             # Next.js app router
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom hooks (useChat)
│   │   └── lib/             # API client
├── public/                  # Static assets
├── requirements.txt         # Python dependencies
├── Procfile                 # Railway start command
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) for details.
