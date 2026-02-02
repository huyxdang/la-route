<p align="center">
  <img src="public/Paper-Rag-logo.png" alt="PaperRAG Logo" width="280"/>
</p>

<p align="center">
  <strong>A RAG system to explore research papers in AI</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Mistral_AI-FF7000?style=for-the-badge&logo=mistral&logoColor=white" alt="Mistral AI"/>
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white" alt="Pinecone"/>
  <img src="https://img.shields.io/badge/Cohere-39594D?style=for-the-badge&logo=cohere&logoColor=white" alt="Cohere"/>
  <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis"/>
  <img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel"/>
  <img src="https://img.shields.io/badge/Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white" alt="Railway"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#rag-flow">RAG Flow</a> •
  <a href="#chunk_&_embed">Chunk & Embed </a> •
  <a href="#tech-stack">Tech Stack</a>
</p>

---

## Overview

PaperRAG is a production-ready Retrieval-Augmented Generation (RAG) system that enables conversational exploration of ~6,000 NeurIPS 2025 research papers. 

It is built upon an **agentic RAG workflow**, combining  Adaptive RAG ([paper](https://arxiv.org/pdf/2403.14403)), Corrective RAG ([paper](https://arxiv.org/pdf/2401.15884)), and Self-RAG ([paper](https://arxiv.org/pdf/2310.11511)). The system is orchestrated by LangGraph.

Powered by **Mistral models** (Mistral Large, Mistral Small 3.2, Ministral 3B, Mistral Embed), Cohere (Re-ranker) and Tavily (Web-search).

## RAG Flow

<p>
  <img src="public/RAG-flow.jpg" alt="Rag-workflow"/>
</p>

## Features

- **Streaming Responses** — Real-time token streaming via Server-Sent Events (SSE)
- **Structured Citations** — Click-to-inspect source cards with relevance scores
- **Intelligent Routing** — Automatic classification of query intent
- **Hybrid Search** — Best of semantic understanding + keyword matching
- **Corrective RAG** — Self-healing pipeline with fallback strategies
- **Latency Tracking** — Per-step timing metrics in the RAG console

## Chunk & Embed
<p>
  <img src="public/chunk_embed.png" alt="chunk-embed"/>
</p>


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
