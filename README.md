<p align="center">
  <img src="public/Paper-Rag-logo.png" alt="PaperRAG Logo" width="280"/>
</p>

<p align="center">
  <strong>Agentic RAG for exploring 6,000 NeurIPS 2025 papers — built on Mistral models end-to-end</strong>
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
  <a href="#why-this-exists">Why This Exists</a> •
  <a href="#mistral-core">Mistral-Core</a> •
  <a href="#agentic-rag-workflow">Agentic RAG Workflow</a> •
  <a href="#chunk--embed-pipeline">Chunk & Embed</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a>
</p>

---

## Why This Exists

Most RAG demos retrieve documents and dump them into a prompt. PaperRAG goes further — it **routes**, **retrieves**, **grades**, **generates**, and **self-corrects** in a multi-step agentic loop, falling back to web search when the corpus isn't enough. The result is a system that knows when it doesn't know.

The entire intelligence layer runs on **Mistral models**, from lightweight routing (Ministral 3B) to generation and grading (Mistral Large), with Mistral Embed powering the vector space. This is a deliberate design choice: Mistral's model family covers the full spectrum from fast-and-cheap to powerful-and-precise, making it possible to assign the right model to each pipeline stage without leaving the ecosystem.

## Mistral-Core

Every LLM call in the pipeline is a Mistral model, selected by capability-to-cost ratio:

| Pipeline Stage | Model | Why |
|---|---|---|
| **Query rewriting + routing** | Mistral Small 3.2 | Rewrites queries with conversation context and routes in a single call — balances speed with contextual understanding |
| **Document grading** | Mistral Large | Evaluating relevance requires nuanced comprehension of both query intent and document content |
| **Answer generation** | Mistral Large | Core generation — needs to synthesize across multiple sources with inline citations |
| **Hallucination check** | Mistral Small 3.2 | Structured yes/no grading against source text — reliable at this model size |
| **Embeddings** | Mistral Embed | 1024-dim vectors for semantic search across 173,990 chunks |

This tiered approach keeps latency low on the hot path while reserving heavyweight models for the steps that matter most.

## Agentic RAG Workflow

The system implements a LangGraph state machine combining three RAG research papers:
- [**Adaptive RAG**](https://arxiv.org/pdf/2403.14403) — route queries to the right retrieval strategy
- [**Corrective RAG**](https://arxiv.org/pdf/2401.15884) — grade retrieved documents and fall back to web search when they're insufficient
- [**Self-RAG**](https://arxiv.org/pdf/2310.11511) — verify generations against source material and check for hallucinations

<p>
  <img src="public/RAG-flow.jpg" alt="Agentic RAG workflow"/>
</p>

**Pipeline in detail:**

```
User query
  → Rewrite + route in one step (Mistral Small)
  → Hybrid retrieval: BM25 + dense vectors → top 50 (Pinecone + Mistral Embed)
  → Rerank: 50 → 10 (Cohere Rerank v3.5)
  → Grade each document for relevance (Mistral Large)
  → If insufficient docs → web search fallback (Tavily)
  → Generate answer with inline citations (Mistral Large)
  → Verify: grounded in sources? answers the question? (Mistral Small)
  → Stream response + structured citations to frontend
```

## Chunk & Embed Pipeline

~6,000 NeurIPS 2025 papers → 173,990 chunks indexed in Pinecone with hybrid sparse-dense search.

<p>
  <img src="public/chunk_embed.png" alt="Chunking and embedding pipeline"/>
</p>

## Features

**Process Inspector** — a real-time console showing every pipeline step with latency metrics. Users see exactly what the system is doing: routing decisions, retrieval counts, grading outcomes, generation timing.

**Structured Citations** — inline `[1]`, `[2]` references that expand to show the exact claim, the source quote, document metadata, and a relevance score. Every answer is traceable back to its sources.

**Self-Correcting Generation** — answers are checked for hallucination (is the response grounded in the retrieved documents?) and usefulness (does it actually answer the question?). Failed checks trigger re-generation with a retry limit.

**Session Persistence** — conversation history stored in Redis, included in the query rewriting step so follow-up questions resolve correctly ("tell me more about that paper" works).

## Architecture

### Backend (Python / FastAPI)

```
crag/
├── api.py              # FastAPI endpoints — /chat/stream (SSE), sessions CRUD
├── graph.py            # LangGraph state machine — the agentic RAG loop
├── graph_state.py      # TypedDict defining the graph state schema
├── streaming.py        # SSE event pipeline with per-step latency tracking
├── graders.py          # Pydantic-structured LLM calls for routing, grading, rewriting
├── retrieval.py        # Pinecone hybrid search (sparse BM25 + dense embeddings)
├── citations.py        # Citation extraction from generated text
├── session.py          # Redis / in-memory session management
├── config.py           # All model assignments, system prompts, pipeline params
└── main.py             # CLI for local testing
```

### Frontend (Next.js / TypeScript)

```
frontend/src/
├── app/                # Next.js App Router — layout, page, global styles
├── components/
│   ├── chat/           # ChatContainer, MessageBubble, ChatInput, StreamingStatus
│   ├── Header.tsx      # Logo, paper count, new chat
│   ├── Inspector.tsx   # Citation detail panel (claim, quote, source, score)
│   └── RAGConsole.tsx  # Pipeline log sidebar — retro terminal aesthetic
├── hooks/
│   └── useChat.ts      # SSE connection, event parsing, state management
├── lib/
│   └── api.ts          # Backend API client
└── types/
    └── chat.ts         # TypeScript interfaces
```

### Tech Stack

| Layer | Technology | Role |
|---|---|---|
| LLMs | Mistral Large, Mistral Small 3.2 | Generation, grading, routing |
| Embeddings | Mistral Embed | 1024-dim semantic vectors |
| Orchestration | LangGraph | Agentic state machine with conditional edges |
| Vector DB | Pinecone | Hybrid sparse-dense search |
| Reranking | Cohere Rerank v3.5 | Cross-encoder precision reranking |
| Web Search | Tavily | Fallback for out-of-corpus queries |
| Backend | FastAPI + SSE | Streaming API server |
| Frontend | Next.js 14, React 19, Tailwind, shadcn/ui | Chat UI with inspector |
| Sessions | Redis | Conversation history persistence |
| Deployment | Railway (backend) + Vercel (frontend) | Production hosting |

## License

MIT License — see [LICENSE](LICENSE) for details.
