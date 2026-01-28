# Le-Route 🛤️

**Enterprise Document Reader with Intelligent Model Routing**

Route document queries to the optimal model — Ministral 8B for simple queries, Mistral Large for complex ones. Save 60-80% on API costs without sacrificing quality.

---

## 🎯 Project Overview

Le-Route is a grounded document Q&A system that:
1. Ingests documents (paste or upload)
2. Answers questions with **citations** pointing to exact source locations
3. **Intelligently routes** between cheap (8B) and expensive (123B) models
4. Uses an **MLP router** trained on DocBench to make routing decisions

### Demo Value Proposition
> "We don't optimize for intelligence. We optimize for correctness and cost."
> - 80% of queries → Ministral 8B ($0.10/1M tokens)
> - 20% of queries → Mistral Large ($2.00/1M tokens)
> - Result: ~60-80% cost savings with zero quality loss on high-stakes queries

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     React Frontend (Vercel)                      │
│                  Light/Dark Mode + Orange Accent                 │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Doc Input    │  │  Chat UI     │  │  Citation Panel       │  │
│  │ (paste/file) │  │  (Q&A)       │  │  (right side, shows   │  │
│  │              │  │              │  │   highlighted source) │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Railway)                     │
│                                                                  │
│  POST /ingest ──→ Chunk text ──→ Embed (mistral-embed) ──→ Store│
│                                                                  │
│  POST /ask ──────→ Retrieve chunks ──→ MLP Router ──→ Generate  │
│                            │                │                    │
│                            ▼                ▼                    │
│                    ┌───────────────────────────────┐             │
│                    │        MLP Router             │             │
│                    │                               │             │
│                    │  Features:                    │             │
│                    │  - Query embedding (PCA)      │             │
│                    │  - Top chunk similarity       │             │
│                    │  - Num relevant chunks        │             │
│                    │  - Doc type one-hot           │             │
│                    │  - Risk keyword flags         │             │
│                    │                               │             │
│                    │  Output: P(small), P(large)   │             │
│                    └───────────────────────────────┘             │
│                                     │                            │
│                    ┌────────────────┴────────────────┐           │
│                    ▼                                 ▼           │
│           ministral-8b-latest              mistral-large-2501    │
│           (simple queries)                 (complex/risky)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
le-route/
├── backend/
│   ├── main.py              # FastAPI app, endpoints
│   ├── models.py            # Pydantic schemas
│   ├── embeddings.py        # Chunking, mistral-embed, vector store
│   ├── router.py            # Rule-based (fallback) + MLP router
│   ├── prompts.py           # System prompts for grounded Q&A
│   ├── requirements.txt
│   └── Dockerfile           # For Railway deployment
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── DocumentInput.jsx    # Paste/upload document
│   │   │   ├── ChatInterface.jsx    # Q&A with citations
│   │   │   ├── CitationPanel.jsx    # Right panel, highlighted source
│   │   │   ├── RoutingDashboard.jsx # Model, cost, confidence display
│   │   │   └── ThemeToggle.jsx      # Light/dark mode
│   │   ├── hooks/
│   │   │   └── useLeRoute.js        # API hooks
│   │   ├── styles/
│   │   │   └── theme.css            # Mistral-inspired theme
│   │   └── utils/
│   │       └── api.js               # API client
│   ├── package.json
│   └── vercel.json
│
├── data/
│   ├── download_docbench.py         # Download DocBench dataset
│   ├── process_docbench.py          # Process into train/val splits
│   ├── generate_labels.py           # Run both models, generate routing labels
│   └── upload_to_hf.py              # Upload to HuggingFace
│
├── mlp/
│   ├── model.py                     # MLP architecture
│   ├── features.py                  # Feature extraction
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation metrics
│   └── export.py                    # Export for inference
│
├── benchmark/
│   ├── run_benchmark.py             # Run full benchmark
│   ├── metrics.py                   # Accuracy, citation, abstention, cost, latency
│   └── results/                     # Benchmark outputs
│
├── .env.example
├── README.md
└── CURSOR_CONTEXT.md                # Context for Cursor AI
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.11+ |
| Frontend | React 18, Vite, TailwindCSS |
| Models | Mistral API (ministral-8b-latest, mistral-large-2501) |
| Embeddings | mistral-embed |
| Vector Store | In-memory NumPy (MVP), Qdrant (v2) |
| MLP Router | PyTorch |
| Data | DocBench → HuggingFace |
| Deploy | Railway (backend), Vercel (frontend) |

---

## 🎨 Frontend Design Spec

### Theme
- **Light mode**: White background (#FFFFFF), orange (#FF7000) accent, dark text (#1a1a1a)
- **Dark mode**: Near-black (#1a1a1a) background, orange (#FF7000) accent, white text
- **Toggle**: Sun/moon icon in header

### Layout (3 columns when citation selected)
```
┌─────────────────────────────────────────────────────────────────┐
│  Header: Logo | Theme Toggle | Session Info                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │                 │  │                 │  │                 │  │
│  │  Document       │  │  Chat           │  │  Citation       │  │
│  │  Input          │  │  Interface      │  │  Panel          │  │
│  │                 │  │                 │  │  (conditional)  │  │
│  │  - Paste area   │  │  - Messages     │  │                 │  │
│  │  - Upload btn   │  │  - Input        │  │  - Full chunk   │  │
│  │  - Doc type     │  │  - Citations    │  │  - Highlighted  │  │
│  │    selector     │  │    (clickable)  │  │    text         │  │
│  │                 │  │                 │  │  - Context      │  │
│  │                 │  │                 │  │                 │  │
│  └─────────────────┘  └──────────────���──┘  └─────────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Routing Dashboard: Model Used | Confidence | Cost | Latency    │
└─────────────────────────────────────────────────────────────────┘
```

### Citation Panel Behavior
1. User asks question
2. Answer appears with inline citations like `[1]`, `[2]`
3. User clicks `[1]`
4. Right panel slides in showing:
   - The full chunk text
   - The relevant portion **highlighted in orange**
   - Chunk metadata (chunk ID, relevance score)
5. Clicking outside or X closes the panel

### Routing Dashboard (bottom bar)
- **Model**: Badge showing "Ministral 8B" (green) or "Mistral Large" (orange)
- **Confidence**: Progress bar 0-100%
- **Cost**: "$0.0003" with cumulative session total
- **Latency**: "450ms"
- **Risk Level**: Badge "LOW" (green) / "MEDIUM" (yellow) / "HIGH" (red)

---

## 📡 API Endpoints

### POST /ingest
```typescript
// Request
{
  session_id?: string,          // Auto-generated if not provided
  text: string,                 // Document text
  doc_type: "policy" | "contract" | "legal" | "technical" | "general",
  title?: string
}

// Response
{
  session_id: string,
  chunks_created: number,
  total_tokens: number,
  doc_type: string,
  status: "ready"
}
```

### POST /ask
```typescript
// Request
{
  session_id: string,
  question: string
}

// Response
{
  answer: string,                    // With inline [1], [2] citations
  citations: [
    {
      chunk_id: number,
      text: string,                  // Full chunk text
      relevance_score: number,
      highlight_start: number,       // Character offset for highlighting
      highlight_end: number
    }
  ],
  confidence: number,                // 0-1
  risk_level: "low" | "medium" | "high",
  model_used: string,
  routing_reason: string,
  cost_estimate_usd: number,
  abstained: boolean,
  latency_ms: number
}
```

---

## 🧠 MLP Router Spec

### Input Features (25 dimensions)
```python
features = [
    # Query embedding compressed via PCA (16 dims)
    *query_embedding_pca,           # 16 floats
    
    # Retrieval metrics
    top_chunk_similarity,           # 1 float (0-1)
    num_relevant_chunks,            # 1 int (chunks with sim > 0.5)
    
    # Query characteristics
    query_token_count / 100,        # 1 float (normalized)
    
    # Document characteristics  
    doc_token_count / 10000,        # 1 float (normalized)
    
    # Doc type one-hot
    *doc_type_onehot,               # 5 floats (policy, contract, legal, technical, general)
]
# Total: 16 + 1 + 1 + 1 + 1 + 5 = 25 dimensions
```

### Architecture
```python
class RoutingMLP(nn.Module):
    def __init__(self, input_dim=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),  # [P(small), P(large)]
        )
    
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)
```

### Training Labels
From DocBench evaluation:
- Run both `ministral-8b-latest` and `mistral-large-2501` on each example
- **Label = 0 (Small)**: If Small model got it correct
- **Label = 1 (Large)**: If only Large model got it correct, or both failed
- This teaches the router: "use Small when it's sufficient, escalate otherwise"

### Inference
```python
def route(features):
    probs = model(features)
    p_small, p_large = probs[0], probs[1]
    
    # Conservative threshold: use Large if >30% chance it's needed
    if p_large > 0.3:
        return "mistral-large-2501"
    return "ministral-8b-latest"
```

---

## 📊 Benchmark Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Answer Accuracy** | % of answers matching ground truth | >85% |
| **Citation Accuracy** | % of citations pointing to correct chunks | >90% |
| **Abstention Accuracy** | % correct "not found" responses | >80% |
| **Cost Savings** | % reduction vs always using Large | >60% |
| **Avg Latency** | Mean response time per query | <1000ms |
| **Routing Accuracy** | % of optimal routing decisions | >75% |

---

## 🚀 Deployment

### Backend (Railway)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
COPY mlp/mlp_router.pt ./mlp_router.pt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Environment variables on Railway:
- `MISTRAL_API_KEY`
- `HF_TOKEN`
- `ROUTER_TYPE=mlp`
- `MLP_MODEL_PATH=./mlp_router.pt`

### Frontend (Vercel)
```json
{
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist"
}
```

Environment variables on Vercel:
- `VITE_API_URL=https://your-app.railway.app`

---

## 🔑 Environment Variables

```bash
# .env
MISTRAL_API_KEY=your_mistral_api_key
HF_TOKEN=your_huggingface_token

# Backend specific
ROUTER_TYPE=mlp              # "rule" for fallback, "mlp" for trained router
MLP_MODEL_PATH=./mlp_router.pt
LOG_ROUTING=true

# Frontend specific (in frontend/.env)
VITE_API_URL=http://localhost:8000
```

---

## 🏃 Quick Start (Local Dev)

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export MISTRAL_API_KEY=your_key
python main.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev

# Open http://localhost:5173
```

---

## 🎯 Demo Script

1. **Paste a sample policy document**
2. **Simple query**: "What is the expense deadline?" → Ministral 8B, fast
3. **Complex query**: "Can we terminate for social media posts?" → Mistral Large, careful
4. **Click citation** → Show highlighted source in panel
5. **Show dashboard**: "80% of queries went to 8B, 60% cost savings"

---

## 📄 License

MIT