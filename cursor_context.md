# CURSOR_CONTEXT.md

## Context for Cursor AI (Claude Opus 4)

This file contains detailed context and prompts for building Le-Route in Cursor.

---

## 🎯 Project Summary

**Le-Route** is an enterprise document Q&A system with intelligent model routing:
- Users paste/upload documents
- Ask questions, get answers with clickable citations
- System routes queries to cheap (Ministral 8B) or expensive (Mistral Large) based on difficulty
- MLP router trained on DocBench makes routing decisions

**Key differentiator**: Citation panel that shows highlighted source text when user clicks a citation.

---

## 📋 Implementation Order

Build in this order for fastest path to working demo:

### 1. Backend Core (Existing - needs updates)
Files exist: `main.py`, `models.py`, `embeddings.py`, `router.py`, `prompts.py`

**Updates needed:**
- Add `highlight_start`, `highlight_end` to ChunkInfo in `models.py`
- Update `/ask` endpoint to compute highlight positions
- Add MLP router class to `router.py`
- Add PCA compression to `embeddings.py`

### 2. Data Pipeline (New)
Create `data/` directory with:
- `download_docbench.py`
- `process_docbench.py`
- `generate_labels.py`
- `upload_to_hf.py`

### 3. MLP Router (New)
Create `mlp/` directory with:
- `model.py`
- `features.py`
- `train.py`
- `evaluate.py`

### 4. Benchmark (New)
Create `benchmark/` directory with:
- `metrics.py`
- `run_benchmark.py`

### 5. Frontend (New)
Create full React app in `frontend/`

### 6. Deployment Configs
- `backend/Dockerfile`
- `frontend/vercel.json`

---

## 🔧 Detailed Implementation Prompts

Copy these prompts into Cursor to build each component:

---

### PROMPT 1: Update Backend Models

```
Update backend/models.py to add highlight positions for citations.

Add to ChunkInfo class:
- highlight_start: Optional[int] - character offset where relevant text starts
- highlight_end: Optional[int] - character offset where relevant text ends

These will be used by the frontend to highlight the specific part of the chunk that answers the question.
```

---

### PROMPT 2: Data Pipeline - Download DocBench

```
Create data/download_docbench.py that:

1. Downloads DocBench dataset from HuggingFace (Anni-Zou/DocBench)
2. Saves raw data to data/raw/docbench/
3. Prints dataset statistics (num examples, avg doc length, etc.)

Use the datasets library from HuggingFace.
Include error handling and progress bars.
```

---

### PROMPT 3: Data Pipeline - Process DocBench

```
Create data/process_docbench.py that:

1. Loads raw DocBench data from data/raw/docbench/
2. Processes each example into this format:
   {
     "id": str,
     "document": str,           # Full document text
     "question": str,
     "answer": str,             # Ground truth answer
     "doc_type": str,           # Inferred from content (policy/contract/legal/technical/general)
     "chunks": [str],           # Document split into chunks
     "relevant_chunk_ids": [int] # Which chunks contain the answer
   }
3. Splits into train (80%) and val (20%)
4. Saves to data/processed/train.jsonl and data/processed/val.jsonl

For chunking, use 500 token chunks with 100 token overlap.
For doc_type inference, use keyword matching (legal terms → legal, etc.)
```

---

### PROMPT 4: Data Pipeline - Generate Labels

```
Create data/generate_labels.py that:

1. Loads processed data from data/processed/
2. For each example:
   a. Embeds chunks using mistral-embed
   b. Retrieves top-5 chunks for the question
   c. Runs ministral-8b-latest to generate answer
   d. Runs mistral-large-2501 to generate answer
   e. Evaluates both answers against ground truth
   f. Assigns label:
      - 0 (use Small) if Small got it correct
      - 1 (use Large) if only Large got it correct or both failed

3. Saves labeled data to data/labeled/train.jsonl and data/labeled/val.jsonl

Schema for labeled data:
{
  "id": str,
  "question": str,
  "query_embedding": [float],      # For MLP training
  "top_similarities": [float],     # Top-5 chunk similarities
  "num_relevant_chunks": int,      # Chunks with sim > 0.5
  "doc_token_count": int,
  "doc_type": str,
  "small_answer": str,
  "large_answer": str,
  "ground_truth": str,
  "small_correct": bool,
  "large_correct": bool,
  "label": int                     # 0=small, 1=large
}

Use batching to avoid rate limits. Save progress periodically.
Estimate: This will make ~2x API calls per example (both models).
```

---

### PROMPT 5: Data Pipeline - Upload to HuggingFace

```
Create data/upload_to_hf.py that:

1. Loads labeled data from data/labeled/
2. Creates a HuggingFace dataset with train/val splits
3. Uploads to huyxdang/le-route-docbench-data

Include dataset card with:
- Description of Le-Route project
- Data format documentation
- Label generation methodology
- Usage examples

Use HF_TOKEN from environment variable.
```

---

### PROMPT 6: MLP Router - Model

```
Create mlp/model.py with:

1. RoutingMLP class (PyTorch nn.Module):
   - Input: 25 dimensions
   - Hidden: 64 → 32 with ReLU and Dropout(0.1)
   - Output: 2 (softmax probabilities for small/large)

2. Helper functions:
   - save_model(model, path)
   - load_model(path) -> RoutingMLP

Architecture:
Linear(25, 64) → ReLU → Dropout(0.1) → Linear(64, 32) → ReLU → Dropout(0.1) → Linear(32, 2) → Softmax
```

---

### PROMPT 7: MLP Router - Features

```
Create mlp/features.py with:

1. PCA class wrapper that:
   - Fits on training query embeddings
   - Transforms 1024-dim mistral-embed to 16-dim
   - Saves/loads fitted PCA

2. extract_features() function:
   Input:
   - query_embedding: list[float] (1024 dim from mistral-embed)
   - top_similarities: list[float] (top-k chunk similarities)
   - query_token_count: int
   - doc_token_count: int
   - doc_type: str
   
   Output: numpy array of 25 features:
   - query_embedding_pca (16)
   - top_chunk_similarity (1)
   - num_relevant_chunks (1) - count where sim > 0.5
   - query_token_count / 100 (1)
   - doc_token_count / 10000 (1)
   - doc_type_onehot (5) - [policy, contract, legal, technical, general]

3. FeatureExtractor class that holds fitted PCA and provides extract() method
```

---

### PROMPT 8: MLP Router - Training

```
Create mlp/train.py that:

1. Loads labeled data from data/labeled/train.jsonl
2. Fits PCA on query embeddings
3. Extracts features for all examples
4. Creates PyTorch DataLoader
5. Trains RoutingMLP with:
   - CrossEntropyLoss
   - Adam optimizer, lr=1e-3
   - 50 epochs
   - Batch size 32
   - Early stopping on val loss (patience=5)
6. Saves:
   - Trained model to mlp/mlp_router.pt
   - Fitted PCA to mlp/pca.pkl
   - Training metrics to mlp/training_log.json

Print training progress with loss and accuracy per epoch.
```

---

### PROMPT 9: MLP Router - Evaluation

```
Create mlp/evaluate.py that:

1. Loads trained model and PCA
2. Loads val data from data/labeled/val.jsonl
3. Computes metrics:
   - Routing accuracy: % of correct routing decisions
   - Precision/recall for each class
   - Confusion matrix
   - Cost savings: estimated % saved vs always using Large
   - Accuracy preservation: % of queries answered correctly with routing

4. Prints formatted results table
5. Saves results to mlp/eval_results.json
```

---

### PROMPT 10: Benchmark - Metrics

```
Create benchmark/metrics.py with functions:

1. answer_accuracy(predicted: str, ground_truth: str) -> float
   - Normalize both strings
   - Check for exact match or high overlap
   - Return 0.0-1.0 score

2. citation_accuracy(citations: list, relevant_chunk_ids: list) -> float
   - Check if cited chunks match ground truth relevant chunks
   - Return precision score

3. abstention_accuracy(abstained: bool, answer_in_doc: bool) -> float
   - Return 1.0 if abstention matches whether answer exists
   - Return 0.0 otherwise

4. cost_estimate(model: str, input_tokens: int, output_tokens: int) -> float
   - Return USD cost based on Mistral pricing

5. BenchmarkResults dataclass to hold all metrics
```

---

### PROMPT 11: Benchmark - Run

```
Create benchmark/run_benchmark.py that:

1. Loads val data
2. For each example:
   a. Simulate the full pipeline (embed, retrieve, route, generate)
   b. Collect metrics: accuracy, citations, abstention, cost, latency
3. Aggregate results:
   - Mean accuracy
   - Mean citation accuracy
   - Mean abstention accuracy  
   - Total cost vs baseline (always Large)
   - Mean latency
   - Routing distribution (% to each model)

4. Generate report:
   - Print summary table
   - Save detailed results to benchmark/results/run_{timestamp}.json
   - Save CSV for easy analysis

Include progress bar and ETA.
```

---

### PROMPT 12: Frontend Setup

```
Create React frontend in frontend/ directory:

1. Initialize with Vite + React + TypeScript
2. Install dependencies:
   - tailwindcss
   - @headlessui/react (for transitions)
   - lucide-react (for icons)
   - axios

3. Configure Tailwind with custom theme:
   colors: {
     mistral: {
       orange: '#FF7000',
       'orange-light': '#FF8C33',
       dark: '#1a1a1a',
       darker: '#0f0f0f',
     }
   }

4. Set up folder structure:
   src/
     components/
     hooks/
     utils/
     styles/
     App.tsx
     main.tsx
```

---

### PROMPT 13: Frontend - Theme System

```
Create frontend theme system with light/dark mode:

1. src/hooks/useTheme.ts:
   - Persist preference to localStorage
   - Detect system preference
   - Toggle function

2. src/components/ThemeToggle.tsx:
   - Sun/Moon icon button
   - Smooth transition animation

3. Update tailwind.config.js for dark mode class strategy

4. Create base styles in src/styles/globals.css:
   - Light mode: white bg, dark text, orange accent
   - Dark mode: #1a1a1a bg, white text, orange accent
   - Smooth transitions between modes
```

---

### PROMPT 14: Frontend - Document Input

```
Create src/components/DocumentInput.tsx:

Features:
1. Large textarea for pasting document text
2. File upload button (accepts .txt, .md, .pdf)
3. Document type selector dropdown:
   - Policy
   - Contract
   - Legal
   - Technical
   - General
4. "Load Document" button
5. Show loading state while ingesting
6. Show success state with chunk count

Styling:
- Clean, minimal design
- Orange accent on focus
- Responsive (full width on mobile)
```

---

### PROMPT 15: Frontend - Chat Interface

```
Create src/components/ChatInterface.tsx:

Features:
1. Message list showing Q&A history
2. User messages aligned right (gray bg)
3. Assistant messages aligned left (white/dark bg)
4. Citations in answers rendered as clickable badges [1], [2]
5. Input field at bottom with send button
6. Loading indicator while waiting for response

Citation badges:
- Orange background
- Hover effect (slightly darker)
- onClick: emit event with citation data

Props:
- messages: array of {role, content, citations?}
- onSend: (question: string) => void
- onCitationClick: (citation: Citation) => void
- isLoading: boolean
```

---

### PROMPT 16: Frontend - Citation Panel

```
Create src/components/CitationPanel.tsx:

Features:
1. Sliding panel from right side
2. Shows when a citation is clicked
3. Content:
   - Header: "Source [Chunk {id}]" with X close button
   - Relevance score badge
   - Full chunk text with highlighted portion
4. Highlight the relevant text in orange background
5. Click outside or X to close
6. Smooth slide animation

Props:
- citation: { chunk_id, text, relevance_score, highlight_start, highlight_end } | null
- onClose: () => void

Use @headlessui/react Transition for animation.
```

---

### PROMPT 17: Frontend - Routing Dashboard

```
Create src/components/RoutingDashboard.tsx:

Fixed bottom bar showing:
1. Model badge:
   - "Ministral 8B" with green bg if small model
   - "Mistral Large" with orange bg if large model
2. Confidence meter:
   - Progress bar 0-100%
   - Color: green >70%, yellow 40-70%, red <40%
3. Cost display:
   - Current query cost: "$0.0003"
   - Session total: "Total: $0.0045"
4. Latency: "450ms"
5. Risk level badge:
   - "LOW" green
   - "MEDIUM" yellow  
   - "HIGH" red

Props:
- routingInfo: {
    model_used: string,
    confidence: number,
    cost_estimate_usd: number,
    latency_ms: number,
    risk_level: string
  } | null
- sessionTotalCost: number
```

---

### PROMPT 18: Frontend - API Client

```
Create src/utils/api.ts:

1. Configure axios instance with base URL from env
2. Types for all request/response schemas
3. Functions:
   - ingestDocument(text, docType, title?) -> IngestResponse
   - askQuestion(sessionId, question) -> AskResponse
   - getSession(sessionId) -> SessionInfo
   - deleteSession(sessionId) -> void

4. Error handling with typed errors
5. Request/response interceptors for logging

Use VITE_API_URL environment variable.
```

---

### PROMPT 19: Frontend - Main App

```
Create src/App.tsx that brings everything together:

Layout:
1. Header with logo "Le-Route" and theme toggle
2. Main content area (3-column when citation open):
   - Left: DocumentInput (collapsible after doc loaded)
   - Center: ChatInterface
   - Right: CitationPanel (conditional)
3. Bottom: RoutingDashboard

State management:
- sessionId: string | null
- messages: Message[]
- selectedCitation: Citation | null
- routingInfo: RoutingInfo | null
- sessionTotalCost: number
- isDocumentLoaded: boolean

Flow:
1. User pastes document → call /ingest → set sessionId
2. User asks question → call /ask → append to messages
3. User clicks citation → open CitationPanel
4. Dashboard updates after each query
```

---

### PROMPT 20: Backend - Dockerfile

```
Create backend/Dockerfile for Railway deployment:

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy MLP model if exists
COPY mlp/mlp_router.pt ./mlp_router.pt 2>/dev/null || true
COPY mlp/pca.pkl ./pca.pkl 2>/dev/null || true

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### PROMPT 21: Frontend - Vercel Config

```
Create frontend/vercel.json:

{
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}

Also create frontend/.env.example:
VITE_API_URL=http://localhost:8000
```

---

## 🧪 Testing Prompts

### Test Backend Locally
```
I want to test the backend locally. Help me:
1. Create a test script that ingests a sample document
2. Asks 3 different questions (simple, medium, complex)
3. Prints the routing decisions and answers
4. Verifies citations are returned correctly
```

### Test Frontend Locally
```
Help me test the frontend:
1. Mock the API responses
2. Test the citation panel highlighting
3. Test theme toggle persistence
4. Test responsive layout
```

---

## 🐛 Common Issues & Fixes

### Issue: CORS errors
```
Add to backend/main.py if not present:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Rate limits on Mistral API
```
Add retry logic with exponential backoff in generate_labels.py.
Use batch processing with delays between batches.
```

### Issue: PCA not fitted
```
Ensure PCA is fitted on training data before inference.
Check that pca.pkl exists and is loaded in router.py.
```

---

## 📊 Expected Results

After full implementation:
- **Routing Accuracy**: ~75-80%
- **Cost Savings**: ~60-70% vs always using Large
- **Answer Accuracy**: ~85% (same as using Large)
- **Latency**: ~500ms for Small, ~1500ms for Large

---

## 🎯 Demo Checklist

Before demo:
- [ ] Backend deployed on Railway
- [ ] Frontend deployed on Vercel
- [ ] MLP model trained and loaded
- [ ] Test with sample policy document
- [ ] Verify citation highlighting works
- [ ] Check dark/light mode toggle
- [ ] Prepare 5 demo queries (2 simple, 2 medium, 1 complex)