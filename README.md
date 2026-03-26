---
title: RouteWise Backend
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# 🔀 RouteWise — AI Gateway

A research-grade AI gateway that routes every prompt to the right LLM — and proves it made the right call.

> **PS 2 — AI Gateway** | Built for SISA AI Research Internship

---

## What Is This?

RouteWise is a smart traffic controller that sits between a user and two LLMs. Instead of always using the most powerful (and expensive) model, the gateway reads each prompt, estimates how hard the task is, and picks the best model for the job — balancing **cost**, **speed**, and **quality**.

## The Two Models

| Label | Model | Provider | Use Case |
|-------|-------|----------|----------|
| **Fast model** | `groq/llama-3.1-8b-instant` | [Groq](https://groq.com) (free tier) | Simple queries, factual Q&A, short summaries, low-complexity tasks |
| **Capable model** | `gemini/gemini-2.0-flash` | [Google AI Studio](https://aistudio.google.com) (free tier) | Reasoning, code generation, multi-step tasks, complex analysis |

**Why these two?**
- **Groq Llama3-8B** delivers <100ms latency for simple tasks — the fastest free inference available. No credit card needed.
- **Gemini 2.0 Flash** handles complex reasoning and code generation with high quality at zero cost. No credit card needed.
- Both are accessed through **LiteLLM** (`litellm.completion()`), giving us a unified interface, automatic cost tracking, and consistent OpenAI-format responses.

---

## Architecture — 4 Pieces

```
User  →  POST /chat  →  ┌─────────────────┐
                         │  Semantic Cache  │──→ HIT → return cached response
                         └────────┬────────┘
                                  │ MISS
                         ┌────────▼────────┐
                         │  Routing Model  │──→ classify prompt (< 5ms)
                         └────────┬────────┘
                            ┌─────┴─────┐
                            │           │
                       ┌────▼───┐  ┌────▼────┐
                       │  Fast  │  │ Capable │
                       │ (Groq) │  │(Gemini) │
                       └────┬───┘  └────┬────┘
                            └─────┬─────┘
                         ┌────────▼────────┐
                         │    Response +   │
                         │    Metadata     │──→ logged by GatewayLogger
                         └─────────────────┘
```

| Piece | File | Description |
|-------|------|-------------|
| **1. Gateway Server** | `gateway/server.py` | FastAPI app with `POST /chat`. Uses `litellm.completion()` and `litellm.completion_cost()` for unified LLM access and cost tracking. |
| **2. Routing Model** | `gateway/routing_model.py` | Rule-based scoring classifier with 5 features, explicit weights, and a decision boundary. |
| **3. Semantic Cache** | `gateway/cache.py` | Custom cache using `sentence-transformers` (all-MiniLM-L6-v2) with cosine similarity. Threshold configurable via `CACHE_THRESHOLD`. |
| **4. Log Viewer** | `dashboard/app.py` | Streamlit app showing the request log: timestamp, prompt, model, reason, latency, cost, cache status. |

---

## Routing Model — How It Works

The routing model is a **weighted rule-based scoring classifier** (not an if/else function). It extracts 5 features from each prompt and computes a weighted score.

### Features & Weights

| ID | Feature | Weight | What It Detects |
|----|---------|--------|-----------------|
| F1 | Token count | 0.20 | Longer prompts → more likely complex |
| F2 | Code detection | 0.25 | Regex for \`\`\`, `def`, `import`, `class` → code tasks need the Capable model |
| F3 | Math/reasoning | 0.20 | Regex for operators, `equation`, `formula`, `prove`, `calculate` |
| F4 | Complexity keywords | 0.25 | Phrases like "explain why", "compare", "analyze", "debug", "step by step" |
| F5 | Multi-question | 0.10 | Count of `?` and `.` → compound tasks |

### Decision Boundary

```
score = Σ(weight_i × normalized_feature_i)    for i in {F1..F5}

if score ≥ 0.35  →  route to Capable model
if score < 0.35  →  route to Fast model

confidence = |score - 0.35| × 2    (0.0 = uncertain, 1.0 = certain)
```

### Latency

The routing decision adds **< 5ms** overhead per request (measured in PoC).

---

## Setup (5 commands)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/routewise-gateway.git
cd routewise-gateway

# 2. Create virtual environment
python -m venv venv && venv\Scripts\activate    # Windows
# python -m venv venv && source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
copy .env.example .env
# Edit .env with your free-tier API keys from groq.com and aistudio.google.com
# Required keys: GROQ_API_KEY, GEMINI_API_KEY

# 5. Start the gateway
python -m uvicorn gateway.server:app --reload --port 8000
```

---

## Usage

### Send a request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "model_used": "groq/llama-3.1-8b-instant",
  "model_label": "Fast (Groq Llama3-8B)",
  "routing_reason": "Prompt appears straightforward → fast model",
  "confidence_score": 0.82,
  "raw_score": 0.09,
  "latency_ms": 245.3,
  "cost_usd": 0.0000012,
  "cache_hit": false,
  "cache_similarity": 0.0
}
```

### Start the log viewer

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 to see the live log table.

---

## PoC — Standalone Routing Model Evaluator

The PoC runs the routing model on the 20-prompt test suite **without any API keys or running server**.

```bash
python poc.py                     # uses default test_suite.json
python poc.py test_suite.json     # explicit file path
python poc.py my_prompts.csv      # CSV with columns: prompt, ground_truth
```

**Output:**
```
==================================================
  RouteWise PoC — Routing Model Evaluation
==================================================

  ID  Correct       GT  Predicted   Conf  Score  Latency  Prompt
------------------------------------------------------------
   1        ✓   simple       fast   0.82  0.090    0.1ms  What is the capital of France?
   2        ✓   simple       fast   0.92  0.040    0.1ms  Hi there! How are you today?
  ...
  20        ✓  complex    capable   0.74  0.870    0.2ms  Calculate the probability of...

==================================================
  SUMMARY
==================================================
  Overall Accuracy         : 95.0%
  False Positives (FP)     : 0  (complex → Fast)
  False Negatives (FN)     : 1  (simple → Capable)
  Avg Routing Latency      : 0.15 ms

  ✅ PASS — Accuracy 95.0% meets the >75% success bar
```

---

## Project Structure

```
routewise-gateway/
├── gateway/
│   ├── __init__.py           # Package init
│   ├── server.py             # FastAPI + litellm.completion() + GatewayLogger
│   ├── routing_model.py      # Standalone rule-based classifier (5 features)
│   └── cache.py              # Semantic cache (sentence-transformers)
├── dashboard/
│   └── app.py                # Streamlit log viewer
├── poc.py                    # Standalone routing model evaluator
├── test_suite.json           # 20 labeled prompts (10 simple, 10 complex)
├── .env                      # API keys (copy from .env.example template)
├── .env.example              # API key template (copy to .env and fill in)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Research Questions (Addressed)

| # | Question | Where Answered |
|---|----------|---------------|
| 1 | Did the routing model work? | PoC output — accuracy, FP, FN on 20 prompts |
| 2 | Cost difference? | Compare "always Capable" vs. smart routing (PPT slide 6) |
| 3 | Where did routing fail? | PoC failure analysis section shows mis-routes with root cause |
| 4 | Cache hit rate? | `/cache/stats` endpoint + dashboard sidebar |
| 5 | What would you change? | PPT slide 9 |

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API Server | FastAPI | Async, fast, auto-docs at `/docs` |
| LLM Layer | LiteLLM | Single `completion()` call for both Groq & Gemini, built-in cost tracking |
| Routing | Custom Python (rule-based model) | < 5ms latency, no ML inference overhead, explainable |
| Cache | sentence-transformers + in-memory dict | Semantic similarity, 384-dim embeddings, configurable threshold |
| Dashboard | Streamlit | Minimal log viewer — judges can inspect decisions |
| Cost Tracking | `litellm.completion_cost()` | Reads from LiteLLM's model pricing database automatically |

---

## License

MIT
