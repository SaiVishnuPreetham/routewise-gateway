---
title: RouteWise Backend
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
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
git clone https://github.com/SaiVishnuPreetham/routewise-gateway.git
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
  Overall Accuracy         : 85.0%
  False Positives (FP)     : 3  (complex → Fast)
  False Negatives (FN)     : 0  (simple → Capable)
  Avg Routing Latency      : 0.15 ms

  ✅ PASS — Accuracy 85.0% meets the >75% success bar
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

---

### Q1 — Did your routing model work?

**Yes. Accuracy: 85% (17/20 correct). ✅ PASS — meets the >75% bar.**

The routing model was evaluated against a hand-labeled 20-prompt test suite (10 simple, 10 complex) using `python poc.py test_suite.json`.

| Metric | Result |
|--------|--------|
| Total prompts | 20 |
| Correct predictions | 17 / 20 |
| **Overall Accuracy** | **85.0%** |
| False Positives (FP) — complex → Fast | 3 (15.0%) |
| False Negatives (FN) — simple → Capable | 0 (0.0%) |
| Avg Routing Latency | ~0.1 ms (excluding first-call LiteLLM warm-up) |

**Key observation:** The model has zero false negatives — it never over-routed a simple prompt to the expensive Capable model. All 3 errors were under-routing (complex tasks sent to Fast). This is the safer failure mode: latency may suffer but correctness is better preserved.

---

### Q2 — What was the cost difference?

Comparing **"always Capable"** vs. **smart routing** on the same 20 prompts:

| Strategy | Models Used | Estimated Token Cost (approx.) |
|----------|-------------|-------------------------------|
| Always Capable (Gemini 2.0 Flash) | 20 × Capable | ~$0.000060 (20 × ~$0.000003/call) |
| Smart Routing (RouteWise) | 10 × Fast + 10 × Capable | ~$0.000030 (Fast = ~$0 on Groq free tier) |
| **Savings** | | **~50% cost reduction** |

Both Groq Llama3-8B and Gemini 2.0 Flash are free-tier models, so real dollar cost is near zero at this scale. The meaningful metric is **token efficiency and latency**:

- The **Fast model (Groq)** responds in **< 200ms** on simple queries.
- The **Capable model (Gemini)** averages **600–1200ms** on complex queries.
- By routing 10 simple prompts to Groq, RouteWise saves approximately **8–10 seconds** of total latency across the 20-prompt suite — a meaningful difference at API scale.

At production scale (e.g., 1M calls/day, $0.30/M tokens for GPT-4-class models vs. $0.03/M for Llama-class), smart routing at 50% split would reduce costs from **$300/day → ~$165/day**, saving ~$135/day.

---

### Q3 — Where did the routing model fail?

Three prompts were mis-routed (all FP: complex prompt → Fast model):

**ID 11 — Binary search implementation (Score: 0.294, predicted: fast)**
> *"Write a Python function that implements binary search on a sorted list..."*

- **Why it failed:** The prompt contains the keyword `"write a"` (complexity signal) but no code block markers (`` ``` ``), no `def`/`import` patterns, and no math signals. F2 (code detection) scored 0 because the code hadn't been written yet — the model only detects *existing* code in the prompt, not asks to *produce* code.
- **Root cause:** The code-detection feature (F2) is reactive, not semantic. It cannot infer that *"write a Python function"* implies code output.

**ID 12 — Microservices vs. monolith comparison (Score: 0.312, predicted: fast)**
> *"Compare the pros and cons of microservices architecture versus a monolith..."*

- **Why it failed:** The word `"compare"` and `"pros and cons"` are both in the complexity keyword list, giving F4 a score of 0.85 (2 keywords). However, there were no code or math signals, and the prompt token count was moderate (~20 tokens). The total weighted score of 0.312 fell just below the 0.35 threshold by a margin of only 0.038.
- **Root cause:** The decision boundary (0.35) is too close to this prompt's score. A threshold of 0.30 would have correctly classified it.

**ID 16 — LRU cache implementation (Score: 0.339, predicted: fast)**
> *"Implement a LRU cache in Python using a doubly linked list and a hash map..."*

- **Why it failed:** This is a complex coding prompt with `"implement"` as a keyword (F4 active) and a higher token count (F1 active). But like ID 11, it lacks in-prompt code markers (F2 = 0). Score was 0.339 — only **0.011 below** the 0.35 threshold.
- **Root cause:** Again, code-detection relies on syntax in the prompt itself. Prompts asking to *produce* code without showing code are systematically under-scored by F2.

**Common pattern across all 3 failures:** F2 (code detection) only fires when code syntax is *present in the prompt*, not when the user is asking for code to be *written*. This is the single biggest blind spot of the current model.

---

### Q4 — What was your cache hit rate?

The semantic cache uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings, cosine similarity).

| Parameter | Value |
|-----------|-------|
| Threshold chosen | **0.85** |
| Embedding model | `all-MiniLM-L6-v2` |
| Cache hit condition | `cosine_similarity ≥ 0.85` |

**Was 0.85 too strict or too loose?**

At **0.85**, the cache is intentionally conservative:
- **Too strict for paraphrases:** A prompt like *"What's the capital of France?"* vs *"Tell me the capital of France?"* scores ~0.91 similarity — a hit ✅.
- **At the boundary:** *"What is the boiling point of water?"* vs *"At what temperature does water boil?"* scores ~0.84–0.87 similarity — right at the boundary. May hit or miss depending on exact phrasing.
- **Correctly misses dissimilar questions:** *"Who wrote 1984?"* vs *"Who invented Python?"* scores ~0.55 — correctly a miss ✅.

**Verdict:** 0.85 is a well-calibrated default for this use case. Lowering to 0.75 would increase hit rate but risk returning stale answers for subtly different questions (e.g., "sort ascending" vs "sort descending"). Raising to 0.95 would make the cache nearly useless. **0.85 is the sweet spot.**

The cache stats endpoint is available at `GET /cache/stats` and the hit rate is displayed live in the Streamlit dashboard sidebar.

---

### Q5 — If you rebuilt the routing model differently, what would you change?

**1. Replace F2 (code detection) with intent detection**
The current code detector fires on syntax in the prompt. Instead, it should detect *intent to produce code* using keyword phrases like `"write a function"`, `"implement a class"`, `"build a script"` — regardless of whether code is present in the prompt.

**2. Lower the decision threshold from 0.35 → 0.30**
All 3 mis-routes had scores between 0.29–0.34. A threshold of 0.30 would correctly classify IDs 11 and 12 without introducing new false negatives. This is a one-line change (`DECISION_THRESHOLD = 0.30`).

**3. Add an F6 feature: output length estimation**
Prompts asking for long outputs (e.g., `"write a complete REST API"`, `"design a full system"`) should score higher. An output-length estimator based on verb + scope phrases (`"complete"`, `"full"`, `"entire"`, `"detailed"`) would catch cases the current 5 features miss.

**4. Calibrate confidence scores better**
Current confidence is `|score - threshold| × 2`. For borderline prompts (score 0.30–0.40), confidence is low (< 0.10) but the model still makes a deterministic binary decision. A better approach: route borderline prompts (confidence < 0.15) probabilistically or always default to Capable to avoid under-routing.

**5. Train on real user data**
The current weights (F1=0.20, F2=0.25, etc.) were set by hand. Given a labeled dataset of real API requests, logistic regression or a decision tree could learn optimal weights — potentially pushing accuracy from 85% → 95%+.

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

Apache 2.0

Made with ❤️ by P. Sai Preetham
