"""
RouteWise — Routing Model
=========================
A rule-based scoring classifier that decides whether a prompt should be
handled by the **Fast model** (simple tasks) or the **Capable model**
(complex tasks).

Model Architecture
------------------
Type        : Weighted rule-based scoring model
Inputs      : Raw text prompt
Output      : RoutingDecision(label, confidence, reason)
Features    : 5 documented features (F1–F5)
Boundary    : Weighted sum >= 0.35 -> Capable model, else Fast model

Feature Definitions & Weights
-----------------------------
| ID | Feature                  | Weight | Rationale                                   |
|----|--------------------------|--------|---------------------------------------------|
| F1 | Token count              | 0.20   | Longer prompts often need deeper reasoning  |
| F2 | Code detection           | 0.25   | Code tasks need stronger generation ability |
| F3 | Math / reasoning detect  | 0.20   | Math & proofs require step-by-step logic    |
| F4 | Complexity keywords      | 0.25   | Explicit analytical intent signals          |
| F5 | Multi-question detect    | 0.10   | Multiple questions imply compound tasks     |

Decision Boundary
-----------------
    score = Σ(weight_i × feature_i)     for i in {F1..F5}
    if score >= 0.35  ->  route to Capable model
    else              ->  route to Fast model
    confidence = |score - 0.35| * 2      (0.0 = uncertain, 1.0 = certain)

This module has ZERO runtime dependency on the gateway server or LiteLLM.
It can optionally import litellm.token_counter for F1, with a pure-Python
fallback.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAST_LABEL = "fast"
CAPABLE_LABEL = "capable"

# Feature weights — each sums to give a score in [0, 1]
WEIGHTS = {
    "F1_token_count": 0.20,
    "F2_code_detect": 0.25,
    "F3_math_reason": 0.20,
    "F4_complexity_kw": 0.25,
    "F5_multi_question": 0.10,
}

DECISION_THRESHOLD = 0.35

# F2 — code detection patterns
CODE_PATTERNS = [
    re.compile(r"```"),                       # fenced code blocks
    re.compile(r"\bdef\s+\w+\s*\("),          # Python function defs
    re.compile(r"\bimport\s+\w+"),            # import statements
    re.compile(r"\bclass\s+\w+"),             # class definitions
    re.compile(r"\bfunction\s+\w+\s*\("),     # JS function defs
    re.compile(r"console\.log|System\.out"),   # common print calls
    re.compile(r"for\s*\(.*;.*;.*\)"),         # C-style for loops
    re.compile(r"#include\s*<"),              # C/C++ includes
]

# F3 — math / reasoning patterns
MATH_PATTERNS = [
    re.compile(r"[+\-*/^=]{2,}"),             # chained operators
    re.compile(r"\b(equation|formula|prove|theorem|integral|derivative)\b", re.I),
    re.compile(r"\b(calculate|compute|solve)\b", re.I),
    re.compile(r"\b(probability|statistics|matrix|vector)\b", re.I),
    re.compile(r"\d+\s*[+\-*/^]\s*\d+"),      # arithmetic expressions
    re.compile(r"\\frac|\\sum|\\int"),         # LaTeX math
]

# F4 — complexity keywords (phrases implying analytical work)
COMPLEXITY_KEYWORDS = [
    "explain why",
    "explain",
    "compare",
    "analyze",
    "debug",
    "step by step",
    "difference between",
    "implement",
    "refactor",
    "design",
    "architect",
    "optimize",
    "trade-off",
    "pros and cons",
    "evaluate",
    "critique",
    "write a",
    "how does",
    "how do",
    "how would",
    "what are the",
    "why does",
    "why is",
    "provide",
    "include",
    "show",
    "prove",
    "calculate",
    "complexity",
]

# F1 — token count thresholds for normalization
TOKEN_SHORT = 15    # prompts <= 15 tokens -> score 0.0
TOKEN_LONG = 60     # prompts >= 60 tokens -> score 1.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """Raw feature values extracted from a prompt."""
    token_count: int = 0
    code_signals: int = 0
    math_signals: int = 0
    complexity_keyword_count: int = 0
    question_marks: int = 0
    sentence_endings: int = 0


@dataclass
class RoutingDecision:
    """The output of the routing model."""
    label: str                    # "fast" or "capable"
    confidence: float             # 0.0–1.0
    reason: str                   # human-readable explanation
    raw_score: float = 0.0       # weighted sum before thresholding
    features: dict = field(default_factory=dict)  # normalized feature values
    latency_ms: float = 0.0      # time to compute the decision


# ---------------------------------------------------------------------------
# Token counting helper (standalone fallback)
# ---------------------------------------------------------------------------

def _count_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens in *text*.  Uses litellm.token_counter if available,
    otherwise falls back to a simple whitespace split.
    """
    try:
        from litellm import token_counter as _tc
        return _tc(model=model or "gpt-3.5-turbo", text=text)
    except Exception:
        # Pure-Python fallback — ~1.3 words per token is a rough heuristic
        return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(prompt: str) -> FeatureVector:
    """Extract the 5 documented features from a raw prompt string."""
    fv = FeatureVector()

    # F1 — token count
    fv.token_count = _count_tokens(prompt)

    # F2 — code detection (count matching patterns)
    fv.code_signals = sum(1 for p in CODE_PATTERNS if p.search(prompt))

    # F3 — math / reasoning detection
    fv.math_signals = sum(1 for p in MATH_PATTERNS if p.search(prompt))

    # F4 — complexity keyword count
    prompt_lower = prompt.lower()
    fv.complexity_keyword_count = sum(
        1 for kw in COMPLEXITY_KEYWORDS if kw in prompt_lower
    )

    # F5 — multi-question / multi-sentence
    fv.question_marks = prompt.count("?")
    fv.sentence_endings = prompt.count(".") + prompt.count("!") + prompt.count("?")

    return fv


# ---------------------------------------------------------------------------
# Normalization helpers  (map raw values → [0, 1])
# ---------------------------------------------------------------------------

def _norm_token_count(count: int) -> float:
    """Linear interpolation between TOKEN_SHORT and TOKEN_LONG."""
    if count <= TOKEN_SHORT:
        return 0.0
    if count >= TOKEN_LONG:
        return 1.0
    return (count - TOKEN_SHORT) / (TOKEN_LONG - TOKEN_SHORT)


def _norm_code(signals: int) -> float:
    """Any code signal → 1.0 (binary feature with mild graduation)."""
    if signals == 0:
        return 0.0
    if signals == 1:
        return 0.7
    return 1.0  # 2+ signals → full score


def _norm_math(signals: int) -> float:
    """Similar to code — graduated binary."""
    if signals == 0:
        return 0.0
    if signals == 1:
        return 0.7
    return 1.0


def _norm_complexity_kw(count: int) -> float:
    """0 -> 0.0, 1 -> 0.6, 2 -> 0.85, 3+ -> 1.0."""
    mapping = {0: 0.0, 1: 0.6, 2: 0.85}
    return mapping.get(count, 1.0)


def _norm_multi_question(question_marks: int, sentence_endings: int) -> float:
    """Multiple questions or very long multi-sentence prompts."""
    if question_marks >= 3:
        return 1.0
    if question_marks == 2:
        return 0.8
    if sentence_endings >= 5:
        return 0.7
    if sentence_endings >= 3:
        return 0.5
    if question_marks == 1 and sentence_endings >= 2:
        return 0.4
    return 0.0


# ---------------------------------------------------------------------------
# Scoring & classification
# ---------------------------------------------------------------------------

def compute_score(fv: FeatureVector) -> Tuple[float, dict]:
    """
    Compute the weighted score from a FeatureVector.

    Returns
    -------
    score : float       Weighted sum in [0, 1]
    normed : dict       Normalized per-feature values (for logging)
    """
    normed = {
        "F1_token_count": _norm_token_count(fv.token_count),
        "F2_code_detect": _norm_code(fv.code_signals),
        "F3_math_reason": _norm_math(fv.math_signals),
        "F4_complexity_kw": _norm_complexity_kw(fv.complexity_keyword_count),
        "F5_multi_question": _norm_multi_question(fv.question_marks, fv.sentence_endings),
    }

    score = sum(WEIGHTS[k] * normed[k] for k in WEIGHTS)
    return score, normed


def _build_reason(normed: dict, label: str) -> str:
    """Build a human-readable reason string from normalized features."""
    active = [
        (k.split("_", 1)[1], v) for k, v in normed.items() if v >= 0.4
    ]
    if not active:
        return f"Prompt appears straightforward -> {label} model"

    parts = [f"{name}={v:.2f}" for name, v in sorted(active, key=lambda x: -x[1])]
    top_signal = active[0][0] if active else "general"
    return f"Primary signal: {top_signal} | features: {', '.join(parts)} -> {label} model"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RoutingModel:
    """
    The RouteWise routing classifier.

    Usage
    -----
        model = RoutingModel()
        decision = model.classify("Write a Python function to sort a list")
        print(decision.label, decision.confidence, decision.reason)
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        threshold: float = DECISION_THRESHOLD,
    ):
        self.weights = weights or dict(WEIGHTS)
        self.threshold = threshold

    def classify(self, prompt: str) -> RoutingDecision:
        """
        Classify a prompt as requiring the Fast or Capable model.

        Parameters
        ----------
        prompt : str
            The raw user prompt.

        Returns
        -------
        RoutingDecision
            Contains label, confidence, reason, raw_score, features, latency_ms.
        """
        t0 = time.perf_counter()

        # 1. Extract features
        fv = extract_features(prompt)

        # 2. Compute weighted score
        score, normed = compute_score(fv)

        # 3. Apply decision boundary
        if score >= self.threshold:
            label = CAPABLE_LABEL
        else:
            label = FAST_LABEL

        # 4. Confidence: distance from boundary, scaled to [0, 1]
        confidence = min(1.0, abs(score - self.threshold) * 2)

        # 5. Reason string
        reason = _build_reason(normed, label)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RoutingDecision(
            label=label,
            confidence=round(confidence, 4),
            reason=reason,
            raw_score=round(score, 4),
            features=normed,
            latency_ms=round(elapsed_ms, 3),
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_model = RoutingModel()


def classify(prompt: str) -> RoutingDecision:
    """Convenience function using the default RoutingModel instance."""
    return _default_model.classify(prompt)


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_prompts = [
        "What is the capital of France?",
        "Write a Python function that implements merge sort and explain its time complexity step by step.",
        "Hi there!",
        "Debug this code: def foo(x): return x + ; print(foo(3))",
        "Compare the pros and cons of microservices vs monoliths and explain why you'd choose each.",
    ]
    model = RoutingModel()
    for p in test_prompts:
        d = model.classify(p)
        print(f"[{d.label:>7}] conf={d.confidence:.2f} score={d.raw_score:.3f} | {p[:70]}")
        print(f"         reason: {d.reason}")
        print(f"         latency: {d.latency_ms:.1f}ms")
        print()
