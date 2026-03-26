"""
RouteWise — Semantic Cache
===========================
An in-memory semantic cache that prevents redundant LLM calls by checking
whether a near-identical prompt has already been answered.

How It Works
------------
1. Every prompt is embedded using sentence-transformers (all-MiniLM-L6-v2).
2. Incoming embedding is compared to all stored embeddings via cosine similarity.
3. If best similarity ≥ CACHE_THRESHOLD → return cached response (cache hit).
4. Otherwise → cache miss, the prompt proceeds to the LLM.
5. After the LLM responds, the new (embedding, response, metadata) is stored.

Threshold is a research variable:
    High (0.90+)  → fewer hits, higher answer accuracy
    Low  (0.75)   → more hits, risk of stale / wrong cached answers
    Default: 0.85 (configurable via env CACHE_THRESHOLD)

This is a CUSTOM implementation — it does NOT use LiteLLM's built-in cache,
as the PS requires an explainable, tunable cache with similarity logging.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("routewise.cache")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.85
MAX_CACHE_SIZE = 500  # cap to prevent OOM in long-running POC


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached prompt–response pair."""
    prompt: str
    embedding: np.ndarray
    response: str
    model_used: str
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class CacheLookupResult:
    """Result of a cache lookup — hit or miss."""
    hit: bool
    similarity: float = 0.0
    cached_response: Optional[str] = None
    cached_model: Optional[str] = None
    cached_metadata: Optional[dict] = None
    lookup_ms: float = 0.0


# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    """Lazy-load the sentence-transformer model."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
    return _encoder


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string into a dense vector."""
    encoder = _get_encoder()
    return encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    Since embeddings are already normalized, this is just the dot product.
    """
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    In-memory semantic cache with configurable similarity threshold.

    Usage
    -----
        cache = SemanticCache(threshold=0.85)

        # Check before calling LLM
        result = cache.lookup("What is Python?")
        if result.hit:
            return result.cached_response

        # After LLM call, store the result
        cache.store("What is Python?", response_text, model_used="fast")
    """

    def __init__(self, threshold: Optional[float] = None):
        raw = os.getenv("CACHE_THRESHOLD", "")
        if threshold is not None:
            self.threshold = threshold
        elif raw:
            self.threshold = float(raw)
        else:
            self.threshold = DEFAULT_THRESHOLD

        self._entries: List[CacheEntry] = []
        self._stats = {"hits": 0, "misses": 0}

        logger.info(f"SemanticCache initialized | threshold={self.threshold}")

    # -- Public API --------------------------------------------------------

    def lookup(self, prompt: str) -> CacheLookupResult:
        """
        Check if a semantically similar prompt is already cached.

        Returns a CacheLookupResult with hit=True/False and similarity score.
        """
        t0 = time.perf_counter()

        if not self._entries:
            elapsed = (time.perf_counter() - t0) * 1000
            self._stats["misses"] += 1
            return CacheLookupResult(hit=False, similarity=0.0, lookup_ms=elapsed)

        query_emb = embed_text(prompt)
        best_sim = 0.0
        best_entry: Optional[CacheEntry] = None

        for entry in self._entries:
            sim = cosine_similarity(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        elapsed = (time.perf_counter() - t0) * 1000

        if best_sim >= self.threshold and best_entry is not None:
            best_entry.hit_count += 1
            self._stats["hits"] += 1
            logger.info(
                f"Cache HIT  | similarity={best_sim:.4f} | threshold={self.threshold} "
                f"| prompt='{prompt[:60]}...'"
            )
            return CacheLookupResult(
                hit=True,
                similarity=round(best_sim, 4),
                cached_response=best_entry.response,
                cached_model=best_entry.model_used,
                cached_metadata=best_entry.metadata,
                lookup_ms=round(elapsed, 2),
            )
        else:
            self._stats["misses"] += 1
            logger.info(
                f"Cache MISS | best_similarity={best_sim:.4f} | threshold={self.threshold} "
                f"| prompt='{prompt[:60]}...'"
            )
            return CacheLookupResult(
                hit=False,
                similarity=round(best_sim, 4),
                lookup_ms=round(elapsed, 2),
            )

    def store(
        self,
        prompt: str,
        response: str,
        model_used: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store a new prompt–response pair in the cache."""
        if len(self._entries) >= MAX_CACHE_SIZE:
            # Evict oldest entry
            self._entries.pop(0)

        embedding = embed_text(prompt)
        entry = CacheEntry(
            prompt=prompt,
            embedding=embedding,
            response=response,
            model_used=model_used,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        logger.debug(f"Cache STORE | entries={len(self._entries)} | model={model_used}")

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            "total_lookups": total,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 4),
            "entries_stored": len(self._entries),
            "threshold": self.threshold,
        }

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        self._entries.clear()
        self._stats = {"hits": 0, "misses": 0}
