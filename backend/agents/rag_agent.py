"""
RAG Agent — retrieves top-K similar posts from local_store.json or Supabase pgvector.

Key fix (2026-04-30):
  _confidence_level() NEVER returns "none" when posts are retrieved.
  Any retrieved post set → at minimum "low" confidence → Groq always fires.
  The old behaviour returned "none" for scores below SIMILARITY_LOW (default 0.45),
  which triggered a hard defer path BEFORE the system prompt was ever sent to Groq.
  That is now removed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings (read from env with safe defaults for hash-embedding mode)
# ---------------------------------------------------------------------------
SIMILARITY_LOW    = float(os.getenv("SIMILARITY_LOW",    "0.001"))
SIMILARITY_MEDIUM = float(os.getenv("SIMILARITY_MEDIUM", "0.05"))
SIMILARITY_HIGH   = float(os.getenv("SIMILARITY_HIGH",   "0.20"))
USE_REAL_EMBEDDINGS = os.getenv("USE_REAL_EMBEDDINGS", "false").lower() == "true"
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

_store: dict | None = None
_en_model = None
_ar_model = None


# ---------------------------------------------------------------------------
# Local store loader
# ---------------------------------------------------------------------------

def _load_store() -> dict:
    global _store
    if _store is not None:
        return _store
    store_path = Path(__file__).parent.parent / "data" / "local_store.json"
    if not store_path.exists():
        logger.warning("local_store.json not found — returning empty store")
        _store = {"posts": []}
        return _store
    with open(store_path, encoding="utf-8") as f:
        _store = json.load(f)
    logger.info(f"Loaded {len(_store.get('posts', []))} posts from local_store.json")
    return _store


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _fallback_embedding(text: str) -> list[float]:
    """
    Deterministic 64-dim hash embedding.  Used when sentence_transformers
    is unavailable.  Critically: BOTH the store and the query MUST use this
    same function — otherwise cosine similarity is meaningless.
    """
    vec = [0.0] * 64
    for i, ch in enumerate(text):
        h = int(hashlib.md5(f"{i}:{ch}".encode()).hexdigest(), 16)
        vec[h % 64] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _load_models():
    global _en_model, _ar_model
    if not USE_REAL_EMBEDDINGS:
        return
    if _en_model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading EN embedding model: BAAI/bge-large-en-v1.5")
        _en_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        logger.info("Loading AR embedding model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        _ar_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    except Exception as e:
        logger.warning(f"Could not load real embedding models: {e}. Falling back to hash embeddings.")


def _embed_text(text: str, lang: str = "en") -> list[float]:
    if not USE_REAL_EMBEDDINGS or (_en_model is None and _ar_model is None):
        return _fallback_embedding(text)
    try:
        model = _ar_model if lang == "ar" and _ar_model else _en_model
        if model is None:
            return _fallback_embedding(text)
        return model.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        logger.warning(f"embed_text failed: {e}. Using fallback.")
        return _fallback_embedding(text)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb  = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Confidence level — CRITICAL FIX
# Never returns "none" when posts are present.
# "none" caused a hard-coded defer BEFORE Groq was called.
# ---------------------------------------------------------------------------

def _confidence_level(max_score: float, n_posts: int) -> str:
    """
    Returns one of: "low", "medium", "high".
    
    "none" is intentionally REMOVED from this function.
    If posts were retrieved (n_posts > 0), Groq always fires.
    Groq's own system prompt decides whether to answer or defer —
    that is the correct place for that judgement, not here.
    """
    if n_posts == 0:
        # No posts at all — pipeline should short-circuit before this
        return "none"
    if max_score >= SIMILARITY_HIGH:
        return "high"
    if max_score >= SIMILARITY_MEDIUM:
        return "medium"
    # Below SIMILARITY_LOW or between LOW and MEDIUM — still "low", not "none"
    return "low"


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    lang: str = "en",
    stage_hint: str | None = None,
    topic_hint: str | None = None,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Returns:
        {
          "posts":               list[dict],   # top-K retrieved posts
          "confidence_level":   str,           # "low" | "medium" | "high"
          "max_similarity":     float,
          "query_embedding":    list[float],
        }
    """
    _load_models()
    store = _load_store()
    all_posts = store.get("posts", [])

    if not all_posts:
        logger.warning("local_store.json is empty — nothing to retrieve")
        return {
            "posts": [],
            "confidence_level": "none",
            "max_similarity": 0.0,
            "query_embedding": [],
        }

    # Embed the query
    q_vec = _embed_text(query, lang=lang)

    # Score every post
    scored: list[tuple[float, dict]] = []
    for post in all_posts:
        # Stage filter (if hint provided)
        if stage_hint and post.get("stage") and post["stage"] != stage_hint:
            continue

        # Choose the right embedding field
        emb_key = "ar_embedding" if lang == "ar" else "en_embedding"
        post_vec = post.get(emb_key) or post.get("embedding") or []

        if not post_vec:
            continue

        score = _cosine(q_vec, post_vec)
        scored.append((score, post))

    if not scored:
        # Nothing passed the stage filter — retry without filter
        logger.warning("Stage filter removed all candidates — retrying without filter")
        for post in all_posts:
            emb_key = "ar_embedding" if lang == "ar" else "en_embedding"
            post_vec = post.get(emb_key) or post.get("embedding") or []
            if not post_vec:
                continue
            score = _cosine(q_vec, post_vec)
            scored.append((score, post))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    max_score = top[0][0] if top else 0.0
    posts_out = []
    for score, post in top:
        p = dict(post)
        p["similarity_score"] = round(score, 4)
        # Remove large embedding vectors from output to keep response lean
        p.pop("en_embedding", None)
        p.pop("ar_embedding", None)
        p.pop("embedding", None)
        posts_out.append(p)

    confidence = _confidence_level(max_score, len(posts_out))

    logger.info(
        f"RAG: query={query[:40]!r} stage={stage_hint} "
        f"candidates={len(scored)} top_score={max_score:.4f} confidence={confidence}"
    )

    return {
        "posts": posts_out,
        "confidence_level": confidence,
        "max_similarity": round(max_score, 4),
        "query_embedding": q_vec,
    }