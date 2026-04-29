# RAG Agent: Vector store and hybrid graph retrieval
# Retrieves relevant posts from Supabase pgvector
"""
MumzSense v1 — RAG Agent
pgvector similarity search with metadata pre-filtering (PRD §7).

Gate D: use_graph parameter is a stub for Phase 2 GraphRAG extension.
"""
from __future__ import annotations
import hashlib
import logging
import os
import re
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded embedding models
_en_model = None
_ar_model = None


def _is_real_embeddings_enabled() -> bool:
    """Read USE_REAL_EMBEDDINGS at call time so the env var set after import is honoured."""
    # First try config/settings (picks up pydantic env parsing)
    try:
        from config import settings
        if settings.use_real_embeddings:
            return True
    except Exception:
        pass
    # Fallback: read directly from os.environ
    return os.getenv("USE_REAL_EMBEDDINGS", "").lower() in {"1", "true", "yes"}


# Keep backward-compat alias — re-evaluated each function call via helper above
_use_real_embeddings = None  # unused sentinel; real check via _is_real_embeddings_enabled()


def _fallback_embedding(text: str, dimension: int) -> List[float]:
    """Deterministic hashed bag-of-words embedding for local/offline runs."""
    vector = np.zeros(dimension, dtype=np.float32)
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text.lower())
    if not tokens:
        return vector.tolist()

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vector[index] += sign * weight

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector.tolist()


def _get_en_model():
    if not _is_real_embeddings_enabled():
        return None
    global _en_model
    if _en_model is None:
        from sentence_transformers import SentenceTransformer
        from config import settings
        logger.info(f"Loading EN embedding model: {settings.en_embedding_model}")
        _en_model = SentenceTransformer(settings.en_embedding_model)
    return _en_model


def _get_ar_model():
    if not _is_real_embeddings_enabled():
        return None
    global _ar_model
    if _ar_model is None:
        from sentence_transformers import SentenceTransformer
        from config import settings
        logger.info(f"Loading AR embedding model: {settings.ar_embedding_model}")
        _ar_model = SentenceTransformer(settings.ar_embedding_model)
    return _ar_model


def embed_query(text: str, lang: str) -> List[float]:
    """Embed a query using the appropriate model. Returns a flat list."""
    try:
        from config import settings
        if not _is_real_embeddings_enabled():
            return _fallback_embedding(text, settings.ar_embedding_dim if lang == "ar" else settings.en_embedding_dim)
        if lang == "ar":
            model = _get_ar_model()
            vec = model.encode(text, normalize_embeddings=True)
        else:
            model = _get_en_model()
            # bge-large requires instruction prefix for queries (PRD §7.1)
            prefix = "Represent this question for searching relevant maternal experiences: "
            vec = model.encode(prefix + text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        logger.error(f"embed_query failed: {e}")
        from config import settings
        return _fallback_embedding(text, settings.ar_embedding_dim if lang == "ar" else settings.en_embedding_dim)


def embed_text(text: str, lang: str) -> List[float]:
    """Embed a document (no prefix). Used during indexing."""
    try:
        from config import settings
        if not _is_real_embeddings_enabled():
            return _fallback_embedding(text, settings.ar_embedding_dim if lang == "ar" else settings.en_embedding_dim)
        if lang == "ar":
            model = _get_ar_model()
        else:
            model = _get_en_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        logger.error(f"embed_text failed: {e}")
        from config import settings
        return _fallback_embedding(text, settings.ar_embedding_dim if lang == "ar" else settings.en_embedding_dim)


def _confidence_level(max_score: float, num_results: int) -> str:
    """Map similarity score to confidence label (PRD §7.2)."""
    from config import settings
    if num_results < 2 or max_score < settings.similarity_low:
        return "none"
    if max_score >= settings.similarity_high:
        return "high"
    if max_score >= settings.similarity_medium:
        return "medium"
    return "low"


async def retrieve(
    query: str,
    lang: str,
    topic: Optional[str] = None,
    stage: Optional[str] = None,
    top_k: int = 5,
    use_graph: bool = False,  # Gate D — Phase 2 GraphRAG hook
) -> Dict[str, Any]:
    """
    Main retrieval function. Embeds query and searches pgvector.
    Returns retrieval output schema (PRD §7.4).
    """
    from config import settings
    from db.supabase_client import similarity_search

    embedding = embed_query(query, lang)
    if not embedding:
        return _empty_retrieval(lang)

    embedding_col = "ar_embedding" if lang == "ar" else "en_embedding"

    raw_results = await similarity_search(
        query_embedding=embedding,
        embedding_col=embedding_col,
        stage=stage,
        topic=topic,
        top_k=top_k * 2,  # fetch more, post-filter below
    )

    # Post-filter: keep only similarity > similarity_low threshold
    filtered = [
        r for r in raw_results
        if (1.0 - r.get("distance", 1.0)) >= settings.similarity_low
    ]
    # Sort by similarity desc (distance asc), take top_k
    filtered = sorted(
        filtered,
        key=lambda r: r.get("distance", 1.0),
    )[:top_k]

    # Convert distance to similarity score (pgvector returns cosine distance)
    posts = []
    for r in filtered:
        dist = r.get("distance", 1.0)
        sim = max(0.0, 1.0 - dist)
        posts.append({
            "post_id": r.get("post_id", ""),
            "situation": r.get("situation", ""),
            "advice": r.get("advice", ""),
            "outcome": r.get("outcome"),
            "trust_score": r.get("trust_score", 0.75),
            "similarity_score": round(sim, 4),
            "stage": r.get("stage", ""),
            "topic": r.get("topic", ""),
            "lang": r.get("lang", "en"),
        })

    max_sim = max((p["similarity_score"] for p in posts), default=0.0)
    conf = _confidence_level(max_sim, len(posts))

    return {
        "retrieved_posts": posts,
        "retrieval_confidence": conf,
        "max_similarity": round(max_sim, 4),
        "query_lang": lang,
    }


def _empty_retrieval(lang: str) -> Dict[str, Any]:
    return {
        "retrieved_posts": [],
        "retrieval_confidence": "none",
        "max_similarity": 0.0,
        "query_lang": lang,
    }


# ── LangGraph node wrapper ─────────────────────────────────────────────────────

async def rag_node(state: dict) -> dict:
    """
    LangGraph node — async so it runs directly in FastAPI's event loop.
    LangGraph supports async nodes natively via ainvoke(); no thread hacks needed.
    """
    from config import settings

    lang  = state.get("lang_detected", "en")
    query = state.get("query", "")
    topic = state.get("topic")
    stage = state.get("stage_hint")

    try:
        result = await retrieve(
            query, lang,
            topic=topic,
            stage=stage,
            top_k=settings.rag_top_k,
        )
    except Exception as e:
        logger.error(f"rag_node retrieve failed: {e}")
        result = _empty_retrieval(lang)

    return {
        **state,
        "retrieved_posts":      result["retrieved_posts"],
        "retrieval_confidence": result["retrieval_confidence"],
        "max_similarity":       result["max_similarity"],
    }