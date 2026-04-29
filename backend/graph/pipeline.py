# LangGraph pipeline definition
# Sequential graph structure with extension gates
"""
MumzSense v1 — LangGraph Pipeline
====================================
Sequential graph with conditional routing (PRD §9.1).

Phase 1 graph:
    START
      → cache_check_node
      → classifier_node        [Gate A slot]
      → defer_router           if defer_flag → escalation_node → END
      → rag_node               [Gate C slot]
      → threshold_router       if retrieval_confidence=="none" → uncertainty_node → END
      → synthesis_node
      → cache_write_node
    END

Phase 2 additions (without touching existing nodes):
    Add supervisor_node after classifier_node → conditional edges to specialist agents
    Activate Gate B: real triage agent behind defer_router
    Activate Gate D: dual retrieval in rag_node
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Literal

from langgraph.graph import StateGraph, END  # type: ignore

from graph.state import AgentState

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports (heavy deps loaded once, not at module import time)
# ──────────────────────────────────────────────────────────────────────────────

def _get_classifier():
    from agents.classifier_agent import classifier_node
    return classifier_node


def _get_rag():
    from agents.rag_agent import rag_node
    return rag_node


def _get_response():
    from agents.response_agent import response_node
    return response_node


def _get_escalation():
    from agents.escalation_handler import escalation_node
    return escalation_node


def _get_redis():
    from cache.redis_client import get_redis_client
    return get_redis_client()


# ──────────────────────────────────────────────────────────────────────────────
# Node: Cache Check
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_for_cache(query: str) -> str:
    """Normalise query for cache key (PRD §12.2)."""
    import re
    q = query.lower().strip()
    q = re.sub(r'[^\w\s\u0600-\u06FF]', '', q)  # keep alphanumeric + Arabic
    q = re.sub(r'\s+', ' ', q).strip()
    return q


async def cache_check_node(state: AgentState) -> AgentState:
    """
    Check Upstash Redis for a cached response.
    Key: mumzsense:query:{sha256(normalised_query + stage_hint)}
    """
    import json

    query      = state.get("query", "")
    stage_hint = state.get("stage_hint") or ""

    normalised = _normalise_for_cache(query) + stage_hint
    cache_key  = "mumzsense:query:" + hashlib.sha256(normalised.encode()).hexdigest()

    state["cache_key"] = cache_key
    state["cached"]    = False

    try:
        redis = _get_redis()
        if redis is None:
            return state
        cached_val = redis.get(cache_key)
        if cached_val:
            cached = json.loads(cached_val)
            logger.info("Cache HIT: %s", cache_key[:40])
            return {
                **state,
                **cached,
                "cached": True,
            }
    except Exception as e:
        logger.warning("Redis cache check error: %s", e)

    logger.debug("Cache MISS: %s", cache_key[:40])
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Node: Classifier
# ──────────────────────────────────────────────────────────────────────────────

async def classifier_wrapper_node(state: AgentState) -> AgentState:
    """Wraps classifier_agent.classifier_node with error handling."""
    if state.get("cached"):
        return state

    try:
        fn = _get_classifier()
        return fn(state)  # classifier_node is sync (CPU-only sklearn)
    except Exception as e:
        logger.error("Classifier error: %s", e, exc_info=True)
        return {**state, "error": f"classifier_error: {e}", "defer_flag": True}


# ──────────────────────────────────────────────────────────────────────────────
# Node: RAG
# ──────────────────────────────────────────────────────────────────────────────

async def rag_wrapper_node(state: AgentState) -> AgentState:
    if state.get("cached"):
        return state
    try:
        fn = _get_rag()
        return await fn(state)
    except Exception as e:
        logger.error("RAG error: %s", e, exc_info=True)
        return {
            **state,
            "error": f"rag_error: {e}",
            "retrieval_confidence": "none",
            "retrieved_posts": [],
            "max_similarity": 0.0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Response synthesis wrapper
# ──────────────────────────────────────────────────────────────────────────────

async def synthesis_wrapper_node(state: AgentState) -> AgentState:
    if state.get("cached"):
        return state
    try:
        fn = _get_response()
        return await fn(state)
    except Exception as e:
        logger.error("Synthesis error: %s", e, exc_info=True)
        lang = state.get("lang_detected", "en")
        fallback = (
            "I'm sorry, I'm having trouble generating a response right now. "
            "Please try again in a moment."
            if lang == "en" else
            "أعتذر، أواجه صعوبة في إنشاء رد الآن. يرجى المحاولة مرة أخرى."
        )
        return {
            **state,
            "error": f"synthesis_error: {e}",
            "answer_primary":   fallback,
            "answer_secondary": "",
            "citations":        [],
            "hallucination_risk": False,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Escalation wrapper (Gate B — stub in Phase 1)
# ──────────────────────────────────────────────────────────────────────────────

async def escalation_wrapper_node(state: AgentState) -> AgentState:
    if state.get("cached"):
        return state
    try:
        fn = _get_escalation()
        return await fn(state)
    except Exception as e:
        logger.error("Escalation error: %s", e, exc_info=True)
        return {**state, "error": f"escalation_error: {e}"}


# ──────────────────────────────────────────────────────────────────────────────
# Node: Uncertainty (no good retrieval)
# ──────────────────────────────────────────────────────────────────────────────

async def uncertainty_node(state: AgentState) -> AgentState:
    """
    Returns an honest 'I don't know' response when RAG retrieval fails.
    (PRD §7.2 — top_score < 0.45 OR fewer than 2 results above threshold)
    """
    lang = state.get("lang_detected", "en")

    if lang == "ar":
        msg = (
            "لا أملك تجارب مشابهة كافية للإجابة بثقة على هذا السؤال. "
            "أقترح التواصل مع طبيبك أو مجتمعنا للحصول على مساعدة أفضل."
        )
    else:
        msg = (
            "I don't have enough similar experiences to answer this confidently. "
            "You might find better answers by speaking with your healthcare provider "
            "or visiting our community forum."
        )

    return {
        **state,
        "answer_primary":     "",
        "answer_secondary":   "",
        "citations":          [],
        "defer_message":      msg,
        "urgency_flag":       state.get("urgency", "routine"),
        "hallucination_risk": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Error fallback
# ──────────────────────────────────────────────────────────────────────────────

async def error_fallback_node(state: AgentState) -> AgentState:
    lang  = state.get("lang_detected", "en")
    error = state.get("error", "unknown_error")
    logger.error("Pipeline error caught by fallback: %s", error)

    if lang == "ar":
        msg = "حدث خطأ. يرجى المحاولة مرة أخرى."
    else:
        msg = "Something went wrong. Please try again."

    return {
        **state,
        "defer_message":    msg,
        "answer_primary":   "",
        "answer_secondary": "",
        "citations":        [],
        "hallucination_risk": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Cache write
# ──────────────────────────────────────────────────────────────────────────────

async def cache_write_node(state: AgentState) -> AgentState:
    """
    Write response to Redis with appropriate TTL (PRD §12.3).
    seek-help responses are NEVER cached.
    deferred responses are NEVER cached.
    """
    import json
    from datetime import datetime, timezone

    if state.get("cached") or state.get("error"):
        return state

    urgency       = state.get("urgency", "routine")
    defer_message = state.get("defer_message")

    # PRD §12.3: never cache seek-help or deferred responses
    if urgency == "seek-help" or defer_message:
        return state

    ttl_map = {
        "routine":  86400,  # 24 hours
        "monitor":  21600,  # 6 hours
    }
    ttl = ttl_map.get(urgency, 3600)  # uncertainty responses → 1 hour

    payload = {
        "answer_primary":   state.get("answer_primary", ""),
        "answer_secondary": state.get("answer_secondary", ""),
        "citations":        state.get("citations", []),
        "urgency_flag":     state.get("urgency_flag", urgency),
        "confidence_level": _confidence_level(state.get("max_similarity", 0.0)),
        "cached_at":        datetime.now(timezone.utc).isoformat(),
        "cache_version":    "v1.0",
    }

    try:
        redis = _get_redis()
        if redis is None:
            return state
        redis.setex(state["cache_key"], ttl, json.dumps(payload))
        logger.debug("Cache WRITE: %s TTL=%ds", state["cache_key"][:40], ttl)
    except Exception as e:
        logger.warning("Redis write error: %s", e)

    return state


def _confidence_level(max_sim: float) -> str:
    if max_sim >= 0.75:
        return "high"
    if max_sim >= 0.60:
        return "medium"
    if max_sim >= 0.45:
        return "low"
    return "none"


# ──────────────────────────────────────────────────────────────────────────────
# Routing functions
# ──────────────────────────────────────────────────────────────────────────────

def defer_router(state: AgentState) -> Literal["escalation", "rag", "cached", "error"]:
    """Route after classifier."""
    if state.get("cached"):
        return "cached"
    if state.get("error"):
        return "error"
    if state.get("defer_flag"):
        return "escalation"
    return "rag"


def threshold_router(state: AgentState) -> Literal["synthesis", "uncertainty", "cached", "error"]:
    """Route after RAG: check retrieval confidence threshold."""
    if state.get("cached"):
        return "cached"
    if state.get("error"):
        return "error"
    if state.get("retrieval_confidence") == "none":
        return "uncertainty"
    return "synthesis"


# ──────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assembles the Phase 1 sequential LangGraph pipeline.
    """
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("cache_check",    cache_check_node)
    g.add_node("classifier",     classifier_wrapper_node)
    g.add_node("escalation",     escalation_wrapper_node)
    g.add_node("rag",            rag_wrapper_node)
    g.add_node("synthesis",      synthesis_wrapper_node)
    g.add_node("uncertainty",    uncertainty_node)
    g.add_node("error_fallback", error_fallback_node)
    g.add_node("cache_write",    cache_write_node)

    # Edges
    g.set_entry_point("cache_check")
    g.add_edge("cache_check", "classifier")

    # Defer router (after classifier)
    g.add_conditional_edges(
        "classifier",
        defer_router,
        {
            "cached":     "cache_write",
            "error":      "error_fallback",
            "escalation": "escalation",
            "rag":        "rag",
        }
    )

    # Threshold router (after RAG)
    g.add_conditional_edges(
        "rag",
        threshold_router,
        {
            "cached":      "cache_write",
            "error":       "error_fallback",
            "uncertainty": "uncertainty",
            "synthesis":   "synthesis",
        }
    )

    # Convergence to cache_write then END
    g.add_edge("synthesis",     "cache_write")
    g.add_edge("uncertainty",   "cache_write")
    g.add_edge("escalation",    "cache_write")
    g.add_edge("cache_write",   END)
    g.add_edge("error_fallback", END)

    return g


# Compiled graph singleton
_compiled_graph = None


def get_pipeline():
    """Return the compiled LangGraph pipeline (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        g = build_graph()
        _compiled_graph = g.compile()
        logger.info("LangGraph pipeline compiled.")
    return _compiled_graph


async def run_pipeline(query: str,
                       stage_hint: "str | None" = None,
                       lang_preference: "str | None" = None) -> AgentState:
    """
    Async entry point for the full pipeline.
    Called by FastAPI /query endpoint.
    """
    t0       = time.perf_counter()
    pipeline = get_pipeline()

    initial_state: AgentState = {
        "query":           query,
        "stage_hint":      stage_hint,
        "lang_preference": lang_preference,
        "cached":          False,
        "cache_key":       "",
        "defer_flag":      False,
        "retrieved_posts": [],
        "retrieval_confidence": "none",
        "max_similarity":  0.0,
        "citations":       [],
        "hallucination_risk": False,
        "latency_ms":      0,
        "query_hash":      hashlib.sha256(query.encode()).hexdigest(),
    }

    try:
        final_state = await pipeline.ainvoke(initial_state)
    except Exception as e:
        logger.error("Pipeline invoke error: %s", e, exc_info=True)
        final_state = {
            **initial_state,
            "error":        str(e),
            "defer_message": "An error occurred. Please try again.",
        }

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    final_state["latency_ms"] = elapsed_ms
    logger.info("Pipeline complete in %dms (cached=%s)", elapsed_ms, final_state.get("cached"))

    return final_state