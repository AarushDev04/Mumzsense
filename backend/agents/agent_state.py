"""
MumzSense v1 — Agent State
===========================
TypedDict for LangGraph AgentState (PRD §9.1).

All fields are Optional where they may not be set until
a later node in the pipeline has run.

Phase 2 extension gates are marked in comments.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph pipeline.

    Populated progressively by each node:
        cache_check_node      → cached, cache_key
        classifier_node       → topic, urgency, confidence, lang_detected,
                                 defer_flag, raw_probs_topic, raw_probs_urgency,
                                 bilstm_hidden_state
        rag_node              → retrieved_posts, retrieval_confidence, max_similarity
        synthesis_node        → answer_primary, answer_secondary, citations,
                                 urgency_flag, hallucination_risk
        escalation_node       → defer_message
        cache_write_node      → (no new fields; writes to Redis)
    """

    # ── Input fields (provided by FastAPI handler) ──────────────────────────
    query:           str             # Raw user query
    stage_hint:      Optional[str]   # Stage from UI selector (e.g. "0-3m")
    lang_preference: Optional[str]   # "en" | "ar" — user's explicit preference

    # ── Cache layer ─────────────────────────────────────────────────────────
    cached:    bool    # True if served from Redis
    cache_key: str     # SHA-256 hash key

    # ── Classifier outputs (classifier_node) ────────────────────────────────
    topic:               str    # feeding | sleep | health | development | gear | postpartum | mental_health
    urgency:             str    # routine | monitor | seek-help
    confidence:          float  # min(topic_confidence, urgency_confidence) after calibration
    lang_detected:       str    # en | ar
    defer_flag:          bool   # True → route to escalation_node

    # Sub-fields stored for logging/feedback (not sent to frontend)
    raw_probs_topic:     dict   # {topic: float, ...}  7 classes
    raw_probs_urgency:   dict   # {urgency: float, ...} 3 classes
    bilstm_hidden_state: Optional[list]  # 256-dim vector for Gate A logging

    # ── RAG outputs (rag_node) ───────────────────────────────────────────────
    retrieved_posts:      list   # List of post dicts with similarity scores
    retrieval_confidence: str    # "high" | "medium" | "low" | "none"
    max_similarity:       float  # Highest cosine similarity in retrieved set

    # ── Synthesis outputs (synthesis_node) ───────────────────────────────────
    answer_primary:   str   # Answer in query language
    answer_secondary: str   # Answer in other language
    citations:        list  # List of post_ids cited
    urgency_flag:     str   # Mirrors urgency; used by frontend for badge display
    hallucination_risk: bool  # True if hallucination guard flagged a named entity

    # ── Escalation (escalation_node — Gate B) ────────────────────────────────
    defer_message: Optional[str]  # Populated when pipeline chose not to answer

    # ── Error handling ────────────────────────────────────────────────────────
    error: Optional[str]  # Set by any node on exception; triggers graceful fallback

    # ── Latency tracking ─────────────────────────────────────────────────────
    latency_ms: int  # Total pipeline latency in milliseconds