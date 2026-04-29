# AgentState TypedDict for LangGraph pipeline
"""
MumzSense v1 — AgentState TypedDict
Shared state that flows through the LangGraph pipeline.
"""
from __future__ import annotations
from typing import List, Optional, TypedDict


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
    query:           str                    # Raw user query
    stage_hint:      Optional[str]          # Stage from UI selector (e.g. "0-3m")
    lang_preference: Optional[str]          # "en" | "ar" — user's explicit preference

    # ── Cache layer ─────────────────────────────────────────────────────────
    cached:    bool                         # True if served from Redis
    cache_key: str                          # SHA-256 hash key
    query_hash: str                         # For feedback logging

    # ── Classifier outputs ──────────────────────────────────────────────────
    topic:               str               # feeding | sleep | health | development | gear | postpartum | mental_health
    topic_confidence:    float             # Raw topic confidence
    urgency:             str               # routine | monitor | seek-help
    urgency_confidence:  float             # Raw urgency confidence
    confidence:          float             # min(topic_confidence, urgency_confidence)
    lang_detected:       str               # en | ar
    defer_flag:          bool              # True → route to escalation_node

    # Sub-fields for logging/feedback (not sent to frontend)
    raw_probs_topic:     dict              # {topic: float, ...}  7 classes
    raw_probs_urgency:   dict              # {urgency: float, ...} 3 classes
    bilstm_hidden_state: Optional[List[float]]  # 256-dim vector for Gate A logging

    # ── RAG outputs ─────────────────────────────────────────────────────────
    retrieved_posts:      List[dict]       # List of post dicts with similarity scores
    retrieval_confidence: str              # "high" | "medium" | "low" | "none"
    max_similarity:       float            # Highest cosine similarity in retrieved set

    # ── Synthesis outputs ────────────────────────────────────────────────────
    answer_primary:   str                  # Answer in query language
    answer_secondary: Optional[str]        # Answer in other language
    citations:        List[str]            # List of post_ids cited
    urgency_flag:     str                  # Mirrors urgency; used by frontend for badge display
    confidence_level: str                  # "high" | "medium" | "low" | "none" | "deferred"
    hallucination_risk: bool               # True if hallucination guard flagged a named entity

    # ── Escalation (Gate B) ──────────────────────────────────────────────────
    defer_message: Optional[str]           # Populated when pipeline chose not to answer

    # ── Error handling ────────────────────────────────────────────────────────
    error: Optional[str]                   # Set by any node on exception

    # ── Latency tracking ─────────────────────────────────────────────────────
    latency_ms: int                        # Total pipeline latency in milliseconds