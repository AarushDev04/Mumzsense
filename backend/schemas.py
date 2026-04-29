# Pydantic models for FastAPI request/response validation
"""
MumzSense v1 — Pydantic Schemas
Request/response models for FastAPI endpoints.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, validator


# ── Enums ──────────────────────────────────────────────────────────────────────
VALID_STAGES = {"trimester", "newborn", "0-3m", "3-6m", "6-12m", "toddler"}
VALID_LANGS = {"en", "ar"}
VALID_URGENCIES = {"routine", "monitor", "seek-help"}
VALID_TOPICS = {"feeding", "sleep", "health", "development", "gear", "postpartum", "mental_health"}


# ── Request models ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="User's natural language query")
    stage_hint: Optional[str] = Field(None, description="Baby stage from UI selector")
    lang_preference: Optional[str] = Field(None, description="Language override: en or ar")

    @validator("stage_hint")
    def validate_stage(cls, v):
        if v and v not in VALID_STAGES:
            raise ValueError(f"stage_hint must be one of {VALID_STAGES}")
        return v

    @validator("lang_preference")
    def validate_lang(cls, v):
        if v and v not in VALID_LANGS:
            raise ValueError(f"lang_preference must be one of {VALID_LANGS}")
        return v


class FeedbackRequest(BaseModel):
    query_hash: str = Field(..., min_length=10)
    rating: int = Field(..., ge=1, le=5)
    was_helpful: bool
    urgency_felt: Optional[str] = None

    @validator("urgency_felt")
    def validate_urgency(cls, v):
        if v and v not in VALID_URGENCIES:
            raise ValueError(f"urgency_felt must be one of {VALID_URGENCIES}")
        return v


# ── Response models ────────────────────────────────────────────────────────────
class CitedPost(BaseModel):
    post_id: str
    situation: str
    advice: str
    outcome: Optional[str]
    trust_score: float
    similarity_score: float
    stage: str
    topic: str
    lang: str


class QueryResponse(BaseModel):
    answer_primary: str
    answer_secondary: Optional[str] = None
    citations: List[CitedPost] = []
    urgency_flag: str = "routine"
    confidence_level: str = "high"
    defer_message: Optional[str] = None
    hallucination_risk: bool = False
    cached: bool = False
    latency_ms: int = 0


class HealthResponse(BaseModel):
    status: str
    db_status: str
    redis_status: str
    model_loaded: bool
    corpus_size: int
    version: str


class CorpusStatsResponse(BaseModel):
    total: int
    by_lang: dict
    by_stage: dict
    by_topic: dict
    by_urgency: dict


class EvalsResponse(BaseModel):
    run_at: Optional[str]
    classifier_metrics: dict
    rag_metrics: dict
    system_metrics: dict
    test_cases_passed: int
    test_cases_total: int


class FeedbackResponse(BaseModel):
    logged: bool
    message: str


class CacheFlushResponse(BaseModel):
    flushed: bool
    message: str