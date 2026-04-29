# FastAPI application entry point
# Defines all API endpoints and routes
"""
MumzSense v1 — FastAPI Application
All endpoints defined here (PRD §13).
"""
from __future__ import annotations
from contextlib import asynccontextmanager
import json
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── App init ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MumzSense API starting up...")

    # Pre-load classifier on startup so failures surface immediately
    try:
        from agents.classifier_agent import get_classifier_agent
        agent = get_classifier_agent()
        if agent.is_loaded():
            logger.info("Classifier loaded at startup (TF-IDF%s)", " + BiLSTM" if agent._bilstm else "-only")
        else:
            logger.warning("Classifier NOT loaded at startup — will use keyword fallback on first query")
    except Exception as exc:
        logger.warning("Classifier pre-load error at startup: %s", exc)

    # Startup smoke test — run a benign query through the pipeline to catch
    # mis-configuration before the first real user query arrives
    try:
        import os
        if os.getenv("SKIP_STARTUP_SMOKE_TEST", "").lower() not in {"1", "true", "yes"}:
            from graph.pipeline import run_pipeline
            smoke = await run_pipeline(
                query="my baby won't stop crying after feeding",
                stage_hint="newborn",
            )
            conf = smoke.get("retrieval_confidence", "none")
            max_sim = smoke.get("max_similarity", 0.0)
            logger.info(
                "Startup smoke test complete — retrieval_confidence=%s max_similarity=%.3f",
                conf, max_sim,
            )
            if conf == "none":
                logger.warning(
                    "SMOKE TEST WARN: retrieval_confidence=none on startup. "
                    "Check USE_REAL_EMBEDDINGS=true and that embed_and_index.py was run with real embeddings."
                )
    except Exception as exc:
        logger.warning("Startup smoke test failed (non-fatal): %s", exc)

    yield
    logger.info("MumzSense API shutting down...")


app = FastAPI(
    title="MumzSense API",
    description="Bilingual maternal Q&A assistant powered by RAG + Llama 3.1",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Global error handler — always return JSON, never HTML ──────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler that serialises unhandled exceptions as JSON.
    Prevents the frontend from seeing raw HTML 500 responses which
    cause 'Unexpected token I, Internal S... is not valid JSON' errors.
    """
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again.", "type": type(exc).__name__},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure HTTP exceptions are also returned as JSON."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Serve frontend static files if available
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
FRONTEND_ASSETS = FRONTEND_DIR / "assets"

if FRONTEND_INDEX.exists():
    if FRONTEND_ASSETS.exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS)), name="frontend_assets")

    source_assets = FRONTEND_DIR / "src"
    if source_assets.exists():
        app.mount("/src", StaticFiles(directory=str(source_assets)), name="frontend_src")

    # Middleware to serve index.html for all unmapped routes (SPA fallback)
    class SPAFallbackMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path.startswith(("/api", "/assets", "/src", "/docs", "/openapi.json", "/redoc", "/health", "/query", "/corpus", "/feedback", "/admin", "/evals")):
                return await call_next(request)
            # Serve the actual file if it exists in frontend dir
            file_path = FRONTEND_DIR / request.url.path.lstrip("/")
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            # Otherwise serve index.html for client-side routing
            if request.url.path != "/" and not "." in request.url.path.split("/")[-1]:
                return HTMLResponse(FRONTEND_INDEX.read_text(encoding="utf-8"), status_code=200)
            return await call_next(request)

    app.add_middleware(SPAFallbackMiddleware)



# ── CORS ───────────────────────────────────────────────────────────────────────
# Build allowed origins list — always include localhost + any env-configured origins
_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
for _env_var in ("VITE_APP_ORIGIN", "VERCEL_URL", "FRONTEND_URL", "NETLIFY_URL"):
    _val = os.getenv(_env_var, "").strip()
    if _val:
        _allowed_origins.append(_val)
        # Also add https:// version if http:// was provided and vice versa
        if _val.startswith("http://"):
            _allowed_origins.append("https://" + _val[7:])
        elif _val.startswith("https://"):
            _allowed_origins.append("http://" + _val[8:])

logger.info("CORS allowed origins: %s", _allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"https://.*\.netlify\.app|https://.*\.vercel\.app|https://.*\.railway\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ──────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from starlette.requests import Request

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMIT = "20/minute"
    RATE_LIMIT_ENABLED = True
except ImportError:
    logger.warning("slowapi not installed — rate limiting disabled")
    RATE_LIMIT_ENABLED = False

# ── Schemas ────────────────────────────────────────────────────────────────────
from schemas import (
    QueryRequest, QueryResponse, HealthResponse, CorpusStatsResponse,
    FeedbackRequest, FeedbackResponse, CacheFlushResponse, CitedPost
)


# ── Auth helper ────────────────────────────────────────────────────────────────
async def verify_admin(authorization: Optional[str] = Header(None)):
    from config import settings
    if not authorization or authorization != f"Bearer {settings.admin_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check. Used by deployment platform."""
    from db.supabase_client import check_db_health
    from cache.redis_client import check_redis_health

    db_status = await check_db_health()
    redis_status = await check_redis_health()
    try:
        from agents.classifier_agent import get_classifier_agent
        model_status = get_classifier_agent().is_loaded()
    except Exception:
        model_status = False

    overall = "ok" if db_status == "ok" else "degraded"
    from config import settings
    return HealthResponse(
        status=overall,
        db_status=db_status,
        redis_status=redis_status,
        model_loaded=model_status,
        corpus_size=settings.corpus_size,
        version=settings.version,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Primary query endpoint. Runs the full LangGraph RAG pipeline.
    Returns a bilingual grounded answer with source citations.
    """
    from graph.pipeline import run_pipeline

    try:
        state = await run_pipeline(
            query=request.query,
            stage_hint=request.stage_hint,
            lang_preference=request.lang_preference,
        )
    except Exception as e:
        logger.error(f"/query pipeline error: {e}")
        raise HTTPException(status_code=503, detail="Pipeline temporarily unavailable. Please try again.")

    # Build citation objects for response
    citations = []
    for post in state.get("retrieved_posts", []):
        try:
            citations.append(CitedPost(
                post_id=post.get("post_id", ""),
                situation=post.get("situation", ""),
                advice=post.get("advice", ""),
                outcome=post.get("outcome"),
                trust_score=float(post.get("trust_score", 0.75)),
                similarity_score=float(post.get("similarity_score", 0.0)),
                stage=post.get("stage", ""),
                topic=post.get("topic", ""),
                lang=post.get("lang", "en"),
            ))
        except Exception:
            pass

    return QueryResponse(
        answer_primary=state.get("answer_primary", ""),
        answer_secondary=state.get("answer_secondary"),
        citations=citations,
        urgency_flag=state.get("urgency_flag") or state.get("urgency") or "routine",
        confidence_level=state.get("confidence_level") or "medium",
        defer_message=state.get("defer_message"),
        hallucination_risk=state.get("hallucination_risk", False),
        cached=state.get("cached", False),
        latency_ms=state.get("latency_ms", 0),
    )


@app.get("/corpus/stats", response_model=CorpusStatsResponse, tags=["Data"])
async def corpus_stats():
    """Returns corpus statistics for the UI 'About' section."""
    from db.supabase_client import get_corpus_stats
    stats = await get_corpus_stats()
    if not stats:
        # Return static stats from validation report as fallback
        return CorpusStatsResponse(
            total=560,
            by_lang={"en": 440, "ar": 120},
            by_stage={"toddler": 86, "6-12m": 102, "3-6m": 102,
                    "newborn": 90, "0-3m": 115, "trimester": 65},
            by_topic={"feeding": 135, "mental_health": 64, "gear": 36,
                    "health": 114, "development": 79, "sleep": 88, "postpartum": 44},
            by_urgency={"routine": 350, "seek-help": 115, "monitor": 95},
        )
    return CorpusStatsResponse(**stats)


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback_endpoint(request: FeedbackRequest):
    """Log user feedback for Gate A (MADRL preparation in Phase 2)."""
    from db.supabase_client import log_feedback
    try:
        await log_feedback({
            "query_hash": request.query_hash,
            "user_rating": request.rating,
            "was_helpful": request.was_helpful,
            "urgency_felt": request.urgency_felt,
        })
        return FeedbackResponse(logged=True, message="Feedback recorded. Thank you!")
    except Exception as e:
        logger.error(f"/feedback error: {e}")
        return FeedbackResponse(logged=False, message="Could not record feedback at this time.")


@app.post("/admin/cache/flush", response_model=CacheFlushResponse, tags=["Admin"])
async def flush_cache(admin: bool = Depends(verify_admin)):
    """Flush all Redis cache keys. Auth-protected."""
    from cache.redis_client import cache_flush_all
    success = await cache_flush_all()
    return CacheFlushResponse(
        flushed=success,
        message="Cache flushed successfully" if success else "Cache flush failed or cache not configured",
    )


@app.get("/evals/latest", tags=["Evals"])
async def latest_evals():
    """Return the latest evaluation run results."""
    evals_path = Path(__file__).parent.parent / "evals" / "results.json"
    try:
        with open(evals_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"message": "No evaluation results yet. Run evals/run_evals.py first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/", tags=["System"])
async def root():
    """Serve frontend index if present, otherwise redirect to docs."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )