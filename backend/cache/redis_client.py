# Redis/Upstash client wrapper
"""
MumzSense v1 — Redis / Upstash Cache Client
Implements the caching strategy from PRD §12.
"""
from __future__ import annotations
import json
import hashlib
import re
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# TTL map per urgency (seconds)
CACHE_TTL = {
    "routine":     86400,   # 24 hours
    "monitor":     21600,   # 6 hours
    "seek-help":   None,    # NEVER cache
    "uncertainty": 3600,    # 1 hour
    "deferred":    None,    # NEVER cache
}

CACHE_VERSION = "v1.0"

# Module-level Redis client singleton
_redis_client = None


def get_redis_client():
    """
    Return a Redis client instance (singleton).
    Returns None when REDIS_URL is not configured — callers must handle gracefully.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    from config import settings
    if not settings.redis_url:
        logger.warning("REDIS_URL not set — caching disabled")
        return None

    try:
        import redis
        _redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        # Verify connectivity
        _redis_client.ping()
        logger.info("Redis client connected")
        return _redis_client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        _redis_client = None
        return None


def normalise_query(query: str) -> str:
    """Normalise query string for cache keying (PRD §12.2)."""
    q = query.lower()
    q = re.sub(r"https?://\S+", "", q)           # strip URLs
    q = re.sub(r"[^\w\s\u0600-\u06FF]", "", q)   # keep Arabic Unicode + word chars
    q = re.sub(r"\s+", " ", q).strip()
    return q


def make_cache_key(query: str, stage_hint: Optional[str] = None) -> str:
    """Build deterministic cache key from (query, stage_hint) pair."""
    normalised = normalise_query(query)
    payload    = normalised + (stage_hint or "")
    digest     = hashlib.sha256(payload.encode()).hexdigest()
    return f"mumzsense:query:{digest}"


async def cache_get(key: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached response. Returns None on miss or error."""
    try:
        r = get_redis_client()
        if not r:
            return None
        raw = r.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        # Version guard
        if data.get("cache_version") != CACHE_VERSION:
            return None
        return data
    except Exception as e:
        logger.warning(f"cache_get error (non-fatal): {e}")
        return None


async def cache_set(key: str, response: Dict[str, Any], urgency: str = "routine") -> bool:
    """Write response to Redis with appropriate TTL. Skips for seek-help."""
    ttl = CACHE_TTL.get(urgency)
    if ttl is None:
        logger.debug(f"Caching skipped for urgency={urgency}")
        return False
    try:
        r = get_redis_client()
        if not r:
            return False
        payload = {
            **response,
            "cached_at":     datetime.now(timezone.utc).isoformat(),
            "cache_version": CACHE_VERSION,
        }
        r.setex(key, ttl, json.dumps(payload))
        return True
    except Exception as e:
        logger.warning(f"cache_set error (non-fatal): {e}")
        return False


async def cache_flush_all() -> bool:
    """Flush all mumzsense:query:* keys. Called on corpus/model update."""
    try:
        r = get_redis_client()
        if not r:
            return False
        keys = list(r.scan_iter("mumzsense:query:*"))
        if keys:
            r.delete(*keys)
        logger.info(f"Flushed {len(keys)} cache keys")
        return True
    except Exception as e:
        logger.error(f"cache_flush_all failed: {e}")
        return False


async def check_redis_health() -> str:
    """Returns 'ok', 'no_credentials', or error string."""
    try:
        r = get_redis_client()
        if not r:
            return "no_credentials"
        r.ping()
        return "ok"
    except Exception as e:
        return f"error: {str(e)}"