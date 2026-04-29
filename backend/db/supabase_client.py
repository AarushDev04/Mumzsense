# Supabase client and connection handler
"""
MumzSense v1 — Supabase Client
Handles all database operations including pgvector similarity search.
"""
from __future__ import annotations

import asyncio
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level client singleton
_supabase_client = None
_local_store_notice_logged = False

_LOCAL_STORE_PATH = Path(__file__).parent.parent / "data" / "local_store.json"


def _is_placeholder_database_url() -> bool:
    from config import settings
    database_url = (settings.database_url or "").strip()
    if not database_url:
        return True
    placeholder_markers = (
        "your_database_connection_string_here",
        "example",
        "placeholder",
    )
    return any(marker in database_url.lower() for marker in placeholder_markers)


def _has_local_database_url() -> bool:
    from config import settings
    return bool(settings.database_url and settings.database_url.startswith("postgres"))


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(v):.10f}" for v in values) + "]"


def _ensure_local_store() -> dict[str, Any]:
    _LOCAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _LOCAL_STORE_PATH.exists():
        payload = {
            "posts": [],
            "feedback_log": [],
            "query_cache_log": [],
        }
        _LOCAL_STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    try:
        return json.loads(_LOCAL_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {
            "posts": [],
            "feedback_log": [],
            "query_cache_log": [],
        }
        _LOCAL_STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


def _save_local_store(store: dict[str, Any]) -> None:
    _LOCAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOCAL_STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _local_cosine_distance(left: List[float], right: List[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.size == 0 or right_array.size == 0:
        return 1.0
    if left_array.shape != right_array.shape:
        size = min(left_array.size, right_array.size)
        left_array = left_array[:size]
        right_array = right_array[:size]
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = float(np.dot(left_array, right_array) / (left_norm * right_norm))
    return max(0.0, min(2.0, 1.0 - similarity))


def _local_connect():
    from config import settings
    import psycopg
    return psycopg.connect(settings.database_url)


def _local_fetchall(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    from psycopg.rows import dict_row

    with _local_connect() as conn:
        conn.row_factory = dict_row
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description is None:
                conn.commit()
                return []
            rows = cur.fetchall()
            conn.commit()
            return [dict(row) for row in rows]


def _local_execute(query: str, params: tuple[Any, ...] = ()) -> None:
    with _local_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()


def get_supabase_client():
    """
    Return Supabase client singleton.
    Resolution order (PRD §15):
      1. If supabase_url + supabase_key are set → use Supabase REST client (pgvector via RPC)
      2. If database_url is a real postgres:// URL → use local psycopg connection
      3. Otherwise → use local_store.json file store (dev/offline mode)
    Returns None for options 2 and 3 (callers check _is_placeholder_database_url() /
    _has_local_database_url() to select the right fallback path).
    """
    global _supabase_client
    global _local_store_notice_logged
    if _supabase_client is not None:
        return _supabase_client

    from config import settings

    # --- Priority 1: Real Supabase project (REST client + pgvector RPC) ---
    if settings.supabase_url and settings.supabase_key:
        try:
            from supabase import create_client
            _supabase_client = create_client(settings.supabase_url, settings.supabase_key)
            logger.info("Supabase client initialised (REST/pgvector mode)")
            return _supabase_client
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {e}. Falling back to local store.")

    # --- Priority 2: Placeholder DATABASE_URL → use local file store ---
    if _is_placeholder_database_url():
        if not _local_store_notice_logged:
            logger.info(
                "DATABASE_URL is a placeholder and Supabase credentials are absent "
                "— using local file store (local_store.json). "
                "Set USE_REAL_EMBEDDINGS=true and re-run embed_and_index.py for real search."
            )
            _local_store_notice_logged = True
        return None

    # --- Priority 3: Real local postgres DATABASE_URL (Docker / dev postgres) ---
    logger.info("Using local psycopg connection (DATABASE_URL is a real postgres URL)")
    return None


async def check_db_health() -> str:
    """Returns 'ok', 'no_credentials', or error message."""
    try:
        client = get_supabase_client()
        if not client:
            if _is_placeholder_database_url():
                _ensure_local_store()
                return "ok"
            if not _has_local_database_url():
                return "no_credentials"
            await asyncio.to_thread(_local_fetchall, "SELECT 1 FROM posts LIMIT 1")
            return "ok"
        result = client.table("posts").select("post_id").limit(1).execute()
        return "ok"
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return f"error: {str(e)}"


async def similarity_search(
    query_embedding: List[float],
    embedding_col: str,          # "en_embedding" or "ar_embedding"
    stage: Optional[str] = None,
    topic: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Perform pgvector cosine similarity search with optional metadata pre-filtering.
    Calls the 'match_posts' RPC function defined in schema.sql.
    Returns list of post dicts with distance appended.
    """
    try:
        client = get_supabase_client()
        if not client:
            if _is_placeholder_database_url():
                store = _ensure_local_store()
                results: list[dict[str, Any]] = []
                for post in store.get("posts", []):
                    if stage and post.get("stage") != stage:
                        continue
                    if topic and post.get("topic") != topic:
                        continue
                    embedding = post.get(embedding_col) or []
                    if not embedding:
                        continue
                    results.append({
                        **post,
                        "distance": _local_cosine_distance(query_embedding, embedding),
                    })
                results.sort(key=lambda item: item.get("distance", 1.0))
                return results[:top_k]
            if not _has_local_database_url():
                return []
            query_sql = """
                SELECT *
                FROM match_posts(%s::vector, %s, %s, %s, %s)
            """
            params = (
                _vector_literal(query_embedding),
                embedding_col,
                top_k,
                stage,
                topic,
            )
            return await asyncio.to_thread(_local_fetchall, query_sql, params)

        params = {
            "query_embedding": query_embedding,
            "embedding_col":   embedding_col,
            "match_count":     top_k,
            "filter_stage":    stage,
            "filter_topic":    topic,
        }
        result = client.rpc("match_posts", params).execute()
        return result.data or []

    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []


async def get_post_by_id(post_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single post by UUID."""
    try:
        client = get_supabase_client()
        if not client:
            if not _has_local_database_url():
                return None
            rows = await asyncio.to_thread(
                _local_fetchall,
                "SELECT * FROM posts WHERE post_id = %s LIMIT 1",
                (post_id,),
            )
            return rows[0] if rows else None
        result = client.table("posts").select("*").eq("post_id", post_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"get_post_by_id failed: {e}")
        return None


async def get_corpus_stats() -> Dict[str, Any]:
    """Return corpus statistics for /corpus/stats endpoint."""
    try:
        client = get_supabase_client()
        if not client:
            if not _has_local_database_url():
                return {}
            posts = await asyncio.to_thread(
                _local_fetchall,
                "SELECT post_id, lang, stage, topic, urgency FROM posts",
            )
            stats: Dict[str, Any] = {
                "total": len(posts),
                "by_lang": {},
                "by_stage": {},
                "by_topic": {},
                "by_urgency": {},
            }
            for p in posts:
                for key in ("lang", "stage", "topic", "urgency"):
                    val = p.get(key, "unknown")
                    group = f"by_{key}"
                    stats[group][val] = stats[group].get(val, 0) + 1
            return stats
        result = client.table("posts").select(
            "post_id, lang, stage, topic, urgency"
        ).execute()
        posts = result.data or []

        stats: Dict[str, Any] = {
            "total": len(posts),
            "by_lang":    {},
            "by_stage":   {},
            "by_topic":   {},
            "by_urgency": {},
        }
        for p in posts:
            for key in ("lang", "stage", "topic", "urgency"):
                val   = p.get(key, "unknown")
                group = f"by_{key}"
                stats[group][val] = stats[group].get(val, 0) + 1
        return stats
    except Exception as e:
        logger.error(f"get_corpus_stats failed: {e}")
        return {}


async def insert_post(post: Dict[str, Any]) -> bool:
    """Insert or update a single post (used by embed_and_index.py)."""
    try:
        client = get_supabase_client()
        if not client:
            if _is_placeholder_database_url():
                store = _ensure_local_store()
                posts = store.get("posts", [])
                for index, existing in enumerate(posts):
                    if existing.get("post_id") == post.get("post_id"):
                        posts[index] = post
                        break
                else:
                    posts.append(post)
                store["posts"] = posts
                _save_local_store(store)
                return True
            if not _has_local_database_url():
                return False
            sql = """
                INSERT INTO posts (
                    post_id, baby_age_weeks, stage, topic, urgency,
                    situation, advice, outcome, trust_score, lang,
                    en_embedding, ar_embedding, verified
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s::vector, %s::vector, %s
                )
                ON CONFLICT (post_id) DO UPDATE SET
                    baby_age_weeks = EXCLUDED.baby_age_weeks,
                    stage = EXCLUDED.stage,
                    topic = EXCLUDED.topic,
                    urgency = EXCLUDED.urgency,
                    situation = EXCLUDED.situation,
                    advice = EXCLUDED.advice,
                    outcome = EXCLUDED.outcome,
                    trust_score = EXCLUDED.trust_score,
                    lang = EXCLUDED.lang,
                    en_embedding = EXCLUDED.en_embedding,
                    ar_embedding = EXCLUDED.ar_embedding,
                    verified = EXCLUDED.verified
            """
            params = (
                post.get("post_id"),
                post.get("baby_age_weeks"),
                post.get("stage"),
                post.get("topic"),
                post.get("urgency"),
                post.get("situation"),
                post.get("advice"),
                post.get("outcome"),
                float(post.get("trust_score", 0.75)),
                post.get("lang", "en"),
                _vector_literal(post.get("en_embedding", [])),
                _vector_literal(post.get("ar_embedding", [])),
                bool(post.get("verified", False)),
            )
            await asyncio.to_thread(_local_execute, sql, params)
            return True
        client.table("posts").upsert(post, on_conflict="post_id").execute()
        return True
    except Exception as e:
        logger.error(f"insert_post failed: {e}")
        return False


async def log_feedback(payload: Dict[str, Any]) -> bool:
    """Log classifier decision + user feedback to feedback_log (Gate A)."""
    try:
        client = get_supabase_client()
        if not client:
            if _is_placeholder_database_url():
                store = _ensure_local_store()
                store.setdefault("feedback_log", []).append(payload)
                _save_local_store(store)
                return True
            if not _has_local_database_url():
                return False
            await asyncio.to_thread(
                _local_execute,
                """
                INSERT INTO feedback_log (query_hash, user_rating, was_helpful, urgency_felt)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    payload.get("query_hash"),
                    payload.get("user_rating"),
                    payload.get("was_helpful"),
                    payload.get("urgency_felt"),
                ),
            )
            return True
        client.table("feedback_log").insert(payload).execute()
        return True
    except Exception as e:
        logger.error(f"log_feedback failed: {e}")
        return False


async def log_query_cache(query_hash: str, query_normalised: str) -> None:
    """Upsert query into cache analytics log."""
    try:
        client = get_supabase_client()
        if not client:
            if _is_placeholder_database_url():
                store = _ensure_local_store()
                records = store.setdefault("query_cache_log", [])
                for record in records:
                    if record.get("query_hash") == query_hash:
                        record["cache_hits"] = int(record.get("cache_hits", 0)) + 1
                        record["last_hit"] = "now()"
                        break
                else:
                    records.append({
                        "query_hash": query_hash,
                        "query_normalised": query_normalised,
                        "cache_hits": 0,
                        "last_hit": "now()",
                    })
                _save_local_store(store)
                return
            if not _has_local_database_url():
                return
            await asyncio.to_thread(
                _local_execute,
                """
                INSERT INTO query_cache_log (query_hash, query_normalised, cache_hits, last_hit)
                VALUES (%s, %s, 0, now())
                ON CONFLICT (query_hash) DO UPDATE SET
                    cache_hits = query_cache_log.cache_hits + 1,
                    last_hit = now()
                """,
                (query_hash, query_normalised),
            )
            return
        existing = client.table("query_cache_log").select("cache_hits").eq(
            "query_hash", query_hash
        ).execute()
        if existing.data:
            hits = existing.data[0]["cache_hits"] + 1
            client.table("query_cache_log").update(
                {"cache_hits": hits, "last_hit": "now()"}
            ).eq("query_hash", query_hash).execute()
        else:
            client.table("query_cache_log").insert(
                {"query_hash": query_hash, "query_normalised": query_normalised, "cache_hits": 0}
            ).execute()
    except Exception as e:
        logger.warning(f"log_query_cache failed (non-critical): {e}")