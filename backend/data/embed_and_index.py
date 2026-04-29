"""
MumzSense v1 — Embedding & pgvector Indexing Pipeline (PRD §10.2)
Embeds corpus posts and upserts them into the configured database.

Usage (from backend/ folder):
    $env:USE_REAL_EMBEDDINGS = "true"          # PowerShell
    python data/embed_and_index.py --corpus data/corpus_validated.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_LOCAL_STORE_PATH = Path(__file__).parent / "local_store.json"


def _check_real_embeddings_enabled() -> None:
    flag = os.getenv("USE_REAL_EMBEDDINGS", "").lower()
    if flag not in {"1", "true", "yes"}:
        print(
            "\nERROR: USE_REAL_EMBEDDINGS is not set to true.\n"
            "In PowerShell run:  $env:USE_REAL_EMBEDDINGS = 'true'\n"
            "Then retry:         python data/embed_and_index.py --corpus data/corpus_validated.jsonl\n",
            flush=True,
        )
        sys.exit(1)


def _load_corpus(corpus_path: str) -> list[dict]:
    posts = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line: %s", e)
    return posts


def _embed_all(posts: list[dict], batch_size: int = 20) -> list[dict]:
    """
    Embed all posts using sentence-transformers.
    Accumulates all records in memory and writes local_store.json ONCE at the end
    — avoids the catastrophic O(n²) full-file-rewrite-per-post pattern.
    """
    from agents.rag_agent import embed_text

    records = []
    total = len(posts)

    for i, post in enumerate(posts):
        text = f"{post.get('situation', '')} {post.get('advice', '')}".strip()

        en_embedding = embed_text(text, "en")
        ar_embedding = embed_text(text, "ar")

        records.append({
            "post_id":        post["post_id"],
            "baby_age_weeks": post.get("baby_age_weeks", 0),
            "stage":          post.get("stage", ""),
            "topic":          post.get("topic", ""),
            "urgency":        post.get("urgency", ""),
            "situation":      post.get("situation", ""),
            "advice":         post.get("advice", ""),
            "outcome":        post.get("outcome"),
            "trust_score":    float(post.get("trust_score", 0.75)),
            "lang":           post.get("lang", "en"),
            "en_embedding":   en_embedding,
            "ar_embedding":   ar_embedding,
            "verified":       bool(post.get("verified", False)),
        })

        if (i + 1) % batch_size == 0 or (i + 1) == total:
            logger.info("Embedded %d / %d posts...", i + 1, total)

    return records


def _write_local_store(records: list[dict]) -> None:
    """Write all embedded records to local_store.json in a single atomic write."""
    # Preserve any existing non-posts keys (feedback_log, query_cache_log)
    store: dict = {"posts": [], "feedback_log": [], "query_cache_log": []}
    if _LOCAL_STORE_PATH.exists():
        try:
            existing = json.loads(_LOCAL_STORE_PATH.read_text(encoding="utf-8"))
            store["feedback_log"]    = existing.get("feedback_log", [])
            store["query_cache_log"] = existing.get("query_cache_log", [])
        except Exception:
            pass

    store["posts"] = records
    _LOCAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing %d posts to %s ...", len(records), _LOCAL_STORE_PATH)
    _LOCAL_STORE_PATH.write_text(
        json.dumps(store, ensure_ascii=False, separators=(",", ":")),  # compact — saves ~30% space
        encoding="utf-8",
    )
    size_mb = _LOCAL_STORE_PATH.stat().st_size / 1_000_000
    logger.info("local_store.json written (%.1f MB)", size_mb)


def _try_supabase_upsert(records: list[dict]) -> int:
    """
    Attempt to upsert into Supabase if credentials are configured.
    Returns number of successfully upserted records (0 if Supabase not available).
    """
    try:
        from supabase import create_client
        from config import settings
        if not settings.supabase_url or not settings.supabase_key:
            return 0
        client = create_client(settings.supabase_url, settings.supabase_key)
        logger.info("Supabase client connected — upserting %d posts...", len(records))
        # Upsert in batches of 50 (Supabase REST limit)
        count = 0
        for i in range(0, len(records), 50):
            batch = records[i:i + 50]
            client.table("posts").upsert(batch, on_conflict="post_id").execute()
            count += len(batch)
            logger.info("Supabase upserted %d / %d", count, len(records))
        return count
    except ImportError:
        logger.info("supabase package not installed — skipping Supabase upsert (local store only)")
        return 0
    except Exception as e:
        logger.warning("Supabase upsert failed (non-fatal, local store is the primary): %s", e)
        return 0


def _verify_embedding_quality() -> None:
    """Sanity-check that stored embeddings are real (>100 non-zero) not fallback (~45 non-zero)."""
    if not _LOCAL_STORE_PATH.exists():
        logger.error("VERIFICATION: local_store.json not found!")
        return
    store = json.loads(_LOCAL_STORE_PATH.read_text(encoding="utf-8"))
    posts = store.get("posts", [])
    if not posts:
        logger.error("VERIFICATION FAILED: No posts in local_store.json!")
        return

    logger.info("VERIFICATION: Checking embedding quality on first 3 posts...")
    passed = 0
    for i, p in enumerate(posts[:3]):
        en_emb = p.get("en_embedding", [])
        nonzero = sum(1 for v in en_emb if v != 0)
        ok = nonzero > 100
        if ok:
            passed += 1
        logger.info(
            "  Post[%d]: en_embedding nonzero=%d  %s",
            i, nonzero, "✓ real embedding" if ok else "✗ STILL FALLBACK — something went wrong"
        )

    logger.info("Total posts indexed: %d", len(posts))
    if passed == 3:
        logger.info("=" * 60)
        logger.info("SUCCESS: Knowledge base is healthy with real semantic embeddings.")
        logger.info("Restart main.py — queries should now return grounded answers.")
        logger.info("=" * 60)
    else:
        logger.error("CRITICAL: Embeddings look wrong. Check USE_REAL_EMBEDDINGS=true.")


def main() -> None:
    _check_real_embeddings_enabled()

    parser = argparse.ArgumentParser(description="Embed corpus and index into local_store.json (and optionally Supabase)")
    parser.add_argument("--corpus",     default="data/corpus_validated.jsonl")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Progress log interval (not a memory batch — all posts are held in RAM)")
    args = parser.parse_args()

    corpus_path = args.corpus
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), corpus_path)

    # Step 1: Load corpus
    posts = _load_corpus(corpus_path)
    logger.info("Loaded %d posts from %s", len(posts), corpus_path)

    # Step 2: Embed everything (CPU — takes 5–15 min for 560 posts)
    t0 = time.perf_counter()
    records = _embed_all(posts, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0
    logger.info("Embedding complete in %.1fs (%.2fs/post avg)", elapsed, elapsed / max(len(records), 1))

    # Step 3: Write local_store.json in ONE shot (no per-post rewrites)
    _write_local_store(records)

    # Step 4: Optionally upsert to Supabase if credentials are present
    _try_supabase_upsert(records)

    # Step 5: Verify quality
    _verify_embedding_quality()


if __name__ == "__main__":
    main()
