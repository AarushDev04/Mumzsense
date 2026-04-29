"""
Backward-compatible wrapper for the embedding/indexing entry point.
"""
from __future__ import annotations

from embed_and_index import main


if __name__ == "__main__":
    main()# Exploratory Data Analysis
# Analyzes corpus distributions and generates EDA report
cat > /home/claude/mumzsense/backend/data/embed_and_index.py << 'PYEOF'
"""
MumzSense v1 — Embedding & pgvector Indexing Pipeline (PRD §10.2)
Embeds all 560 corpus posts and upserts them into Supabase with HNSW index.

Usage:
    cd backend
    python data/embed_and_index.py --corpus data/corpus_validated.jsonl
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def embed_and_index(corpus_path: str, batch_size: int = 20):
    from agents.rag_agent import embed_text
    from db.supabase_client import insert_post

    posts = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    posts.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line: {e}")

    logger.info(f"Loaded {len(posts)} posts from {corpus_path}")

    total_inserted = 0
    for i in range(0, len(posts), batch_size):
        batch = posts[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} posts)...")

        for post in batch:
            lang = post.get("lang", "en")
            text = f"{post.get('situation', '')} {post.get('advice', '')}"

            # Embed with both models (PRD §7.2 — dual-column storage)
            en_embedding = embed_text(text, "en")
            ar_embedding = embed_text(text, "ar")

            record = {
                "post_id": post["post_id"],
                "baby_age_weeks": post.get("baby_age_weeks", 0),
                "stage": post.get("stage", ""),
                "topic": post.get("topic", ""),
                "urgency": post.get("urgency", ""),
                "situation": post.get("situation", ""),
                "advice": post.get("advice", ""),
                "outcome": post.get("outcome"),
                "trust_score": float(post.get("trust_score", 0.75)),
                "lang": lang,
                "en_embedding": en_embedding,
                "ar_embedding": ar_embedding,
                "verified": bool(post.get("verified", False)),
            }

            success = await insert_post(record)
            if success:
                total_inserted += 1

        # Respect rate limits
        time.sleep(0.5)

    logger.info(f"✓ Indexed {total_inserted}/{len(posts)} posts in Supabase")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Embed corpus and index in pgvector")
    parser.add_argument("--corpus", default="data/corpus_validated.jsonl")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    corpus_path = args.corpus
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), corpus_path)

    asyncio.run(embed_and_index(corpus_path, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
PYEOF

cat > /home/claude/mumzsense/backend/data/eda.py << 'PYEOF'
"""
MumzSense v1 — Exploratory Data Analysis (PRD §5.2)
Generates eda_report.json and prints distribution summaries.

Usage:
    cd backend
    python data/eda.py --corpus data/corpus_validated.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_eda(corpus_path: str) -> dict:
    posts = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    posts.append(json.loads(line))
                except Exception:
                    pass

    logger.info(f"Loaded {len(posts)} posts")

    # ── Distributions ────────────────────────────────────────────────────────
    topic_counts = Counter(p["topic"] for p in posts)
    urgency_counts = Counter(p["urgency"] for p in posts)
    lang_counts = Counter(p["lang"] for p in posts)
    stage_counts = Counter(p["stage"] for p in posts)

    # ── Token length distribution ─────────────────────────────────────────────
    lengths_by_topic = defaultdict(list)
    for p in posts:
        text = f"{p.get('situation','')} {p.get('advice','')}".split()
        lengths_by_topic[p["topic"]].append(len(text))

    length_stats = {}
    for topic, lengths in lengths_by_topic.items():
        arr = np.array(lengths)
        length_stats[topic] = {
            "mean": round(float(arr.mean()), 1),
            "std": round(float(arr.std()), 1),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "p95": int(np.percentile(arr, 95)),
        }

    # ── Overall P95 token length (for BiLSTM sequence padding) ───────────────
    all_lengths = [
        len(f"{p.get('situation','')} {p.get('advice','')}".split())
        for p in posts
    ]
    p95_length = int(np.percentile(all_lengths, 95))

    # ── Trust score stats ─────────────────────────────────────────────────────
    trust_scores = [p.get("trust_score", 0.75) for p in posts]
    trust_arr = np.array(trust_scores)

    # ── Class balance check (PRD §4.4 — min 30 per class) ────────────────────
    class_balance_ok = all(v >= 30 for v in topic_counts.values())

    report = {
        "total_posts": len(posts),
        "distributions": {
            "topic": dict(topic_counts),
            "urgency": dict(urgency_counts),
            "lang": dict(lang_counts),
            "stage": dict(stage_counts),
        },
        "token_length": {
            "p95_overall": p95_length,
            "by_topic": length_stats,
        },
        "trust_score": {
            "mean": round(float(trust_arr.mean()), 3),
            "std": round(float(trust_arr.std()), 3),
            "min": round(float(trust_arr.min()), 3),
            "max": round(float(trust_arr.max()), 3),
        },
        "class_balance_ok": class_balance_ok,
        "min_class_count": min(topic_counts.values()),
    }

    logger.info(f"Topic distribution: {dict(topic_counts)}")
    logger.info(f"Urgency distribution: {dict(urgency_counts)}")
    logger.info(f"P95 token length: {p95_length}")
    logger.info(f"Trust score mean: {trust_arr.mean():.3f}")
    logger.info(f"Class balance OK (min≥30): {class_balance_ok}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus_validated.jsonl")
    args = parser.parse_args()

    corpus_path = args.corpus
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), corpus_path)

    report = run_eda(corpus_path)

    out_path = os.path.join(os.path.dirname(corpus_path), "eda_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"EDA report saved to {out_path}")


if __name__ == "__main__":
    main()
PYEOF
echo "eda.py and embed_and_index.py written"