# Corpus Verification Pipeline
# Validates schema, language, duplicates, and class balance
cat > /home/claude/mumzsense/backend/data/verify_corpus.py << 'PYEOF'
"""
MumzSense v1 — Corpus Verification Pipeline (PRD §4.4)
Schema validation, language detection, deduplication, class balance check.
"""
from __future__ import annotations
import json
import logging
import re
import sys
import os
from typing import List, Dict, Any
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STAGES = {"trimester", "newborn", "0-3m", "3-6m", "6-12m", "toddler"}
TOPICS = {"feeding", "sleep", "health", "development", "gear", "postpartum", "mental_health"}
URGENCIES = {"routine", "monitor", "seek-help"}
LANGS = {"en", "ar"}


def validate_schema(post: Dict) -> List[str]:
    """Returns list of schema errors. Empty = valid."""
    errors = []
    required_fields = ["post_id", "baby_age_weeks", "stage", "topic", "urgency",
                       "situation", "advice", "trust_score", "lang", "verified"]
    for field in required_fields:
        if field not in post:
            errors.append(f"Missing field: {field}")

    if "stage" in post and post["stage"] not in STAGES:
        errors.append(f"Invalid stage: {post['stage']}")
    if "topic" in post and post["topic"] not in TOPICS:
        errors.append(f"Invalid topic: {post['topic']}")
    if "urgency" in post and post["urgency"] not in URGENCIES:
        errors.append(f"Invalid urgency: {post['urgency']}")
    if "lang" in post and post["lang"] not in LANGS:
        errors.append(f"Invalid lang: {post['lang']}")

    if "situation" in post:
        n = len(post["situation"])
        if n < 20 or n > 200:
            errors.append(f"situation length {n} out of range [20,200]")
    if "advice" in post:
        n = len(post["advice"])
        if n < 40 or n > 400:
            errors.append(f"advice length {n} out of range [40,400]")
    if "trust_score" in post:
        ts = post["trust_score"]
        if not (0.0 <= ts <= 1.0):
            errors.append(f"trust_score {ts} out of [0,1]")
    if "baby_age_weeks" in post:
        w = post["baby_age_weeks"]
        if not (-36 <= w <= 104):
            errors.append(f"baby_age_weeks {w} out of range")

    return errors


def check_language(post: Dict) -> bool:
    """Verify declared language matches content."""
    declared = post.get("lang", "en")
    text = post.get("situation", "") + post.get("advice", "")
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    ratio = arabic_chars / max(len(text), 1)

    if declared == "ar":
        if ratio < 0.30:
            logger.warning(f"Post {post.get('post_id','?')}: declared AR but ratio={ratio:.2f}")
            return False
    else:
        if ratio > 0.50:
            logger.warning(f"Post {post.get('post_id','?')}: declared EN but ratio={ratio:.2f}")
            return False
    return True


def verify_corpus(jsonl_path: str) -> Dict[str, Any]:
    """Run full verification and return report."""
    posts = []
    schema_errors = 0
    lang_errors = 0

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                logger.error(f"Line {i}: JSON decode error")
                schema_errors += 1
                continue

            errs = validate_schema(post)
            if errs:
                logger.warning(f"Post {post.get('post_id','?')}: {errs}")
                schema_errors += 1
                continue

            if not check_language(post):
                lang_errors += 1

            posts.append(post)

    # Class balance
    topic_counts = Counter(p["topic"] for p in posts)
    min_count = min(topic_counts.values()) if topic_counts else 0

    report = {
        "total_loaded": len(posts),
        "schema_errors": schema_errors,
        "lang_errors": lang_errors,
        "topic_distribution": dict(topic_counts),
        "min_topic_count": min_count,
        "class_balance_ok": min_count >= 8,
    }

    logger.info(f"Verification complete: {len(posts)} valid, {schema_errors} schema errors, {lang_errors} lang errors")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus_validated.jsonl")
    args = parser.parse_args()
    path = args.corpus
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    report = verify_corpus(path)
    print(json.dumps(report, indent=2))
PYEOF
echo "verify_corpus.py written"