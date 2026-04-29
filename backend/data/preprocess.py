# Data Preprocessing Pipeline
# Text cleaning and feature engineering
cat > /home/claude/mumzsense/backend/data/preprocess.py << 'PYEOF'
"""
MumzSense v1 — Data Preprocessing Pipeline (PRD §5)
Text cleaning and feature engineering for classifier training.
"""
from __future__ import annotations
import re
import json
import logging
from typing import List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MEDICAL_TERMS = {
    "fever", "temperature", "rash", "vomit", "diarrhea", "diarrhoea",
    "seizure", "convulsion", "unconscious", "breathing", "choking",
    "dehydrated", "jaundice", "fontanelle", "blood", "emergency",
    "hospital", "doctor", "paediatrician", "nurse", "medication",
    "حمى", "تشنج", "طارئ", "مستشفى", "طبيب",
}

URGENCY_SIGNALS = {
    "emergency", "not breathing", "turning blue", "seizure",
    "unconscious", "fainted", "blood", "911", "لا يتنفس", "طارئ",
}

TOPIC_LABELS = ["feeding", "sleep", "health", "development", "gear", "postpartum", "mental_health"]
URGENCY_LABELS = ["routine", "monitor", "seek-help"]


def clean_text(text: str, lang: str) -> str:
    """Full text cleaning pipeline (PRD §5.1)."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w.-]+@[\w.-]+", " ", text)
    text = re.sub(r"\+?[\d\s\-()]{7,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\n+", " ", text)

    if lang == "en":
        text = text.lower()
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
        }
        for k, v in contractions.items():
            text = text.replace(k, v)
    elif lang == "ar":
        text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
        text = re.sub(r"[أإآ]", "ا", text)
        text = text.replace("ة", "ه")

    return text.strip()


def engineer_features(text: str, lang: str) -> np.ndarray:
    """4-dim binary feature vector (PRD §5.3)."""
    tokens = text.lower().split()
    token_set = set(tokens)
    has_medical = float(bool(token_set & MEDICAL_TERMS))
    has_urgency = float(bool(token_set & URGENCY_SIGNALS))
    is_arabic = float(lang == "ar")
    n_tokens = len(tokens)
    length_bin = 0.0 if n_tokens < 30 else (1.0 if n_tokens < 80 else 2.0)
    return np.array([has_medical, has_urgency, is_arabic, length_bin], dtype=np.float32)


def prepare_record(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single corpus record into training-ready form.
    Returns dict with: text (cleaned), lang, topic, urgency, features.
    """
    lang = post.get("lang", "en")
    situation = post.get("situation", "")
    advice = post.get("advice", "")
    combined = f"{situation} {advice}"
    cleaned = clean_text(combined, lang)
    features = engineer_features(cleaned, lang)

    return {
        "post_id": post.get("post_id", ""),
        "text": cleaned,
        "raw_text": combined,
        "lang": lang,
        "topic": post.get("topic", ""),
        "urgency": post.get("urgency", ""),
        "topic_label": TOPIC_LABELS.index(post["topic"]) if post.get("topic") in TOPIC_LABELS else -1,
        "urgency_label": URGENCY_LABELS.index(post["urgency"]) if post.get("urgency") in URGENCY_LABELS else -1,
        "features": features.tolist(),
        "stage": post.get("stage", ""),
        "trust_score": post.get("trust_score", 0.75),
    }


def load_and_prepare_corpus(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load JSONL corpus and prepare all records."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                post = json.loads(line)
                record = prepare_record(post)
                if record["topic_label"] >= 0 and record["urgency_label"] >= 0:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Skipping malformed record: {e}")
    logger.info(f"Loaded and prepared {len(records)} records from {jsonl_path}")
    return records


def stratified_split(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Stratified train/val/test split on (topic × urgency) joint label (PRD §5.4)."""
    import random
    random.seed(seed)

    from collections import defaultdict
    strata: Dict[str, List] = defaultdict(list)
    for r in records:
        key = f"{r['topic']}_{r['urgency']}"
        strata[key].append(r)

    train, val, test = [], [], []
    for key, items in strata.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    logger.info(f"Split → train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
PYEOF
echo "preprocess.py written"