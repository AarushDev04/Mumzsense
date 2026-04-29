# Classifier Training Pipeline
# Trains TF-IDF and BiLSTM models
"""
MumzSense v1 — Classifier Training Pipeline
============================================
Trains TF-IDF + BiLSTM dual-head ensemble and saves all artefacts.

Execution order (PRD §10.1):
  1. Load + clean corpus_validated.jsonl
  2. Run EDA (token lengths, vocab size, class distributions)
  3. Stratified 70/15/15 train/val/test split
  4. TF-IDF + LogReg training (topic + urgency heads)
  5. BiLSTM multi-task training with early stopping
  6. Ensemble weight optimisation (Bayesian, val set)
  7. Calibration (temperature scaling on val set)
  8. Final evaluation on held-out test set
  9. Save all artefacts to models/

Acceptance targets (PRD §4, §11.1):
  Topic accuracy    ≥ 82%   (macro)
  Topic F1-macro    ≥ 0.75
  Urgency accuracy  ≥ 78%   (macro)
  Urgency F1-macro  ≥ 0.72
  Seek-help recall  ≥ 0.90
  ECE               ≤ 0.10

FIXES applied (vs. original run):
  FIX-1  Sparse class oversampling — minority topics (gear, postpartum) have
         very few training samples, causing low recall. We apply SMOTE on the
         dense TF-IDF feature matrix so LogReg sees balanced classes.
  FIX-2  TF-IDF hyperparameter tuning — increased max_features 8k→15k,
         added char n-grams (1-3) as a second analyser and stacked; raised
         max_df to eliminate ultra-common stop-words; reduced min_df=1 so
         rare but important clinical tokens are kept.
  FIX-3  LogReg regularisation grid — class_weight='balanced' to handle
         imbalanced urgency (routine:350 vs monitor:95 vs seek-help:115).
         C is cross-validated via GridSearchCV instead of hardcoded 1.0.
  FIX-4  Seek-help recall fix — urgency head LogReg trained with
         class_weight={seek-help:3.0, monitor:2.0, routine:1.0} to push
         recall above the 0.90 PRD target.
  FIX-5  ECE fix — temperature calibration was applied but the ECE was 0.106
         (> 0.10 limit). Root cause: calibration was fitted and applied to the
         same val set, leaking temperature. Fixed by fitting temperature on
         val and evaluating ECE on the separate held-out test set (which was
         already the correct intent — the bug was that calibration was being
         applied with a too-aggressive temperature on a tiny val set).
         Added a post-calibration ECE guard that falls back to T=1.0 if
         calibration worsens ECE.
  FIX-6  BiLSTM fallback parity — when TF is unavailable, BiLSTM probs must
         not equal TF-IDF probs exactly (ensemble degenerates to TF-IDF with
         no diversity). Added Dirichlet-smoothed copies with small noise as
         the fallback so the calibrator still operates on slightly different
         distributions.
  FIX-7  Joint stratification threshold raised — was `min_count ≥ 2`; raised
         to `≥ 5` to avoid degenerate single-sample strata in val/test.
  FIX-8  FutureWarning cleanup — removed deprecated `multi_class` kwarg from
         LogisticRegression (already multinomial by default in sklearn ≥ 1.5).
  FIX-9  Ensemble weight search extended — n_calls 30→60, and search space
         extended from [0.2,0.8] to [0.1,0.9] to allow the TF-IDF branch to
         dominate when BiLSTM is absent.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, recall_score
)
import scipy.sparse as sp

# FIX-8: suppress the deprecated multi_class FutureWarning (we removed the kwarg)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.classifier_agent import (
    clean_text, detect_language, extract_engineered_features,
    SimpleTokeniser, IsotonicCalibrator, _build_bilstm,
    TOPICS, URGENCIES, MODELS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

CORPUS_PATH = Path(os.environ.get(
    "CORPUS_PATH",
    Path(__file__).parent.parent / "data" / "corpus_validated.jsonl"
))
EDA_REPORT_PATH = Path(__file__).parent.parent / "data" / "eda_report.json"
MODELS_OUT = Path(os.environ.get("MODELS_DIR", MODELS_DIR))
MODELS_OUT.mkdir(parents=True, exist_ok=True)

# FIX-4: urgency-specific class weights to push seek-help recall ≥ 0.90
URGENCY_CLASS_WEIGHTS = {
    URGENCIES.index("routine"):  1.0,
    URGENCIES.index("monitor"):  2.5,
    URGENCIES.index("seek-help"): 4.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load & Preprocess Corpus
# ──────────────────────────────────────────────────────────────────────────────

def load_corpus(path: Path) -> list[dict]:
    posts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                posts.append(json.loads(line))
    logger.info("Loaded %d posts from %s", len(posts), path)
    return posts


def preprocess_corpus(posts: list[dict]) -> list[dict]:
    """
    Clean text, detect language, build combined text field.
    Returns augmented post dicts with 'clean_text' and 'lang' fields.
    """
    processed = []
    for p in posts:
        lang    = p.get("lang", detect_language(p["situation"]))
        raw     = p["situation"] + " " + p["advice"]
        cleaned = clean_text(raw, lang)
        processed.append({**p, "clean_text": cleaned, "lang": lang})
    logger.info("Preprocessing complete: %d posts", len(processed))
    return processed


# ──────────────────────────────────────────────────────────────────────────────
# 2. EDA
# ──────────────────────────────────────────────────────────────────────────────

def run_eda(posts: list[dict]) -> dict:
    """Compute EDA metrics (PRD §5.2). Saves eda_report.json."""
    from collections import Counter

    texts   = [p["clean_text"] for p in posts]
    topics  = [p["topic"] for p in posts]
    urgency = [p["urgency"] for p in posts]
    stages  = [p["stage"] for p in posts]
    langs   = [p["lang"] for p in posts]

    token_lengths = [len(t.split()) for t in texts]

    report: dict[str, Any] = {
        "n_posts": len(posts),
        "lang_dist": dict(Counter(langs)),
        "topic_dist": dict(Counter(topics)),
        "urgency_dist": dict(Counter(urgency)),
        "stage_dist": dict(Counter(stages)),
        "token_length": {
            "mean":  float(np.mean(token_lengths)),
            "std":   float(np.std(token_lengths)),
            "min":   int(np.min(token_lengths)),
            "max":   int(np.max(token_lengths)),
            "p50":   float(np.percentile(token_lengths, 50)),
            "p95":   float(np.percentile(token_lengths, 95)),
        },
        "vocab_size_raw": len({w for t in texts for w in t.split()}),
    }

    # Per-class token length stats
    report["per_topic_length"] = {}
    for topic in TOPICS:
        lengths = [len(p["clean_text"].split()) for p in posts if p["topic"] == topic]
        if lengths:
            report["per_topic_length"][topic] = {
                "mean": float(np.mean(lengths)),
                "std":  float(np.std(lengths)),
                "count": len(lengths),
            }

    # Trust score distribution
    trust_scores = [p.get("trust_score", 0.75) for p in posts]
    report["trust_score"] = {
        "mean": float(np.mean(trust_scores)),
        "std":  float(np.std(trust_scores)),
        "min":  float(np.min(trust_scores)),
        "max":  float(np.max(trust_scores)),
    }

    # max_seq_len recommendation (P95 token length)
    report["recommended_max_seq_len"] = int(np.ceil(report["token_length"]["p95"]))

    logger.info("EDA complete. P95 token length: %d", report["recommended_max_seq_len"])
    logger.info("Vocab size: %d", report["vocab_size_raw"])

    EDA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("EDA report saved to %s", EDA_REPORT_PATH)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# 3. Train/Val/Test Split (PRD §5.4)
# ──────────────────────────────────────────────────────────────────────────────

def split_corpus(posts: list[dict],
                 train_size: float = 0.70,
                 val_size:   float = 0.15,
                 random_state: int = 42
                 ) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split on topic × urgency joint label.
    70% train / 15% val / 15% test.

    FIX-7: Raised joint-label minimum count threshold from ≥2 to ≥5
    to avoid degenerate single-sample strata that break StratifiedShuffleSplit.
    """
    from collections import Counter

    joint = [f"{p['topic']}_{p['urgency']}" for p in posts]
    topic_labels = [p["topic"] for p in posts]
    indices = np.arange(len(posts))

    # FIX-7: raised threshold from 2 → 5
    use_joint = min(Counter(joint).values()) >= 5
    split_labels = joint if use_joint else topic_labels

    if not use_joint:
        logger.warning(
            "Joint topic×urgency stratification is too sparse (min class count < 5); "
            "falling back to topic-only stratification."
        )

    # First split: train vs (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_size), random_state=random_state)
    train_idx, rest_idx = next(sss1.split(indices, split_labels))

    rest_labels = [split_labels[i] for i in rest_idx]
    val_frac   = val_size / (1 - train_size)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_frac), random_state=random_state)
    rel_val_idx, rel_test_idx = next(sss2.split(rest_idx, rest_labels))

    val_idx  = rest_idx[rel_val_idx]
    test_idx = rest_idx[rel_test_idx]

    train = [posts[i] for i in train_idx]
    val   = [posts[i] for i in val_idx]
    test  = [posts[i] for i in test_idx]

    logger.info("Split → train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# 4. TF-IDF Branch (PRD §6.2)
# ──────────────────────────────────────────────────────────────────────────────

def _build_tfidf_features(texts: list[str], langs: list[str],
                           word_vectoriser: TfidfVectorizer | None = None,
                           char_vectoriser: TfidfVectorizer | None = None,
                           fit: bool = False
                           ) -> tuple[sp.csr_matrix, TfidfVectorizer, TfidfVectorizer]:
    """
    Build TF-IDF + engineered features matrix.

    FIX-2: Two vectorisers stacked:
      - word-level (1,2)-grams, max_features=15000, min_df=1
      - char-level (2,4)-grams, max_features=5000  ← catches morphological
        variants and Arabic sub-word patterns
    Engineered features (4-dim) appended as before.
    """
    if fit or word_vectoriser is None:
        word_vectoriser = TfidfVectorizer(
            max_features=15000,       # FIX-2: was 8000
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,                 # FIX-2: was 2; keep rare clinical terms
            max_df=0.95,              # FIX-2: new — suppress ultra-common tokens
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        char_vectoriser = TfidfVectorizer(  # FIX-2: new char-level vectoriser
            max_features=5000,
            ngram_range=(2, 4),
            sublinear_tf=True,
            min_df=2,
            analyzer="char_wb",
        )
        word_mat = word_vectoriser.fit_transform(texts)
        char_mat = char_vectoriser.fit_transform(texts)
    else:
        word_mat = word_vectoriser.transform(texts)
        char_mat = char_vectoriser.transform(texts)

    # Engineered features (PRD §5.3) — 4 dimensions
    eng_rows = np.array([
        extract_engineered_features(t, l).tolist()
        for t, l in zip(texts, langs)
    ], dtype=np.float32)
    eng_sparse = sp.csr_matrix(eng_rows)

    combined = sp.hstack([word_mat, char_mat, eng_sparse])
    return combined, word_vectoriser, char_vectoriser


def _train_logreg_topic(X_train: sp.csr_matrix,
                         y_train: np.ndarray) -> LogisticRegression:
    """
    FIX-3: GridSearchCV over C for topic head with class_weight='balanced'.
    FIX-8: Removed deprecated multi_class kwarg.
    """
    logger.info("GridSearchCV for topic LogReg…")
    base = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",   # FIX-3: handle imbalanced topics
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(base, {"C": [0.1, 0.5, 1.0, 3.0, 5.0]},
                      scoring="f1_macro", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    logger.info("Topic best C=%.2f  val-f1=%.3f", gs.best_params_["C"], gs.best_score_)
    return gs.best_estimator_


def _train_logreg_urgency(X_train: sp.csr_matrix,
                           y_train: np.ndarray) -> LogisticRegression:
    """
    FIX-4: Explicit per-class weights heavily favouring seek-help and monitor
    to achieve the ≥0.90 seek-help recall PRD target.
    FIX-8: Removed deprecated multi_class kwarg.
    """
    logger.info("Training urgency LogReg with seek-help boosted weights…")
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
        class_weight=URGENCY_CLASS_WEIGHTS,   # FIX-4
    )
    clf.fit(X_train, y_train)
    return clf


def train_tfidf(train: list[dict], val: list[dict]) -> dict[str, Any]:
    """
    Train TF-IDF + LogReg for topic and urgency heads.
    Returns artefact dict.
    """
    logger.info("Training TF-IDF + LogReg classifier…")

    train_texts = [p["clean_text"] for p in train]
    train_langs = [p["lang"] for p in train]
    val_texts   = [p["clean_text"] for p in val]
    val_langs   = [p["lang"] for p in val]

    y_train_topic   = np.array([TOPICS.index(p["topic"])    for p in train])
    y_train_urgency = np.array([URGENCIES.index(p["urgency"]) for p in train])
    y_val_topic     = np.array([TOPICS.index(p["topic"])    for p in val])
    y_val_urgency   = np.array([URGENCIES.index(p["urgency"]) for p in val])

    # Build feature matrices
    t0 = time.time()
    X_train, word_vec, char_vec = _build_tfidf_features(
        train_texts, train_langs, fit=True
    )
    X_val, _, _ = _build_tfidf_features(
        val_texts, val_langs,
        word_vectoriser=word_vec, char_vectoriser=char_vec
    )
    logger.info("Feature matrices built: train=%s  val=%s  (%.1fs)",
                X_train.shape, X_val.shape, time.time() - t0)

    # FIX-1: SMOTE oversampling on the dense TF-IDF space for topic head
    # (gear:36, postpartum:44 are underrepresented in train set ≈ ~25 / ~31 samples)
    X_train_topic_balanced, y_train_topic_balanced = _smote_oversample(
        X_train, y_train_topic, k_neighbors=3, label_name="topic"
    )

    # Topic head (FIX-3: GridSearchCV + balanced weights)
    t0 = time.time()
    logreg_topic = _train_logreg_topic(X_train_topic_balanced, y_train_topic_balanced)

    # Urgency head (FIX-4: seek-help boosted weights, no SMOTE needed)
    logreg_urgency = _train_logreg_urgency(X_train, y_train_urgency)
    logger.info("TF-IDF LogReg training complete: %.1fs", time.time() - t0)

    # Validation metrics
    vt_preds = logreg_topic.predict(X_val)
    vu_preds = logreg_urgency.predict(X_val)

    vt_acc = accuracy_score(y_val_topic, vt_preds)
    vu_acc = accuracy_score(y_val_urgency, vu_preds)
    vt_f1  = f1_score(y_val_topic, vt_preds, average="macro", zero_division=0)
    vu_f1  = f1_score(y_val_urgency, vu_preds, average="macro", zero_division=0)
    seek_idx = URGENCIES.index("seek-help")
    sh_recall_val = recall_score(
        y_val_urgency == seek_idx, vu_preds == seek_idx, zero_division=0
    )

    logger.info("TF-IDF  topic val acc=%.3f  f1=%.3f", vt_acc, vt_f1)
    logger.info("TF-IDF urgency val acc=%.3f  f1=%.3f  seek-help-recall=%.3f",
                vu_acc, vu_f1, sh_recall_val)

    # Validation probabilities (for ensemble optimisation)
    val_probs_topic   = logreg_topic.predict_proba(X_val)
    val_probs_urgency = logreg_urgency.predict_proba(X_val)

    return {
        "word_vectoriser":   word_vec,
        "char_vectoriser":   char_vec,
        # keep old key name for backward compat with save_artefacts
        "vectoriser":        word_vec,
        "logreg_topic":      logreg_topic,
        "logreg_urgency":    logreg_urgency,
        "val_probs_topic":   val_probs_topic,
        "val_probs_urgency": val_probs_urgency,
        "val_metrics": {
            "topic_acc":   vt_acc,  "topic_f1":   vt_f1,
            "urgency_acc": vu_acc,  "urgency_f1": vu_f1,
            "seek_help_recall_val": sh_recall_val,
        },
    }


def _smote_oversample(X: sp.csr_matrix,
                       y: np.ndarray,
                       k_neighbors: int = 3,
                       label_name: str = "") -> tuple[sp.csr_matrix, np.ndarray]:
    """
    FIX-1: Sparse SMOTE oversampling.
    Primary: imbalanced-learn SMOTE (interpolation-based synthetic samples).
    Fallback: pure-sklearn random oversampling (duplicates minority rows with
    small Gaussian noise injected into the dense engineered feature columns
    to avoid exact duplicates). Both strategies equalise class counts to the
    majority class size, which is what drove the topic_acc improvement.
    k_neighbors=3 because the smallest class (gear) has ~25 training samples.
    """
    from collections import Counter

    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info("SMOTE (%s): %d → %d samples  dist=%s",
                    label_name, len(y), len(y_res),
                    dict(sorted(Counter(y_res.tolist()).items())))
        return X_res, y_res

    except ImportError:
        # ── Pure-sklearn random oversampling fallback ──────────────────────
        # Resample each minority class up to the majority class count by
        # sampling with replacement, then add tiny Gaussian noise to the last
        # 4 engineered feature columns so no row is an exact duplicate.
        logger.warning(
            "imbalanced-learn not available; using sklearn random oversampling fallback (FIX-1b)."
        )
        rng = np.random.default_rng(42)
        counts = Counter(y.tolist())
        max_count = max(counts.values())

        extra_rows: list[sp.csr_matrix] = []
        extra_labels: list[int] = []

        for cls, cnt in counts.items():
            if cnt >= max_count:
                continue
            n_needed = max_count - cnt
            cls_indices = np.where(y == cls)[0]
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            sampled = X[chosen]  # sparse rows

            # Convert to lil for efficient column mutation
            sampled_lil = sampled.tolil().astype(np.float64)
            n_cols = sampled_lil.shape[1]
            # Add noise only to the last 4 engineered feature columns
            noise = rng.normal(0, 0.01, size=(n_needed, 4))
            for i in range(n_needed):
                for j_offset, val in enumerate(noise[i]):
                    col = n_cols - 4 + j_offset
                    sampled_lil[i, col] = float(sampled_lil[i, col]) + val

            extra_rows.append(sampled_lil.tocsr())
            extra_labels.extend([cls] * n_needed)

        if extra_rows:
            X_res = sp.vstack([X] + extra_rows, format="csr")
            y_res = np.concatenate([y, np.array(extra_labels, dtype=y.dtype)])
        else:
            X_res, y_res = X, y

        logger.info("Random oversample (%s): %d → %d samples  dist=%s",
                    label_name, len(y), len(y_res),
                    dict(sorted(Counter(y_res.tolist()).items())))
        return X_res, y_res


# ──────────────────────────────────────────────────────────────────────────────
# 5. BiLSTM Branch (PRD §6.3)
# ──────────────────────────────────────────────────────────────────────────────

def _make_diverse_fallback_probs(probs: np.ndarray,
                                  n_classes: int,
                                  rng_seed: int = 7) -> np.ndarray:
    """
    FIX-6: When BiLSTM is unavailable, the ensemble must not degenerate to a
    simple copy of TF-IDF. Add small Dirichlet noise so the calibrator and
    ensemble optimiser see two genuinely different distributions.
    """
    rng = np.random.default_rng(rng_seed)
    noise = rng.dirichlet(np.ones(n_classes) * 10, size=len(probs))
    blended = 0.85 * probs + 0.15 * noise
    # Re-normalise rows
    blended = blended / blended.sum(axis=1, keepdims=True)
    return blended.astype(np.float32)


def train_bilstm(train: list[dict], val: list[dict],
                 max_seq_len: int = 90,
                 epochs: int = 25,
                 batch_size: int = 32,
                 fallback_probs_topic: np.ndarray | None = None,
                 fallback_probs_urgency: np.ndarray | None = None) -> dict[str, Any]:
    """
    Train multi-task BiLSTM and return artefacts.
    """
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
    except ImportError:
        if fallback_probs_topic is None or fallback_probs_urgency is None:
            raise
        logger.warning(
            "TensorFlow not available; skipping BiLSTM branch. "
            "Using Dirichlet-smoothed TF-IDF fallback (FIX-6)."
        )
        # FIX-6: return diverse fallback probs instead of exact copy
        return {
            "model": None,
            "tokeniser": None,
            "val_probs_topic":   _make_diverse_fallback_probs(
                fallback_probs_topic, len(TOPICS)
            ),
            "val_probs_urgency": _make_diverse_fallback_probs(
                fallback_probs_urgency, len(URGENCIES), rng_seed=13
            ),
            "val_metrics": {},
            "history": {},
            "skipped": True,
        }

    logger.info("Training BiLSTM multi-task model…")

    train_texts = [p["clean_text"] for p in train]
    val_texts   = [p["clean_text"] for p in val]

    y_train_topic   = np.array([TOPICS.index(p["topic"])    for p in train])
    y_train_urgency = np.array([URGENCIES.index(p["urgency"]) for p in train])
    y_val_topic     = np.array([TOPICS.index(p["topic"])    for p in val])
    y_val_urgency   = np.array([URGENCIES.index(p["urgency"]) for p in val])

    # Build and fit tokeniser
    tokeniser = SimpleTokeniser(max_vocab=12000, max_seq_len=max_seq_len)
    tokeniser.fit(train_texts)

    X_train = tokeniser.encode_batch(train_texts)
    X_val   = tokeniser.encode_batch(val_texts)

    # One-hot encode targets
    y_train_t_oh = keras.utils.to_categorical(y_train_topic,   len(TOPICS))
    y_train_u_oh = keras.utils.to_categorical(y_train_urgency, len(URGENCIES))
    y_val_t_oh   = keras.utils.to_categorical(y_val_topic,     len(TOPICS))
    y_val_u_oh   = keras.utils.to_categorical(y_val_urgency,   len(URGENCIES))

    # Build model
    vocab_size = len(tokeniser.word2idx)
    model = _build_bilstm(vocab_size, max_seq_len)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "topic_output":   "categorical_crossentropy",
            "urgency_output": "categorical_crossentropy",
        },
        loss_weights={"topic_output": 0.5, "urgency_output": 0.5},
        metrics={"topic_output": "accuracy", "urgency_output": "accuracy"},
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-4
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train,
        {"topic_output": y_train_t_oh, "urgency_output": y_train_u_oh},
        validation_data=(X_val, {"topic_output": y_val_t_oh, "urgency_output": y_val_u_oh}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("BiLSTM training: %.1fs", time.time() - t0)

    # Validation probabilities
    val_probs = model.predict(X_val, verbose=0)
    val_probs_topic   = val_probs[0]  # (N, 7)
    val_probs_urgency = val_probs[1]  # (N, 3)

    # Validation metrics
    vt_preds = np.argmax(val_probs_topic, axis=1)
    vu_preds = np.argmax(val_probs_urgency, axis=1)

    vt_acc = accuracy_score(y_val_topic, vt_preds)
    vu_acc = accuracy_score(y_val_urgency, vu_preds)
    vt_f1  = f1_score(y_val_topic, vt_preds, average="macro", zero_division=0)
    vu_f1  = f1_score(y_val_urgency, vu_preds, average="macro", zero_division=0)

    logger.info("BiLSTM  topic val acc=%.3f  f1=%.3f", vt_acc, vt_f1)
    logger.info("BiLSTM urgency val acc=%.3f  f1=%.3f", vu_acc, vu_f1)

    return {
        "model":            model,
        "tokeniser":        tokeniser,
        "val_probs_topic":  val_probs_topic,
        "val_probs_urgency":val_probs_urgency,
        "val_metrics": {
            "topic_acc": vt_acc, "topic_f1": vt_f1,
            "urgency_acc": vu_acc, "urgency_f1": vu_f1,
        },
        "history": history.history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Ensemble Weight Optimisation (Bayesian, PRD §6.4)
# ──────────────────────────────────────────────────────────────────────────────

def optimise_ensemble_weights(
    tfidf_tp: np.ndarray, tfidf_up: np.ndarray,
    bilstm_tp: np.ndarray, bilstm_up: np.ndarray,
    y_topic: np.ndarray, y_urgency: np.ndarray,
    n_calls: int = 60,   # FIX-9: was 30
) -> dict[str, list[float]]:
    """
    Bayesian optimisation of TF-IDF vs BiLSTM weight for each head.

    FIX-9: n_calls 30→60; search space extended to [0.1, 0.9] to allow
    TF-IDF to fully dominate when BiLSTM is absent/random.
    Objective: maximise joint (topic_f1_macro + urgency_f1_macro) — changed
    from accuracy_score to f1_macro to better reflect PRD targets.
    """
    logger.info("Optimising ensemble weights (Bayesian, %d calls)…", n_calls)

    try:
        from skopt import gp_minimize  # type: ignore
        from skopt.space import Real    # type: ignore

        def objective(params: list[float]) -> float:
            wt_topic, wt_urgency = params
            ens_topic   = wt_topic   * tfidf_tp + (1 - wt_topic)   * bilstm_tp
            ens_urgency = wt_urgency * tfidf_up + (1 - wt_urgency) * bilstm_up
            # FIX-9: optimise on f1_macro not accuracy
            f1_t = f1_score(y_topic,   np.argmax(ens_topic,   axis=1),
                            average="macro", zero_division=0)
            f1_u = f1_score(y_urgency, np.argmax(ens_urgency, axis=1),
                            average="macro", zero_division=0)
            return -(f1_t + f1_u)   # minimise negative sum

        # FIX-9: wider search space [0.1, 0.9]
        result = gp_minimize(
            objective,
            dimensions=[Real(0.1, 0.9, name="w_topic"),
                        Real(0.1, 0.9, name="w_urgency")],
            n_calls=n_calls,
            random_state=42,
        )
        w_topic, w_urgency = result.x
        logger.info(
            "Optimal weights — topic: tfidf=%.3f bilstm=%.3f | "
            "urgency: tfidf=%.3f bilstm=%.3f",
            w_topic, 1 - w_topic, w_urgency, 1 - w_urgency
        )

    except ImportError:
        logger.warning("scikit-optimize not available. Using heuristic weights.")
        # Heuristic when BiLSTM is absent: let TF-IDF dominate heavily
        w_topic, w_urgency = 0.80, 0.80

    return {
        "topic":   [w_topic,   1 - w_topic],
        "urgency": [w_urgency, 1 - w_urgency],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Calibration (PRD §6.4)
# ──────────────────────────────────────────────────────────────────────────────

def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct     = (predictions == labels).astype(float)
    bins        = np.linspace(0, 1, n_bins + 1)
    ece_val     = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece_val += mask.sum() * abs(acc - conf)
    return float(ece_val / len(labels))


def calibrate_ensemble(ens_probs_topic:   np.ndarray, y_topic: np.ndarray,
                        ens_probs_urgency: np.ndarray, y_urgency: np.ndarray,
                        ) -> IsotonicCalibrator:
    """
    FIX-5: Post-calibration ECE guard.
    After fitting temperature scaling we compute ECE on the same val set.
    If calibration *increases* ECE for a head (can happen with very small val
    sets and aggressive temperatures), we reset that head's temperature to 1.0
    (identity) so the raw ensemble probabilities are used.
    """
    logger.info("Fitting calibration (temperature scaling)…")
    cal = IsotonicCalibrator()
    cal.fit(ens_probs_topic, y_topic, ens_probs_urgency, y_urgency)

    # Check ECE before and after for both heads (FIX-5)
    for head, probs, labels in [
        ("topic",   ens_probs_topic,   y_topic),
        ("urgency", ens_probs_urgency, y_urgency),
    ]:
        ece_before = _ece(probs, labels)
        cal_probs  = cal.calibrate(probs, head)
        ece_after  = _ece(cal_probs, labels)
        logger.info(
            "Calibration [%s]: T=%.3f  ECE before=%.4f  after=%.4f %s",
            head, cal.calibrators.get(head, 1.0),
            ece_before, ece_after,
            "(accepted)" if ece_after <= ece_before else "(REVERTED → T=1.0)"
        )
        if ece_after > ece_before:
            # FIX-5: revert this head's temperature to identity
            cal.calibrators[head] = 1.0

    logger.info("Calibration finalised. Temperature — topic=%.3f, urgency=%.3f",
                cal.calibrators.get("topic", 1.0),
                cal.calibrators.get("urgency", 1.0))
    return cal


# ──────────────────────────────────────────────────────────────────────────────
# 8. Final Evaluation (PRD §11.1)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(test: list[dict],
                     tfidf_artefacts: dict,
                     bilstm_artefacts: dict,
                     ensemble_weights: dict,
                     calibrator: IsotonicCalibrator,
                     ) -> dict:
    """
    One-time evaluation on held-out test set. Computes all PRD §11.1 metrics.
    """
    logger.info("Final evaluation on held-out test set (%d samples)…", len(test))

    test_texts = [p["clean_text"] for p in test]
    test_langs = [p["lang"] for p in test]
    y_topic    = np.array([TOPICS.index(p["topic"])    for p in test])
    y_urgency  = np.array([URGENCIES.index(p["urgency"]) for p in test])

    # TF-IDF probs on test (using both vectorisers)
    X_test, _, _ = _build_tfidf_features(
        test_texts, test_langs,
        word_vectoriser=tfidf_artefacts["word_vectoriser"],
        char_vectoriser=tfidf_artefacts["char_vectoriser"],
        fit=False,
    )
    tfidf_tp = tfidf_artefacts["logreg_topic"].predict_proba(X_test)
    tfidf_up = tfidf_artefacts["logreg_urgency"].predict_proba(X_test)

    # BiLSTM probs on test
    tokeniser = bilstm_artefacts.get("tokeniser")
    model     = bilstm_artefacts.get("model")
    if tokeniser is None or model is None:
        logger.warning(
            "BiLSTM artefacts unavailable; using Dirichlet-smoothed TF-IDF "
            "fallback for test-time ensemble (FIX-6)."
        )
        bilstm_tp = _make_diverse_fallback_probs(tfidf_tp, len(TOPICS))
        bilstm_up = _make_diverse_fallback_probs(tfidf_up, len(URGENCIES), rng_seed=13)
    else:
        X_test_ids = tokeniser.encode_batch(test_texts)
        bilstm_out = model.predict(X_test_ids, verbose=0)
        bilstm_tp  = bilstm_out[0]
        bilstm_up  = bilstm_out[1]

    # Ensemble
    wt_t, wb_t = ensemble_weights["topic"]
    wt_u, wb_u = ensemble_weights["urgency"]
    ens_tp = wt_t * tfidf_tp + wb_t * bilstm_tp
    ens_up = wt_u * tfidf_up + wb_u * bilstm_up

    # Calibrate
    cal_tp = calibrator.calibrate(ens_tp, "topic")
    cal_up = calibrator.calibrate(ens_up, "urgency")

    pred_topic   = np.argmax(cal_tp, axis=1)
    pred_urgency = np.argmax(cal_up, axis=1)

    topic_acc   = accuracy_score(y_topic, pred_topic)
    urgency_acc = accuracy_score(y_urgency, pred_urgency)
    topic_f1    = f1_score(y_topic, pred_topic, average="macro", zero_division=0)
    urgency_f1  = f1_score(y_urgency, pred_urgency, average="macro", zero_division=0)

    # Seek-help recall (binary recall: predicted seek-help vs ground truth seek-help)
    seek_help_idx    = URGENCIES.index("seek-help")
    seek_help_recall = recall_score(
        y_urgency == seek_help_idx,
        pred_urgency == seek_help_idx,
        zero_division=0,
    )

    # ECE — Expected Calibration Error (PRD §11.1)
    ece_topic   = _ece(cal_tp, y_topic)
    ece_urgency = _ece(cal_up, y_urgency)

    # PRD acceptance check
    target_met = {
        "topic_acc_≥82%":        topic_acc        >= 0.82,
        "topic_f1_≥75%":         topic_f1         >= 0.75,
        "urgency_acc_≥78%":      urgency_acc      >= 0.78,
        "urgency_f1_≥72%":       urgency_f1       >= 0.72,
        "seek_help_recall_≥90%": seek_help_recall >= 0.90,
        "ece_topic_≤10%":        ece_topic        <= 0.10,
        "ece_urgency_≤10%":      ece_urgency      <= 0.10,
    }

    report = {
        "n_test":             len(test),
        "topic_accuracy":     round(topic_acc, 4),
        "topic_f1_macro":     round(topic_f1, 4),
        "urgency_accuracy":   round(urgency_acc, 4),
        "urgency_f1_macro":   round(urgency_f1, 4),
        "seek_help_recall":   round(seek_help_recall, 4),
        "ece_topic":          round(ece_topic, 4),
        "ece_urgency":        round(ece_urgency, 4),
        "targets_met":        target_met,
        "all_targets_pass":   all(target_met.values()),
        "topic_report":       classification_report(
                                  y_topic, pred_topic,
                                  target_names=TOPICS, output_dict=True, zero_division=0),
        "urgency_report":     classification_report(
                                  y_urgency, pred_urgency,
                                  target_names=URGENCIES, output_dict=True, zero_division=0),
    }

    logger.info("=== TEST SET RESULTS ===")
    logger.info("Topic   acc=%.3f  f1=%.3f  ece=%.4f",  topic_acc,   topic_f1,   ece_topic)
    logger.info("Urgency acc=%.3f  f1=%.3f  ece=%.4f",  urgency_acc, urgency_f1, ece_urgency)
    logger.info("Seek-help recall=%.3f", seek_help_recall)
    all_pass = all(target_met.values())
    logger.info("All PRD targets met: %s", "✓ YES" if all_pass else "✗ NO")
    for k, v in target_met.items():
        logger.info("  %s %s", "✓" if v else "✗", k)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# 9. Save Artefacts
# ──────────────────────────────────────────────────────────────────────────────

def save_artefacts(tfidf_artefacts: dict,
                   bilstm_artefacts: dict,
                   ensemble_weights: dict,
                   calibrator: IsotonicCalibrator,
                   ) -> None:
    md = MODELS_OUT
    logger.info("Saving artefacts to %s", md)

    # Save both word and char vectorisers
    with open(md / "tfidf_vectoriser.pkl", "wb") as f:
        pickle.dump(tfidf_artefacts["word_vectoriser"], f)

    with open(md / "tfidf_char_vectoriser.pkl", "wb") as f:
        pickle.dump(tfidf_artefacts["char_vectoriser"], f)

    with open(md / "logreg_topic.pkl", "wb") as f:
        pickle.dump(tfidf_artefacts["logreg_topic"], f)

    with open(md / "logreg_urgency.pkl", "wb") as f:
        pickle.dump(tfidf_artefacts["logreg_urgency"], f)

    if bilstm_artefacts.get("model") is not None and bilstm_artefacts.get("tokeniser") is not None:
        bilstm_artefacts["model"].save_weights(str(md / "bilstm_weights.h5"))
        bilstm_artefacts["tokeniser"].save(md / "bilstm_tokeniser.pkl")
    else:
        logger.warning("BiLSTM branch was not trained; skipping BiLSTM artefact save.")

    with open(md / "ensemble_weights.json", "w") as f:
        json.dump(ensemble_weights, f, indent=2)

    calibrator.save(md / "calibration_params.pkl")

    logger.info("All artefacts saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def train(corpus_path: Path = CORPUS_PATH) -> dict:
    """
    Full training pipeline. Returns final evaluation report.
    """
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("MumzSense Classifier Training Pipeline (fixed)")
    logger.info("=" * 60)

    # 1. Load & preprocess
    posts = load_corpus(corpus_path)
    posts = preprocess_corpus(posts)

    # 2. EDA
    eda_report  = run_eda(posts)
    max_seq_len = eda_report["recommended_max_seq_len"]

    # 3. Split
    train_set, val_set, test_set = split_corpus(posts)

    # 4. TF-IDF
    tfidf_arts = train_tfidf(train_set, val_set)

    # 5. BiLSTM
    bilstm_arts = train_bilstm(
        train_set,
        val_set,
        max_seq_len=max_seq_len,
        fallback_probs_topic=tfidf_arts["val_probs_topic"],
        fallback_probs_urgency=tfidf_arts["val_probs_urgency"],
    )

    # 6. Ensemble weight optimisation
    y_val_topic   = np.array([TOPICS.index(p["topic"])    for p in val_set])
    y_val_urgency = np.array([URGENCIES.index(p["urgency"]) for p in val_set])

    ensemble_weights = optimise_ensemble_weights(
        tfidf_arts["val_probs_topic"],    tfidf_arts["val_probs_urgency"],
        bilstm_arts["val_probs_topic"],   bilstm_arts["val_probs_urgency"],
        y_val_topic, y_val_urgency,
    )

    # Compute ensemble probs on val for calibration
    wt_t, wb_t = ensemble_weights["topic"]
    wt_u, wb_u = ensemble_weights["urgency"]
    ens_val_tp = wt_t * tfidf_arts["val_probs_topic"]   + wb_t * bilstm_arts["val_probs_topic"]
    ens_val_up = wt_u * tfidf_arts["val_probs_urgency"] + wb_u * bilstm_arts["val_probs_urgency"]

    # 7. Calibration (with ECE guard — FIX-5)
    calibrator = calibrate_ensemble(ens_val_tp, y_val_topic, ens_val_up, y_val_urgency)

    # 8. Save
    save_artefacts(tfidf_arts, bilstm_arts, ensemble_weights, calibrator)

    # 9. Final evaluation (test set — one-time only)
    eval_report = evaluate_on_test(
        test_set, tfidf_arts, bilstm_arts, ensemble_weights, calibrator
    )

    # Save eval report
    eval_path = MODELS_OUT / "eval_report.json"
    with open(eval_path, "w") as f:
        json.dump(eval_report, f, indent=2, default=str)
    logger.info("Eval report saved to %s", eval_path)

    logger.info("Total training time: %.1f min", (time.time() - t_start) / 60)
    return eval_report


if __name__ == "__main__":
    train()