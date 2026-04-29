# Classifier Agent: TF-IDF + BiLSTM ensemble
# Classifies queries into topic and urgency categories
"""
MumzSense v1 — Classifier Agent
================================
Ensemble of TF-IDF + BiLSTM for dual-head classification:
  • topic   → 7-class (feeding, sleep, health, development, gear, postpartum, mental_health)
  • urgency → 3-class (routine, monitor, seek-help)

Architecture (PRD §6):
  TF-IDF + LogReg baseline  ─┐
                              ├─► weighted soft-vote ─► Platt-calibrated output
  BiLSTM multi-task model   ─┘

Gate A (Phase 2 MADRL): BiLSTM 256-dim hidden state is logged to feedback_log
for future PPO policy head attachment.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _install_numpy_pickle_compat() -> None:
    """
    Older pickles in this repo reference numpy._core, while newer NumPy
    exposes the implementation under numpy.core. Register a small alias so
    those pickles can be loaded without retraining the model artefacts.
    """
    try:
        import numpy.core as numpy_core  # type: ignore
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)

    for submodule_name in (
        "multiarray",
        "_multiarray_umath",
        "numeric",
        "umath",
        "overrides",
        "shape_base",
        "defchararray",
        "records",
    ):
        try:
            module = __import__(f"numpy.core.{submodule_name}", fromlist=["*"])
            sys.modules.setdefault(f"numpy._core.{submodule_name}", module)
        except Exception:
            continue

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TOPICS    = ["feeding", "sleep", "health", "development", "gear", "postpartum", "mental_health"]
URGENCIES = ["routine", "monitor", "seek-help"]
STAGES    = ["trimester", "newborn", "0-3m", "3-6m", "6-12m", "toddler"]

# Keyword lists for engineered features (PRD §5.3)
MEDICAL_TERMS = {
    "fever", "temperature", "seizure", "convulsion", "rash", "vomit", "diarrhea",
    "diarrhoea", "jaundice", "infection", "antibiotic", "paracetamol", "ibuprofen",
    "hospital", "emergency", "doctor", "gp", "paediatrician", "pediatrician",
    "medication", "prescription", "diagnosis", "syndrome", "disorder", "reflux",
    "pyloric", "meningitis", "sepsis", "anaphylaxis", "allergy", "eczema",
    "bronchiolitis", "pneumonia", "fontanelle", "umbilical",
    "hypoglycemia", "hypoglycaemia", "apnea", "apnoea", "botulism", "dvt",
    "preeclampsia", "hyperemesis", "cholestasis", "dehydration", "thrombosis",
    "حمى", "تشنج", "طارئ", "طبيب", "مستشفى", "دواء", "تشخيص", "إسعاف",
    "التهاب", "عدوى", "حساسية", "يرقان", "جفاف",
}

URGENCY_SIGNALS = {
    "emergency", "urgent", "immediately", "right now", "call 999", "call 911",
    "go to a&e", "go to er", "go to hospital", "not breathing", "can't breathe",
    "unconscious", "unresponsive", "seizure", "convulsion", "severe", "critical",
    "danger", "dangerous", "life-threatening", "blood", "bleeding", "collapse",
    "anaphylaxis", "swelling throat", "difficulty breathing",
    "الآن", "فوراً", "طارئ", "إسعاف", "اتصلي", "روحي الطوارئ", "خطر",
    "لا تتأخري", "لا تنتظري",
}

ARABIC_UNICODE_RANGE = (0x0600, 0x06FF)

# Confidence threshold — read from config/env at classify() time (PRD §6.4)
# Default 0.45 (lowered from 0.60 while BiLSTM artefacts are unavailable)
_CONFIDENCE_THRESHOLD_DEFAULT = 0.45

# Models directory (relative to this file's package root)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))


# ──────────────────────────────────────────────────────────────────────────────
# Text Preprocessing  (PRD §5.1)
# ──────────────────────────────────────────────────────────────────────────────

def _is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return ARABIC_UNICODE_RANGE[0] <= cp <= ARABIC_UNICODE_RANGE[1]


def _arabic_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    ar = sum(1 for ch in text if _is_arabic_char(ch))
    return ar / len(text)


def detect_language(text: str) -> str:
    """
    Detect language from text.
    Primary: langdetect library.
    Fallback: Arabic Unicode character ratio > 0.30 → 'ar'.
    """
    # Fast path: Arabic character ratio check
    if _arabic_char_ratio(text) > 0.30:
        return "ar"

    try:
        from langdetect import detect, LangDetectException  # type: ignore
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except Exception:
        return "en"


def _expand_contractions(text: str) -> str:
    """Expand common English contractions for better TF-IDF coverage (PRD §5.1)."""
    contractions = {
        r"won't":      "will not",
        r"can't":      "cannot",
        r"n't\b":      " not",
        r"'re\b":      " are",
        r"'s\b":       " is",
        r"'d\b":       " would",
        r"'ll\b":      " will",
        r"'ve\b":      " have",
        r"'m\b":       " am",
        r"it's":       "it is",
        r"i'm":        "i am",
        r"they're":    "they are",
        r"we're":      "we are",
        r"you're":     "you are",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _normalise_arabic(text: str) -> str:
    """
    Arabic normalisation (PRD §5.1):
    - Remove tashkeel (diacritics)
    - Normalise alef variants → ا
    - Normalise taa marbuta → ة
    """
    # Remove tashkeel (harakat) and other diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Alef variants → ا
    text = re.sub(r'[أإآ]', 'ا', text)
    # Alef wasla → ا
    text = re.sub(r'ٱ', 'ا', text)
    # Waw with hamza → و
    text = re.sub(r'ؤ', 'و', text)
    # Yaa with hamza → ي
    text = re.sub(r'ئ', 'ي', text)
    return text


def clean_text(text: str, lang: str = "en") -> str:
    """
    Full text cleaning pipeline (PRD §5.1).
    Applied to situation + advice concatenation.
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    # Remove phone numbers
    text = re.sub(r'\b[\+]?[0-9][\d\s\-\(\)]{7,}\b', '', text)
    # Strip excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if lang == "en":
        text = text.lower()
        text = _expand_contractions(text)
        # Remove non-linguistic punctuation (keep sentence-ending for BiLSTM)
        text = re.sub(r'[^a-z0-9\s\.\!\?\,\-\']', ' ', text)
    else:  # Arabic
        text = _normalise_arabic(text)
        # Remove non-linguistic punctuation but keep Arabic sentence endings
        text = re.sub(r'[^\u0600-\u06FF\s\.\!\?،]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _token_length_bin(text: str) -> int:
    """Returns 0/1/2 for short/medium/long token count (PRD §5.3)."""
    n = len(text.split())
    if n < 20:
        return 0
    if n < 60:
        return 1
    return 2


# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering for TF-IDF branch  (PRD §5.3)
# ──────────────────────────────────────────────────────────────────────────────

def extract_engineered_features(text: str, lang: str) -> np.ndarray:
    """
    Returns a 4-element binary feature vector:
      [has_medical_term, has_urgency_signal, is_arabic, text_length_bin_0_or_1_or_2]
    (text_length_bin is encoded as a single int 0/1/2 for simplicity)
    """
    text_lower = text.lower()
    tokens = set(text_lower.split())

    has_medical = int(bool(tokens & MEDICAL_TERMS))
    has_urgency = int(any(sig in text_lower for sig in URGENCY_SIGNALS))
    is_ar       = int(lang == "ar")
    len_bin     = _token_length_bin(text)

    return np.array([has_medical, has_urgency, is_ar, len_bin], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# BiLSTM Model (Keras / TensorFlow)  (PRD §6.3)
# ──────────────────────────────────────────────────────────────────────────────

def _build_bilstm(vocab_size: int, max_seq_len: int,
                  embedding_dim: int = 128,
                  lstm_units: int = 128,
                  n_topics: int = 7,
                  n_urgencies: int = 3) -> "keras.Model":  # type: ignore
    """
    Multi-task BiLSTM architecture (PRD §6.3):
      Input → Embedding(128) → BiLSTM(128*2=256) → Dropout(0.3)
            → Dense(128, ReLU) → [Topic head Dense(7), Urgency head Dense(3)]
    """
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers, regularizers  # type: ignore

    inp = keras.Input(shape=(max_seq_len,), name="token_ids")

    x = layers.Embedding(vocab_size, embedding_dim, name="embedding")(inp)

    bilstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False, name="lstm"),
        name="bilstm"
    )(x)

    bilstm_out = layers.Dropout(0.3, name="dropout")(bilstm_out)

    shared = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
        name="shared_dense"
    )(bilstm_out)

    topic_out   = layers.Dense(n_topics,    activation="softmax", name="topic_output")(shared)
    urgency_out = layers.Dense(n_urgencies, activation="softmax", name="urgency_output")(shared)

    model = keras.Model(inputs=inp, outputs=[topic_out, urgency_out], name="mumzsense_bilstm")
    return model


def _build_hidden_state_extractor(model: "keras.Model") -> "keras.Model":  # type: ignore
    """
    Extract 256-dim BiLSTM hidden state for Gate A (MADRL logging).
    Returns a sub-model that outputs the bilstm layer's output.
    """
    from tensorflow import keras  # type: ignore
    return keras.Model(
        inputs=model.input,
        outputs=model.get_layer("bilstm").output,
        name="hidden_state_extractor"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tokeniser for BiLSTM
# ──────────────────────────────────────────────────────────────────────────────

class SimpleTokeniser:
    """
    Lightweight word tokeniser.
    Vocabulary built from training corpus (PRD §6.3: vocab_size ≤ 12,000).
    Pads/truncates to max_seq_len.
    OOV token: <UNK>.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab: int = 12000, max_seq_len: int = 90):
        self.max_vocab   = max_vocab
        self.max_seq_len = max_seq_len
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._fitted = False

    def fit(self, texts: list[str]) -> "SimpleTokeniser":
        from collections import Counter
        counter: Counter = Counter()
        for t in texts:
            counter.update(t.split())

        # Reserve 0=PAD, 1=UNK
        vocab = [self.PAD_TOKEN, self.UNK_TOKEN] + [
            w for w, _ in counter.most_common(self.max_vocab - 2)
        ]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._fitted = True
        return self

    def encode(self, text: str) -> list[int]:
        unk = self.word2idx.get(self.UNK_TOKEN, 1)
        ids = [self.word2idx.get(w, unk) for w in text.split()]
        # Pad / truncate to max_seq_len
        if len(ids) >= self.max_seq_len:
            return ids[:self.max_seq_len]
        return ids + [0] * (self.max_seq_len - len(ids))

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int32)

    def save(self, path: "str | Path") -> None:
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx,
                         "idx2word": self.idx2word,
                         "max_vocab": self.max_vocab,
                         "max_seq_len": self.max_seq_len}, f)

    @classmethod
    def load(cls, path: "str | Path") -> "SimpleTokeniser":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(max_vocab=state["max_vocab"], max_seq_len=state["max_seq_len"])
        obj.word2idx = state["word2idx"]
        obj.idx2word = state["idx2word"]
        obj._fitted  = True
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Calibration helper
# ──────────────────────────────────────────────────────────────────────────────

class IsotonicCalibrator:
    """
    Per-class isotonic regression calibrator.
    Temperature scaling on ensemble soft-vote outputs.
    """

    def __init__(self) -> None:
        self.calibrators: dict[str, Any] = {}  # "topic" | "urgency"

    def fit(self, probs_topic: np.ndarray, y_topic: np.ndarray,
            probs_urgency: np.ndarray, y_urgency: np.ndarray) -> "IsotonicCalibrator":
        """Fit one calibrator per head."""
        self.calibrators["topic"]   = self._fit_temperature(probs_topic, y_topic)
        self.calibrators["urgency"] = self._fit_temperature(probs_urgency, y_urgency)
        return self

    def _fit_temperature(self, probs: np.ndarray, y: np.ndarray) -> float:
        """
        Fit temperature scaling: T = argmin NLL( softmax(logits / T), y ).
        """
        from scipy.optimize import minimize_scalar  # type: ignore
        from scipy.special import softmax  # type: ignore

        eps = 1e-7
        logits = np.log(probs + eps)

        def nll(temp: float) -> float:
            scaled = softmax(logits / temp, axis=1)
            return -np.mean(np.log(scaled[np.arange(len(y)), y] + eps))

        result = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
        return float(result.x)

    def calibrate(self, probs: np.ndarray, head: str) -> np.ndarray:
        """Apply temperature scaling to a probability matrix."""
        from scipy.special import softmax  # type: ignore
        eps = 1e-7
        temp = self.calibrators.get(head, 1.0)
        logits = np.log(probs + eps)
        return softmax(logits / temp, axis=1)

    def save(self, path: "str | Path") -> None:
        with open(path, "wb") as f:
            pickle.dump(self.calibrators, f)

    @classmethod
    def load(cls, path: "str | Path") -> "IsotonicCalibrator":
        obj = cls()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            obj.calibrators = payload
        elif hasattr(payload, "calibrators"):
            obj.calibrators = getattr(payload, "calibrators")
        else:
            obj.calibrators = {}
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Main Classifier Agent
# ──────────────────────────────────────────────────────────────────────────────

class ClassifierAgent:
    """
    MumzSense dual-head ensemble classifier.

    Loads artefacts from MODELS_DIR:
      tfidf_vectoriser.pkl       – sklearn TfidfVectorizer
      logreg_topic.pkl           – sklearn LogisticRegression (topic)
      logreg_urgency.pkl         – sklearn LogisticRegression (urgency)
      bilstm_weights.h5          – Keras model weights
      bilstm_tokeniser.pkl       – SimpleTokeniser
      ensemble_weights.json      – {"topic": [w_tfidf, w_bilstm], "urgency": [...]}
      calibration_params.pkl     – IsotonicCalibrator
    """

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self._loaded    = False

        # Artefact slots
        self._tfidf_vec        = None
        self._tfidf_char_vec   = None
        self._logreg_topic     = None
        self._logreg_urgency   = None
        self._bilstm           = None
        self._bilstm_extractor = None  # hidden state sub-model (Gate A)
        self._tokeniser        = None
        self._ensemble_weights = {"topic": [0.55, 0.45], "urgency": [0.40, 0.60]}
        self._calibrator       = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "ClassifierAgent":
        """Load all model artefacts from disk."""
        md = self.models_dir
        logger.info("Loading classifier artefacts from %s", md)

        _install_numpy_pickle_compat()

        try:
            with open(md / "tfidf_vectoriser.pkl", "rb") as f:
                try:
                    self._tfidf_vec = pickle.load(f)
                except Exception as e:
                    logger.warning("Failed to unpickle tfidf_vectoriser.pkl: %s", e)
                    self._tfidf_vec = None

            char_vec_path = md / "tfidf_char_vectoriser.pkl"
            if char_vec_path.exists():
                with open(char_vec_path, "rb") as f:
                    self._tfidf_char_vec = pickle.load(f)
            else:
                self._tfidf_char_vec = None

            with open(md / "logreg_topic.pkl", "rb") as f:
                self._logreg_topic = pickle.load(f)

            with open(md / "logreg_urgency.pkl", "rb") as f:
                self._logreg_urgency = pickle.load(f)

            with open(md / "ensemble_weights.json") as f:
                self._ensemble_weights = json.load(f)

            if (md / "calibration_params.pkl").exists():
                self._calibrator = IsotonicCalibrator.load(md / "calibration_params.pkl")

            self._loaded = True
            try:
                self._tokeniser = SimpleTokeniser.load(md / "bilstm_tokeniser.pkl")
                vocab_size = len(self._tokeniser.word2idx)
                max_seq_len = self._tokeniser.max_seq_len
                self._bilstm = _build_bilstm(vocab_size, max_seq_len)
                self._bilstm.load_weights(str(md / "bilstm_weights.h5"))
                self._bilstm_extractor = _build_hidden_state_extractor(self._bilstm)
                logger.info("Classifier loaded successfully (TF-IDF + BiLSTM)")
            except FileNotFoundError as e:
                logger.warning("BiLSTM artefact missing (%s); using TF-IDF-only mode.", e)
                self._bilstm = None
                self._bilstm_extractor = None
                self._tokeniser = None
                logger.info("Classifier loaded successfully (TF-IDF-only)")
            except Exception as e:
                logger.warning("BiLSTM unavailable (%s); using TF-IDF-only mode.", e)
                self._bilstm = None
                self._bilstm_extractor = None
                self._tokeniser = None
                logger.info("Classifier loaded successfully (TF-IDF-only)")

        except FileNotFoundError as e:
            logger.warning("Artefact missing (%s). Run training/train_classifier.py first.", e)
        except Exception as e:
            logger.warning("Unexpected error loading classifier artefacts: %s. Continuing without classifier.", e)
            # Ensure we don't leave partially initialised artefacts that will crash later
            self._tfidf_vec = None
            self._tfidf_char_vec = None
            self._logreg_topic = None
            self._logreg_urgency = None
            self._calibrator = None
            self._loaded = False

        return self

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ─────────────────────────────────────────────────────────────

    def _tfidf_probs(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (topic_probs [7], urgency_probs [3]) from TF-IDF + LogReg branch.
        Appends engineered features before prediction.
        """
        lang     = detect_language(text)
        cleaned  = clean_text(text, lang)
        eng_feat = extract_engineered_features(text, lang)

        tfidf_word_vec = self._tfidf_vec.transform([cleaned])
        if self._tfidf_char_vec is not None:
            tfidf_char_vec = self._tfidf_char_vec.transform([cleaned])
        else:
            import scipy.sparse as sp  # type: ignore
            tfidf_char_vec = sp.csr_matrix((1, 0))

        # Append engineered features (PRD §5.3)
        import scipy.sparse as sp  # type: ignore
        eng_sparse = sp.csr_matrix(eng_feat.reshape(1, -1))
        combined   = sp.hstack([tfidf_word_vec, tfidf_char_vec, eng_sparse])

        expected_features = getattr(self._logreg_topic, "n_features_in_", combined.shape[1])
        if combined.shape[1] < expected_features:
            pad_width = expected_features - combined.shape[1]
            combined = sp.hstack([combined, sp.csr_matrix((1, pad_width))])
        elif combined.shape[1] > expected_features:
            combined = combined[:, :expected_features]

        topic_p   = self._logreg_topic.predict_proba(combined)[0]
        urgency_p = self._logreg_urgency.predict_proba(combined)[0]

        return topic_p, urgency_p

    def _bilstm_probs(self, text: str, lang: str
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (topic_probs [7], urgency_probs [3], hidden_state [256])
        from BiLSTM branch.
        """
        cleaned = clean_text(text, lang)
        ids     = self._tokeniser.encode_batch([cleaned])  # (1, max_seq_len)

        topic_p, urgency_p = self._bilstm.predict(ids, verbose=0)
        hidden             = self._bilstm_extractor.predict(ids, verbose=0)

        return topic_p[0], urgency_p[0], hidden[0]

    def _ensemble(self, tfidf_tp: np.ndarray, tfidf_up: np.ndarray,
                  bilstm_tp: np.ndarray, bilstm_up: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        Weighted soft-vote:
          P_ensemble = w_tfidf * P_tfidf + w_bilstm * P_bilstm
        Weights loaded from ensemble_weights.json (Bayesian-optimised).
        """
        wt, wb     = self._ensemble_weights["topic"]
        wu_t, wu_b = self._ensemble_weights["urgency"]

        topic_ens   = wt   * tfidf_tp + wb   * bilstm_tp
        urgency_ens = wu_t * tfidf_up + wu_b * bilstm_up
        return topic_ens, urgency_ens

    def classify(self, query: str,
                 stage_hint: Optional[str] = None,
                 lang_preference: Optional[str] = None,
                 log_hidden_state: bool = True
                 ) -> dict[str, Any]:
        """
        Full classification pipeline for a single query.
        """
        if not self._loaded:
            logger.warning("Classifier not loaded; returning fallback output")
            return self._fallback_output(query)

        # 1. Language detection
        lang = lang_preference if lang_preference in ("en", "ar") else detect_language(query)

        # 2. TF-IDF branch
        tfidf_tp, tfidf_up = self._tfidf_probs(query)

        # 3. BiLSTM branch
        if self._bilstm is not None and self._tokeniser is not None and self._bilstm_extractor is not None:
            bilstm_tp, bilstm_up, hidden_state = self._bilstm_probs(query, lang)
            # 4. Ensemble
            topic_ens, urgency_ens = self._ensemble(tfidf_tp, tfidf_up, bilstm_tp, bilstm_up)
        else:
            logger.warning("BiLSTM not loaded; using TF-IDF-only classifier outputs.")
            topic_ens, urgency_ens = tfidf_tp, tfidf_up
            hidden_state = None

        # 5. Calibration
        if self._calibrator is not None:
            topic_cal   = self._calibrator.calibrate(topic_ens.reshape(1, -1),   "topic")[0]
            urgency_cal = self._calibrator.calibrate(urgency_ens.reshape(1, -1), "urgency")[0]
        else:
            topic_cal, urgency_cal = topic_ens, urgency_ens

        # 6. Argmax → labels
        topic_idx   = int(np.argmax(topic_cal))
        urgency_idx = int(np.argmax(urgency_cal))
        topic_label   = TOPICS[topic_idx]
        urgency_label = URGENCIES[urgency_idx]

        topic_conf   = float(topic_cal[topic_idx])
        urgency_conf = float(urgency_cal[urgency_idx])

        # 7. Defer flag logic (PRD §6.4) — threshold read from settings/env
        try:
            from config import settings as _cfg
            threshold = _cfg.confidence_threshold
        except Exception:
            threshold = _CONFIDENCE_THRESHOLD_DEFAULT
        overall_conf = min(topic_conf, urgency_conf)
        defer_flag   = (overall_conf < threshold) or (urgency_label == "seek-help")

        # 8. Raw prob dicts
        raw_probs_topic   = {t: float(topic_cal[i])   for i, t in enumerate(TOPICS)}
        raw_probs_urgency = {u: float(urgency_cal[i]) for i, u in enumerate(URGENCIES)}

        # 9. Gate A: serialise BiLSTM hidden state for feedback_log
        hidden_state_serialised: Optional[list] = (
            hidden_state.tolist() if (log_hidden_state and hidden_state is not None) else None
        )

        return {
            "topic":               topic_label,
            "topic_confidence":    round(topic_conf, 4),
            "urgency":             urgency_label,
            "urgency_confidence":  round(urgency_conf, 4),
            "lang_detected":       lang,
            "defer_flag":          defer_flag,
            "raw_probs_topic":     raw_probs_topic,
            "raw_probs_urgency":   raw_probs_urgency,
            "bilstm_hidden_state": hidden_state_serialised,   # Gate A slot
            "stage_hint":          stage_hint,
        }

    # ── Fallback (no artefacts loaded) ────────────────────────────────────────

    def _fallback_output(self, query: str) -> dict[str, Any]:
        """
        Returns a safe output when models are not loaded.
        Uses keyword heuristics so RAG is still attempted for clear non-urgent queries.
        Only sets defer_flag=True for queries with explicit urgency signals.
        """
        lang = detect_language(query)
        text_lower = query.lower()
        tokens = set(text_lower.split())

        # Only hard-defer when there are genuine urgency signals
        has_urgency = bool(tokens & URGENCY_SIGNALS) or any(s in text_lower for s in URGENCY_SIGNALS)

        # Best-effort topic guess from keywords
        topic_keywords = {
            "feeding": {"feed", "latch", "breastfeed", "bottle", "milk", "nursing", "wean"},
            "sleep": {"sleep", "nap", "bedtime", "wake", "night", "tired"},
            "health": {"fever", "sick", "rash", "vomit", "diarrhea", "cough", "cry", "crying"},
            "development": {"crawl", "walk", "talk", "milestone", "growth", "teeth", "teething"},
            "postpartum": {"postpartum", "recovery", "birth", "labour", "labor", "c-section"},
            "mental_health": {"anxious", "depressed", "overwhelmed", "stress", "mood"},
        }
        topic = "health"
        for t, kw in topic_keywords.items():
            if tokens & kw:
                topic = t
                break

        urgency = "seek-help" if has_urgency else "routine"
        return {
            "topic":               topic,
            "topic_confidence":    0.50,
            "urgency":             urgency,
            "urgency_confidence":  0.50,
            "lang_detected":       lang,
            "defer_flag":          has_urgency,   # only defer on explicit urgency signals
            "raw_probs_topic":     {t: 1 / len(TOPICS) for t in TOPICS},
            "raw_probs_urgency":   {u: 1 / len(URGENCIES) for u in URGENCIES},
            "bilstm_hidden_state": None,
            "stage_hint":          None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Singleton + LangGraph node wrapper
# ──────────────────────────────────────────────────────────────────────────────

# Singleton instance (loaded once at startup by the graph pipeline)
_agent: Optional[ClassifierAgent] = None


def get_classifier_agent() -> ClassifierAgent:
    global _agent
    if _agent is None:
        _agent = ClassifierAgent().load()
    return _agent


def models_loaded() -> bool:
    """Convenience function called by /health endpoint."""
    return get_classifier_agent().is_loaded()


def classifier_node(state: dict) -> dict:
    """
    LangGraph node function.
    Reads state["query"] and state.get("stage_hint").
    Writes classifier outputs back into state.
    """
    agent  = get_classifier_agent()
    result = agent.classify(
        query           = state["query"],
        stage_hint      = state.get("stage_hint"),
        lang_preference = state.get("lang_preference"),
    )

    return {
        **state,
        "topic":               result["topic"],
        "urgency":             result["urgency"],
        "confidence":          min(result["topic_confidence"], result["urgency_confidence"]),
        "lang_detected":       result["lang_detected"],
        "defer_flag":          result["defer_flag"],
        "raw_probs_topic":     result["raw_probs_topic"],
        "raw_probs_urgency":   result["raw_probs_urgency"],
        "bilstm_hidden_state": result["bilstm_hidden_state"],
    }