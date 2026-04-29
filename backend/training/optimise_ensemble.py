# Ensemble Weight Optimization
# Bayesian optimization for TF-IDF + BiLSTM weights
"""
MumzSense v1 — Ensemble Weight Optimisation
============================================
Bayesian optimisation of TF-IDF vs BiLSTM soft-vote weights
for topic and urgency heads independently.

Can be run standalone to re-tune weights after partial retraining
without rerunning the full training pipeline.

Usage:
    python optimise_ensemble.py \
        --tfidf_tp  val_probs_tfidf_topic.npy \
        --tfidf_up  val_probs_tfidf_urgency.npy \
        --bilstm_tp val_probs_bilstm_topic.npy \
        --bilstm_up val_probs_bilstm_urgency.npy \
        --y_topic   val_labels_topic.npy \
        --y_urgency val_labels_urgency.npy \
        --n_calls   30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))

# ──────────────────────────────────────────────────────────────────────────────
# Objective
# ──────────────────────────────────────────────────────────────────────────────

def _objective(params: list[float],
               tfidf_tp: np.ndarray, tfidf_up: np.ndarray,
               bilstm_tp: np.ndarray, bilstm_up: np.ndarray,
               y_topic: np.ndarray, y_urgency: np.ndarray,
               objective_metric: str = "joint_acc") -> float:
    """
    Compute negative objective (for minimisation).

    objective_metric options:
        "joint_acc"   – maximise (topic_acc + urgency_acc)
        "joint_f1"    – maximise (topic_f1  + urgency_f1)
        "seek_recall" – maximise urgency seek-help recall (with acc constraint)
    """
    wt_topic, wt_urgency = params[0], params[1]

    ens_topic   = wt_topic   * tfidf_tp + (1 - wt_topic)   * bilstm_tp
    ens_urgency = wt_urgency * tfidf_up + (1 - wt_urgency) * bilstm_up

    pred_topic   = np.argmax(ens_topic,   axis=1)
    pred_urgency = np.argmax(ens_urgency, axis=1)

    if objective_metric == "joint_f1":
        val = (
            f1_score(y_topic,   pred_topic,   average="macro", zero_division=0)
            + f1_score(y_urgency, pred_urgency, average="macro", zero_division=0)
        )
    elif objective_metric == "seek_recall":
        seek_idx = 2  # URGENCIES.index("seek-help")
        recall   = float(np.mean((pred_urgency == seek_idx) & (y_urgency == seek_idx)))
        # penalise if overall urgency acc drops below 0.70
        urgency_acc = accuracy_score(y_urgency, pred_urgency)
        val = recall if urgency_acc >= 0.70 else recall * 0.5
    else:  # joint_acc (default)
        val = (
            accuracy_score(y_topic,   pred_topic)
            + accuracy_score(y_urgency, pred_urgency)
        )

    return -val  # minimise


# ──────────────────────────────────────────────────────────────────────────────
# Bayesian Optimisation
# ──────────────────────────────────────────────────────────────────────────────

def bayesian_optimise(
    tfidf_tp: np.ndarray, tfidf_up: np.ndarray,
    bilstm_tp: np.ndarray, bilstm_up: np.ndarray,
    y_topic: np.ndarray, y_urgency: np.ndarray,
    n_calls: int = 30,
    objective_metric: str = "joint_acc",
) -> dict[str, list[float]]:
    """
    1-D Bayesian search over w_tfidf ∈ [0.2, 0.8] for each head independently.
    Returns {"topic": [w_tfidf, w_bilstm], "urgency": [w_tfidf, w_bilstm]}.
    """
    try:
        from skopt import gp_minimize  # type: ignore
        from skopt.space import Real    # type: ignore

        logger.info("Running Bayesian optimisation (%d calls, metric=%s)…",
                    n_calls, objective_metric)

        result = gp_minimize(
            func=lambda p: _objective(
                p, tfidf_tp, tfidf_up, bilstm_tp, bilstm_up,
                y_topic, y_urgency, objective_metric
            ),
            dimensions=[
                Real(0.20, 0.80, name="w_topic"),
                Real(0.20, 0.80, name="w_urgency"),
            ],
            n_calls=n_calls,
            random_state=42,
            acq_func="EI",        # Expected Improvement
            n_initial_points=10,  # random exploration before Bayesian
        )

        w_topic, w_urgency = result.x
        best_score = -result.fun

        logger.info("Optimisation complete. Best score=%.4f", best_score)
        logger.info("  topic:   tfidf=%.3f  bilstm=%.3f", w_topic,   1 - w_topic)
        logger.info("  urgency: tfidf=%.3f  bilstm=%.3f", w_urgency, 1 - w_urgency)

        # Log all evaluations for diagnostics
        evaluations = [
            {
                "w_topic": float(xi[0]),
                "w_urgency": float(xi[1]),
                "score": float(-yi),
            }
            for xi, yi in zip(result.x_iters, result.func_vals)
        ]

        return {
            "topic":       [float(w_topic),   float(1 - w_topic)],
            "urgency":     [float(w_urgency), float(1 - w_urgency)],
            "best_score":  float(best_score),
            "n_calls":     n_calls,
            "metric":      objective_metric,
            "evaluations": evaluations,
        }

    except ImportError:
        logger.warning("scikit-optimize not installed. Falling back to grid search.")
        return _grid_search_weights(
            tfidf_tp, tfidf_up, bilstm_tp, bilstm_up,
            y_topic, y_urgency, objective_metric
        )


def _grid_search_weights(
    tfidf_tp: np.ndarray, tfidf_up: np.ndarray,
    bilstm_tp: np.ndarray, bilstm_up: np.ndarray,
    y_topic: np.ndarray, y_urgency: np.ndarray,
    objective_metric: str = "joint_acc",
) -> dict[str, list[float]]:
    """
    Fallback grid search when scikit-optimize is not available.
    Searches w ∈ {0.2, 0.25, ..., 0.80} (13 values) for each head = 169 evaluations.
    """
    logger.info("Grid search over weights (169 evaluations)…")
    weights = np.arange(0.20, 0.85, 0.05)

    best_score = float("inf")
    best_wt, best_wu = 0.55, 0.40

    for wt in weights:
        for wu in weights:
            score = _objective(
                [wt, wu], tfidf_tp, tfidf_up, bilstm_tp, bilstm_up,
                y_topic, y_urgency, objective_metric
            )
            if score < best_score:
                best_score = score
                best_wt, best_wu = wt, wu

    logger.info("Grid search complete. Best: topic_tfidf=%.2f  urgency_tfidf=%.2f",
                best_wt, best_wu)

    return {
        "topic":      [float(best_wt), float(1 - best_wt)],
        "urgency":    [float(best_wu), float(1 - best_wu)],
        "best_score": float(-best_score),
        "n_calls":    len(weights) ** 2,
        "metric":     objective_metric,
        "evaluations": [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ──────────────────────────────────────────────────────────────────────────────

def sweep_and_report(
    tfidf_tp: np.ndarray, tfidf_up: np.ndarray,
    bilstm_tp: np.ndarray, bilstm_up: np.ndarray,
    y_topic: np.ndarray, y_urgency: np.ndarray,
    step: float = 0.10,
) -> list[dict]:
    """
    Sweep w_tfidf in [0, 1] with `step` resolution for both heads.
    Returns a list of metric dicts for diagnostics / plotting.
    """
    rows = []
    for wt in np.arange(0.0, 1.0 + step, step):
        wt = round(float(wt), 2)
        for wu in np.arange(0.0, 1.0 + step, step):
            wu = round(float(wu), 2)
            ens_t = wt * tfidf_tp + (1 - wt) * bilstm_tp
            ens_u = wu * tfidf_up + (1 - wu) * bilstm_up
            pt    = np.argmax(ens_t, axis=1)
            pu    = np.argmax(ens_u, axis=1)

            seek_idx = 2
            sh_recall = float(np.mean((pu == seek_idx) & (y_urgency == seek_idx))) \
                if np.sum(y_urgency == seek_idx) > 0 else 0.0

            rows.append({
                "w_tfidf_topic":    wt,
                "w_tfidf_urgency":  wu,
                "topic_acc":        round(accuracy_score(y_topic, pt), 4),
                "topic_f1":         round(f1_score(y_topic, pt, average="macro", zero_division=0), 4),
                "urgency_acc":      round(accuracy_score(y_urgency, pu), 4),
                "urgency_f1":       round(f1_score(y_urgency, pu, average="macro", zero_division=0), 4),
                "seek_help_recall": round(sh_recall, 4),
            })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimise TF-IDF + BiLSTM ensemble weights for MumzSense classifier."
    )
    parser.add_argument("--tfidf_tp",  type=str, help="Path to TF-IDF topic probs (.npy)")
    parser.add_argument("--tfidf_up",  type=str, help="Path to TF-IDF urgency probs (.npy)")
    parser.add_argument("--bilstm_tp", type=str, help="Path to BiLSTM topic probs (.npy)")
    parser.add_argument("--bilstm_up", type=str, help="Path to BiLSTM urgency probs (.npy)")
    parser.add_argument("--y_topic",   type=str, help="Path to topic labels (.npy)")
    parser.add_argument("--y_urgency", type=str, help="Path to urgency labels (.npy)")
    parser.add_argument("--n_calls",   type=int, default=30, help="Bayesian calls (default 30)")
    parser.add_argument("--metric",    type=str, default="joint_acc",
                        choices=["joint_acc", "joint_f1", "seek_recall"],
                        help="Optimisation objective")
    parser.add_argument("--sweep",     action="store_true", help="Run full sweep and save CSV")
    parser.add_argument("--out_dir",   type=str, default=str(MODELS_DIR))
    args = parser.parse_args()

    if not args.tfidf_tp:
        # Load from saved models dir (training pipeline integration)
        logger.info("No prob arrays provided; loading from models dir for re-optimisation…")
        logger.error("Provide --tfidf_tp etc. or run from train_classifier.py")
        return

    tfidf_tp  = np.load(args.tfidf_tp)
    tfidf_up  = np.load(args.tfidf_up)
    bilstm_tp = np.load(args.bilstm_tp)
    bilstm_up = np.load(args.bilstm_up)
    y_topic   = np.load(args.y_topic)
    y_urgency = np.load(args.y_urgency)

    result = bayesian_optimise(
        tfidf_tp, tfidf_up, bilstm_tp, bilstm_up,
        y_topic, y_urgency,
        n_calls=args.n_calls,
        objective_metric=args.metric,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ensemble_weights.json"

    # Save only the weights (strip diagnostics for the artefact)
    weights_only = {"topic": result["topic"], "urgency": result["urgency"]}
    with open(out_path, "w") as f:
        json.dump(weights_only, f, indent=2)
    logger.info("Weights saved to %s", out_path)

    # Save full diagnostics
    diag_path = out_dir / "ensemble_optimisation_log.json"
    with open(diag_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Diagnostics saved to %s", diag_path)

    if args.sweep:
        sweep_rows = sweep_and_report(
            tfidf_tp, tfidf_up, bilstm_tp, bilstm_up, y_topic, y_urgency
        )
        import csv
        sweep_path = out_dir / "ensemble_sweep.csv"
        with open(sweep_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
            writer.writeheader()
            writer.writerows(sweep_rows)
        logger.info("Sweep saved to %s", sweep_path)


if __name__ == "__main__":
    main()