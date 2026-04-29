"""
Convenience entry point for training, indexing, and optional evaluation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classifier training and corpus indexing")
    parser.add_argument("--corpus", default="data/corpus_validated.jsonl")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--no-train", action="store_false", dest="train")
    parser.add_argument("--index", action="store_true", default=True)
    parser.add_argument("--no-index", action="store_false", dest="index")
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    if args.train:
        from training.train_classifier import train
        logger.info("Starting classifier training...")
        train(Path(args.corpus))

    if args.index:
        from data.embed_and_index import embed_and_index
        logger.info("Starting corpus indexing...")
        import asyncio
        asyncio.run(embed_and_index(str(Path(args.corpus)), batch_size=args.batch_size))

    if args.evaluate:
        from training.evaluate import evaluate_corpus, run_eval_harness
        logger.info("Starting evaluation...")
        evaluate_corpus("test")
        run_eval_harness()


if __name__ == "__main__":
    main()