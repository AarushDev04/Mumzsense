# Evaluation Harness Script
# Runs 20+ test cases and computes metrics
"""
MumzSense v1 — Evaluation Harness (PRD §11)
Run all 20 test cases, compute metrics, save to results.json.

Usage:
    cd backend
    python ../evals/run_evals.py
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EVALS_DIR = os.path.dirname(__file__)
TEST_CASES_PATH = os.path.join(EVALS_DIR, "test_cases.json")
RESULTS_PATH = os.path.join(EVALS_DIR, "results.json")


async def run_case(case: dict) -> dict:
    """Run a single test case through the full pipeline."""
    from graph.pipeline import run_pipeline
    start = time.time()
    try:
        state = await run_pipeline(
            query=case["query"],
            stage_hint=case.get("stage_hint"),
            lang_preference=case.get("lang"),
        )
        latency_ms = int((time.time() - start) * 1000)
        got_defer = bool(state.get("defer_message") or state.get("defer_flag"))
        expected_defer = case.get("expect_defer", False)
        passed = expected_defer == got_defer

        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "expected_defer": expected_defer,
            "got_defer": got_defer,
            "topic": state.get("topic"),
            "urgency": state.get("urgency"),
            "retrieval_confidence": state.get("retrieval_confidence"),
            "max_similarity": state.get("max_similarity", 0.0),
            "latency_ms": latency_ms,
            "hallucination_risk": state.get("hallucination_risk", False),
            "cached": state.get("cached", False),
            "pass": passed,
            "error": state.get("error"),
        }
    except Exception as e:
        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "pass": False,
            "error": str(e),
        }


async def main():
    with open(TEST_CASES_PATH) as f:
        test_cases = json.load(f)

    logger.info(f"Running {len(test_cases)} test cases...")
    results = []
    for case in test_cases:
        logger.info(f"  Running case: {case['id']}")
        result = await run_case(case)
        results.append(result)
        status = "✓ PASS" if result.get("pass") else "✗ FAIL"
        logger.info(f"  {status}  latency={result.get('latency_ms',0)}ms")

    # Aggregate
    passed = sum(1 for r in results if r.get("pass", False))
    total = len(results)
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]

    by_category: dict = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, {"pass": 0, "total": 0})
        by_category[cat]["total"] += 1
        if r.get("pass"):
            by_category[cat]["pass"] += 1

    summary = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_cases_passed": passed,
        "test_cases_total": total,
        "pass_rate": round(passed / max(total, 1), 3),
        "latency": {
            "mean_ms": round(sum(latencies) / max(len(latencies), 1)),
            "max_ms": max(latencies, default=0),
            "min_ms": min(latencies, default=0),
        },
        "by_category": by_category,
        "cases": results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Results: {passed}/{total} passed ({summary['pass_rate']*100:.0f}%)")
    logger.info(f"Mean latency: {summary['latency']['mean_ms']}ms")
    logger.info(f"Saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
