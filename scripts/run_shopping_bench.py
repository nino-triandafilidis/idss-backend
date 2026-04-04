#!/usr/bin/env python3
"""
run_shopping_bench.py — CLI entry point for IDSS-Shopping-Bench.

Runs TauBench-style shopping scenarios against a live IDSS /chat endpoint.
Scoring: 55% deterministic constraint checks + 45% GPT-4o-mini judge.

Usage:
  # Run all 20 tasks
  python scripts/run_shopping_bench.py --url http://localhost:8001

  # Run a specific task
  python scripts/run_shopping_bench.py --task brand_excl_01 --url http://localhost:8001 --verbose

  # Run a category
  python scripts/run_shopping_bench.py --category budget --verbose

  # Save results
  python scripts/run_shopping_bench.py --save scripts/shopping_bench_v1_$(date +%Y%m%d).json

  # Dry-run: list tasks without calling IDSS
  python scripts/run_shopping_bench.py --list
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

import httpx
from dotenv import load_dotenv

# Ensure scripts/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from shopping_bench.runner import run_task, CONSTRAINT_WEIGHT, JUDGE_WEIGHT, PASS_THRESHOLD
from shopping_bench.tasks import (
    TASKS,
    TASKS_BY_CATEGORY,
    VALID_CATEGORIES,
    ShoppingTask,
    get_task,
    get_tasks_for_category,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# Async OpenAI setup
# ---------------------------------------------------------------------------

def _make_oai():
    """Create an AsyncOpenAI client. Raises if key is missing."""
    try:
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_score(score: float) -> str:
    """Return a colored score indicator."""
    if score >= 0.7:
        return f"✓ {score:.3f}"
    elif score >= 0.5:
        return f"~ {score:.3f}"
    else:
        return f"✗ {score:.3f}"


def _print_result_verbose(result, task: ShoppingTask) -> None:
    """Print detailed output for one task result."""
    status = "PASS" if result.passed else "FAIL"
    print(f"\n{'─'*60}")
    print(f"[{status}] {result.task_id}  ({result.task_category})")
    print(f"  Description: {task.description}")
    print(f"  Constraint:  {result.final_constraint_score:.3f}  "
          f"Judge: {result.judge_score:.3f}  "
          f"Final: {result.final_score:.3f}")
    print(f"  Turns: {len(result.turn_results)}")
    if result.notes:
        for note in result.notes:
            print(f"  NOTE: {note}")
    if result.error:
        print(f"  ERROR: {result.error}")

    for tr in result.turn_results:
        print(f"\n  Turn {tr.turn_number} [{tr.response_type}]")
        print(f"    USER: {tr.user_message[:100]}")
        print(f"    IDSS: {tr.assistant_message[:150]}")
        if tr.products:
            for p in tr.products[:3]:
                price = f"${p.get('price_value') or '?'}" if p.get("price_value") else ""
                print(f"      ↳ {p.get('brand','')} {p.get('name','')} {price}")


def _print_summary_table(results, tasks_map) -> None:
    """Print a compact summary table of all results."""
    print(f"\n{'='*70}")
    print(f"{'IDSS-Shopping-Bench Results':^70}")
    print(f"{'='*70}")
    print(f"{'Task ID':<22}{'Category':<22}{'Constraint':<12}{'Judge':<8}{'Final':<8}{'Pass'}")
    print(f"{'─'*70}")

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    category_scores: dict = {}

    for r in results:
        task = tasks_map.get(r.task_id)
        cat_short = r.task_category[:18]
        p = "✓" if r.passed else "✗"
        print(
            f"{r.task_id:<22}{cat_short:<22}"
            f"{r.final_constraint_score:.3f}       "
            f"{r.judge_score:.3f}   "
            f"{r.final_score:.3f}   {p}"
        )
        category_scores.setdefault(r.task_category, []).append(r.final_score)

    print(f"{'─'*70}")

    # Averages per category
    print("\nPer-category averages:")
    for cat, scores in sorted(category_scores.items()):
        avg = sum(scores) / len(scores)
        cat_pass = sum(1 for r in results if r.task_category == cat and r.passed)
        print(f"  {cat:<25} avg={avg:.3f}  pass={cat_pass}/{len(scores)}")

    # Overall
    all_scores = [r.final_score for r in results]
    all_constraints = [r.final_constraint_score for r in results]
    all_judge = [r.judge_score for r in results]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    avg_constraint = sum(all_constraints) / len(all_constraints) if all_constraints else 0
    avg_judge = sum(all_judge) / len(all_judge) if all_judge else 0

    print(f"\n{'='*70}")
    print(f"OVERALL: {passed}/{total} passed ({100*passed/total:.1f}%)")
    print(f"  Avg constraint score: {avg_constraint:.3f}")
    print(f"  Avg judge score:      {avg_judge:.3f}")
    print(f"  Avg final score:      {avg_score:.3f}")
    print(f"  Scoring:              {CONSTRAINT_WEIGHT:.0%} constraint + {JUDGE_WEIGHT:.0%} judge")
    print(f"  Pass threshold:       {PASS_THRESHOLD}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

async def main_async(args) -> None:
    # Determine which tasks to run
    if args.list:
        print(f"IDSS-Shopping-Bench — {len(TASKS)} tasks:\n")
        for t in TASKS:
            n_constraints = len(t.success_criteria)
            print(f"  {t.id:<22} [{t.category}]  {n_constraints} constraint(s)  —  {t.description}")
        return

    if args.task:
        tasks_to_run = [get_task(args.task)]
    elif args.category:
        tasks_to_run = get_tasks_for_category(args.category)
    else:
        tasks_to_run = list(TASKS)

    if not tasks_to_run:
        print("No tasks to run.")
        return

    print(f"Running {len(tasks_to_run)} task(s) against {args.url} ...")
    print(f"Concurrency: {args.concurrency}  |  Scoring: {CONSTRAINT_WEIGHT:.0%} constraint + {JUDGE_WEIGHT:.0%} judge")
    print()

    oai = _make_oai()
    sem = asyncio.Semaphore(args.concurrency)
    tasks_map = {t.id: t for t in tasks_to_run}

    t_start = time.monotonic()

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        coroutines = [
            run_task(t, args.url, http_client, oai, sem)
            for t in tasks_to_run
        ]
        results = await asyncio.gather(*coroutines)

    elapsed = time.monotonic() - t_start

    if args.verbose:
        for result, task in zip(results, tasks_to_run):
            _print_result_verbose(result, task)

    _print_summary_table(list(results), tasks_map)
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    if args.save:
        output = {
            "meta": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "idss_url": args.url,
                "tasks_run": len(tasks_to_run),
                "constraint_weight": CONSTRAINT_WEIGHT,
                "judge_weight": JUDGE_WEIGHT,
                "pass_threshold": PASS_THRESHOLD,
                "elapsed_s": round(elapsed, 1),
            },
            "summary": {
                "passed": sum(1 for r in results if r.passed),
                "total": len(results),
                "avg_constraint": round(sum(r.final_constraint_score for r in results) / len(results), 4),
                "avg_judge": round(sum(r.judge_score for r in results) / len(results), 4),
                "avg_final": round(sum(r.final_score for r in results) / len(results), 4),
            },
            "results": [
                {
                    "task_id": r.task_id,
                    "task_category": r.task_category,
                    "final_constraint_score": round(r.final_constraint_score, 4),
                    "judge_score": round(r.judge_score, 4),
                    "final_score": round(r.final_score, 4),
                    "passed": r.passed,
                    "turns": len(r.turn_results),
                    "notes": r.notes,
                    "error": r.error,
                    "turn_results": [
                        {
                            "turn_number": tr.turn_number,
                            "user_message": tr.user_message,
                            "assistant_message": tr.assistant_message[:500],
                            "response_type": tr.response_type,
                            "products": tr.products,
                            "elapsed_s": round(tr.elapsed_s, 2),
                        }
                        for tr in r.turn_results
                    ],
                }
                for r in results
            ],
        }
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {args.save}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IDSS-Shopping-Bench: TauBench-style pre-purchase shopping evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8001",
        help="IDSS /chat endpoint base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--task",
        metavar="TASK_ID",
        help="Run a single task by ID (e.g. brand_excl_01)",
    )
    parser.add_argument(
        "--category",
        choices=sorted(VALID_CATEGORIES),
        help="Run all tasks in a category",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent IDSS requests (default: 1 — serial for rate limiting)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-turn conversation details",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all task IDs and descriptions without running",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save results to a JSON file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
