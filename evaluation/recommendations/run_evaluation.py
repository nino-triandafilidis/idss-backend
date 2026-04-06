#!/usr/bin/env python3
"""
Run Pipeline 2 evaluation: query->UCP, UCP->recs, baseline.

Full system (query->recs) is not run because the chat agent is multiturn and often
returns a question instead of recommendations; we test the merchant agent (UCP->recs) separately.
Writes results to evaluation/recommendations/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Load .env from repo root for OPENAI_API_KEY, DATABASE_URL, etc.
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_REPO_ROOT, ".env")
    if os.path.isfile(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass
_MCP = os.path.join(_REPO_ROOT, "mcp-server")
if _MCP not in sys.path:
    sys.path.insert(0, _MCP)

from evaluation.recommendations.schema import load_golden_dataset
from evaluation.recommendations.runners import (
    run_query_to_ucp,
    run_query_to_ucp_baseline,
    run_ucp_to_recs,
    run_baseline_eval,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Skip evaluation for items with fewer than this many ground-truth product IDs
# (database is not static; too few ground truths makes scoring unreliable)
MIN_GROUND_TRUTH_IDS = 3


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _run_one_eval(
    golden_path: str,
    skip_baseline: bool,
    ucp_recs_only: bool,
) -> tuple[list, list, list, list, int, int]:
    """Run one full evaluation; return (query_to_ucp, query_to_ucp_baseline, ucp_recs, baseline, num_items, skipped)."""
    all_items = load_golden_dataset(golden_path)
    items = [g for g in all_items if len(g.ground_truth_product_ids) >= MIN_GROUND_TRUTH_IDS]
    skipped = len(all_items) - len(items)

    query_to_ucp_results = []
    query_to_ucp_baseline_results = []
    if not ucp_recs_only:
        query_to_ucp_results = [run_query_to_ucp(g) for g in items]
        query_to_ucp_baseline_results = [run_query_to_ucp_baseline(g) for g in items]

    async def _ucp_recs():
        return await asyncio.gather(*[run_ucp_to_recs(g) for g in items])

    ucp_recs_results = asyncio.run(_ucp_recs())
    baseline_results = []
    if not ucp_recs_only and not skip_baseline:
        baseline_results = [run_baseline_eval(g) for g in items]

    return query_to_ucp_results, query_to_ucp_baseline_results, ucp_recs_results, baseline_results, len(items), skipped


def _average_by_query_id(runs: list[list[dict]], numeric_keys: list[str], query_key: str = "query_id") -> list[dict]:
    """Average numeric_keys across runs per query_id; use first run for non-numeric fields (e.g. query)."""
    from collections import defaultdict
    by_id = defaultdict(lambda: {k: [] for k in numeric_keys})
    template = {}  # query_id -> first row for string fields
    for run in runs:
        for r in run:
            qid = r.get(query_key)
            if qid is None:
                continue
            for k in numeric_keys:
                if k in r and isinstance(r[k], (int, float)):
                    by_id[qid][k].append(r[k])
            if qid not in template:
                template[qid] = {kk: r.get(kk) for kk in r if kk not in numeric_keys}
    out = []
    for qid in sorted(by_id.keys()):
        row = dict(template.get(qid, {}))
        row[query_key] = qid
        for k in numeric_keys:
            vals = by_id[qid].get(k, [])
            row[k] = sum(vals) / len(vals) if vals else 0.0
        out.append(row)
    return out


def run_all(
    golden_path: str,
    skip_baseline: bool = False,
    ucp_recs_only: bool = False,
    n_runs: int = 50,
) -> None:
    ensure_results_dir()
    n_runs = max(1, n_runs)
    print(f"Running {n_runs} evaluation run(s); metrics will be averaged.")

    runs_ucp = []
    runs_ucp_baseline = []
    runs_ucp_recs = []
    runs_baseline = []
    num_items = 0
    skipped = 0

    for run_idx in range(n_runs):
        if n_runs > 1:
            print(f"  Run {run_idx + 1}/{n_runs}...")
        a, b, c, d, num_items, skipped = _run_one_eval(golden_path, skip_baseline, ucp_recs_only)
        if not ucp_recs_only:
            runs_ucp.append(a)
            runs_ucp_baseline.append(b)
        runs_ucp_recs.append(c)
        if not ucp_recs_only and not skip_baseline:
            runs_baseline.append(d)

    if skipped:
        print(f"Skipping {skipped} items with < {MIN_GROUND_TRUTH_IDS} ground-truth IDs (evaluating {num_items} items)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def agg_recs(name: str, rows: list):
        if not rows:
            return None
        h = [r["hard_pass_rate"] for r in rows if "hard_pass_rate" in r]
        s = [r["mean_soft_similarity"] for r in rows if "mean_soft_similarity" in r]
        return {"run": name, "mean_hard_pass_rate": sum(h) / len(h) if h else 0, "mean_soft_similarity": sum(s) / len(s) if s else 0}

    def agg_ucp(name: str, rows: list):
        if not rows:
            return None
        m = [r["expected_ucp_match"] for r in rows if "expected_ucp_match" in r]
        return {"run": name, "mean_ucp_match_rate": sum(m) / len(m) if m else 0}

    query_to_ucp_results = []
    query_to_ucp_baseline_results = []
    if not ucp_recs_only:
        query_to_ucp_results = _average_by_query_id(runs_ucp, ["expected_ucp_match"])
        query_to_ucp_baseline_results = _average_by_query_id(runs_ucp_baseline, ["expected_ucp_match"])
        # Restore "query" from first run
        for r in query_to_ucp_results:
            r["query"] = runs_ucp[0][next(i for i, x in enumerate(runs_ucp[0]) if x["query_id"] == r["query_id"])]["query"]
        for r in query_to_ucp_baseline_results:
            r["query"] = runs_ucp_baseline[0][next(i for i, x in enumerate(runs_ucp_baseline[0]) if x["query_id"] == r["query_id"])]["query"]

        write_csv(
            RESULTS_DIR / f"query_to_ucp_results_{ts}.csv",
            [{"query_id": r["query_id"], "query": r["query"], "expected_ucp_match": r["expected_ucp_match"]} for r in query_to_ucp_results],
            ["query_id", "query", "expected_ucp_match"],
        )
        with open(RESULTS_DIR / f"query_to_ucp_results_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(query_to_ucp_results, f, indent=2)
        print(f"Wrote query_to_ucp_results (CSV + JSON)")

        write_csv(
            RESULTS_DIR / f"query_to_ucp_baseline_results_{ts}.csv",
            [{"query_id": r["query_id"], "query": r["query"], "expected_ucp_match": r["expected_ucp_match"]} for r in query_to_ucp_baseline_results],
            ["query_id", "query", "expected_ucp_match"],
        )
        with open(RESULTS_DIR / f"query_to_ucp_baseline_results_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(query_to_ucp_baseline_results, f, indent=2)
        print(f"Wrote query_to_ucp_baseline_results (CSV + JSON)")

    ucp_recs_results = _average_by_query_id(runs_ucp_recs, ["hard_pass_rate", "mean_soft_similarity", "recommended_count"])
    for r in ucp_recs_results:
        r["query"] = runs_ucp_recs[0][next(i for i, x in enumerate(runs_ucp_recs[0]) if x["query_id"] == r["query_id"])]["query"]

    write_csv(
        RESULTS_DIR / f"ucp_to_recs_results_{ts}.csv",
        [{"query_id": r["query_id"], "query": r["query"], "hard_pass_rate": r["hard_pass_rate"], "mean_soft_similarity": r["mean_soft_similarity"], "recommended_count": r["recommended_count"]} for r in ucp_recs_results],
        ["query_id", "query", "hard_pass_rate", "mean_soft_similarity", "recommended_count"],
    )
    with open(RESULTS_DIR / f"ucp_to_recs_results_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(ucp_recs_results, f, indent=2)
    print(f"Wrote ucp_to_recs_results (CSV + JSON)")

    baseline_results = []
    if not ucp_recs_only and not skip_baseline:
        baseline_results = _average_by_query_id(runs_baseline, ["hard_pass_rate", "mean_soft_similarity", "recommended_count"])
        for r in baseline_results:
            r["query"] = runs_baseline[0][next(i for i, x in enumerate(runs_baseline[0]) if x["query_id"] == r["query_id"])]["query"]
        write_csv(
            RESULTS_DIR / f"baseline_results_{ts}.csv",
            [{"query_id": r["query_id"], "query": r["query"], "hard_pass_rate": r["hard_pass_rate"], "mean_soft_similarity": r["mean_soft_similarity"], "recommended_count": r["recommended_count"]} for r in baseline_results],
            ["query_id", "query", "hard_pass_rate", "mean_soft_similarity", "recommended_count"],
        )
        with open(RESULTS_DIR / f"baseline_results_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"Wrote baseline_results (CSV + JSON)")
    else:
        print("Skipped baseline (--skip-baseline)")

    summary = {
        "timestamp": ts,
        "golden_path": golden_path,
        "n_runs": n_runs,
        "num_queries": num_items,
        "num_skipped_insufficient_ground_truth": skipped,
        "min_ground_truth_ids": MIN_GROUND_TRUTH_IDS,
        "ucp_recs_only": ucp_recs_only,
        "aggregates": {
            "query_to_ucp": agg_ucp("query_to_ucp", query_to_ucp_results) if not ucp_recs_only else None,
            "query_to_ucp_baseline": agg_ucp("query_to_ucp_baseline", query_to_ucp_baseline_results) if not ucp_recs_only else None,
            "ucp_to_recs": agg_recs("ucp_to_recs", ucp_recs_results),
            "baseline": agg_recs("baseline", baseline_results) if not ucp_recs_only and not skip_baseline else None,
        },
    }
    with open(RESULTS_DIR / f"summary_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary_{ts}.json")


def main() -> None:
    n_default = 50
    try:
        n_default = int(os.environ.get("N_RUNS", str(n_default)))
    except (TypeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Run Pipeline 2 recommendation evaluation")
    parser.add_argument(
        "--golden",
        default=os.path.join(os.path.dirname(__file__), "golden_dataset.json"),
        help="Path to golden dataset JSON",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline LLM run")
    parser.add_argument("--ucp-recs-only", action="store_true", help="Run only UCP->recs (skip query->UCP and baseline)")
    parser.add_argument("--n-runs", type=int, default=n_default, metavar="N", help=f"Number of runs to average (default: {n_default})")
    args = parser.parse_args()
    run_all(args.golden, skip_baseline=args.skip_baseline, ucp_recs_only=args.ucp_recs_only, n_runs=args.n_runs)


if __name__ == "__main__":
    main()
