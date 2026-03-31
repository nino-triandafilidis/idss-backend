#!/usr/bin/env python3
"""
GPT-4o-mini Shopping Baseline Evaluator
========================================
Fair comparison baseline for the IDSS G-Eval paper.

Sends each query in QUERIES to GPT-4o-mini with a minimal shopping assistant
system prompt, then scores the response using the IDENTICAL judge and metrics
as run_geval.py.  Differences from the IDSS agent:
  - No product database (GPT hallucinates product specs)
  - No brand exclusion enforcement (no structured output)
  - No filter extraction (no structured output)
  - No availability check (no structured output)
  - Quality scoring: SAME LLM judge (GEVAL_SYSTEM + GEVAL_USER_TEMPLATE)
  - Response type scoring: inferred from free text

This makes quality the primary comparable metric.  Type accuracy and brand
adherence measure structured-output capabilities that GPT-4o-mini (without
function calling) cannot demonstrate — these are reported separately.

Usage:
    python scripts/run_gpt_baseline.py
    python scripts/run_gpt_baseline.py --save scripts/geval_gpt_baseline.json
    python scripts/run_gpt_baseline.py --group expert
    python scripts/run_gpt_baseline.py --concurrency 10
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai not installed. Run: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# ── Import shared data and scoring from run_geval ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from run_geval import (
        QUERIES,
        GEVAL_SYSTEM,
        GEVAL_USER_TEMPLATE,
        score_quality_async,
        compute_final_score,
        PASS_THRESHOLD,
    )
except ImportError as e:
    print(f"ERROR: Cannot import from run_geval.py: {e}")
    sys.exit(1)

# ── ANSI colours ──────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"
BOLD  = "\033[1m";  RST = "\033[0m"

def color_score(s: float) -> str:
    c = GREEN if s >= PASS_THRESHOLD else (YEL if s >= 0.5 else RED)
    return f"{c}{s:.3f}{RST}"

# ── GPT baseline system prompt ─────────────────────────────────────────────
BASELINE_SYSTEM = """\
You are a helpful online shopping assistant specialising in laptops and personal
computers.  When the user describes what they need, give them 3–5 concrete
product recommendations with a short reason for each.  If the user's request is
too vague (one or two words), ask ONE focused clarifying question.
Be concise, friendly, and match the user's communication style.
"""


# ============================================================================
# Response type inference from free text
# ============================================================================

_QUESTION_RE = re.compile(
    r"(what|which|how|could you|can you|tell me|do you).{0,80}\?",
    re.IGNORECASE,
)
_REC_BULLETS_RE = re.compile(
    r"^[\s]*[\-\*\d]+[\.\)]\s+\S",
    re.MULTILINE,
)
_PRODUCT_LINE_RE = re.compile(
    r"(laptop|notebook|macbook|thinkpad|inspiron|xps|aspire|envy|surface|chromebook)",
    re.IGNORECASE,
)


def infer_response_type(text: str) -> str:
    """
    Heuristic: decide whether GPT gave recommendations or asked a question.

    Returns "recommendations" | "question" | "unknown".
    """
    lines = text.strip().splitlines()
    # Count bullet/numbered lines that look like product recommendations
    bullet_lines = [l for l in lines if _REC_BULLETS_RE.match(l)]
    product_mentions = len(_PRODUCT_LINE_RE.findall(text))

    # If 2+ bullet lines containing product-like content → recommendations
    if len(bullet_lines) >= 2 and product_mentions >= 2:
        return "recommendations"

    # If ends with a question or the dominant structure is a single question
    question_matches = _QUESTION_RE.findall(text)
    if question_matches and len(bullet_lines) < 2:
        return "question"

    # If there are product mentions but no bullets, it may still be recommendations
    if product_mentions >= 3:
        return "recommendations"

    # Single question mark at end of the response
    if text.rstrip().endswith("?"):
        return "question"

    return "unknown"


def check_response_type_baseline(query: Dict, response_type: str) -> Tuple[float, str]:
    """Same logic as run_geval.check_response_type but takes inferred type string."""
    exp_recs = query.get("expect_recs_on_first", False)
    exp_q    = query.get("expect_question", False)

    if exp_recs and not exp_q:
        if response_type == "recommendations":
            return 1.0, "✓ Gave recommendations as expected"
        elif response_type == "unknown":
            return 0.5, "~ Response type ambiguous"
        else:
            return 0.0, f"✗ Expected recommendations, got {response_type!r}"
    elif exp_q and not exp_recs:
        if response_type == "question":
            return 1.0, "✓ Asked clarifying question as expected"
        elif response_type == "unknown":
            return 0.5, "~ Response type ambiguous"
        else:
            return 0.0, f"✗ Expected question, got {response_type!r}"
    else:
        # Both or neither expected → either is fine
        return 1.0, f"~ Either acceptable; got {response_type}"


# ============================================================================
# Call GPT-4o-mini as the baseline agent
# ============================================================================

async def call_baseline_async(
    oai: AsyncOpenAI,
    message: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, int]:
    """
    Send a single message to GPT-4o-mini shopping assistant.

    Returns (response_text, elapsed_ms).
    """
    async with semaphore:
        t0 = time.perf_counter()
        try:
            completion = await oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM},
                    {"role": "user",   "content": message},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            text = completion.choices[0].message.content.strip()
        except Exception as e:
            text = f"[ERROR: {e}]"
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return text, elapsed_ms


# ============================================================================
# Main evaluation loop
# ============================================================================

async def run_baseline(
    queries: List[Dict],
    save_path: Optional[str],
    concurrency: int,
    verbose: bool,
) -> List[Dict]:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    oai = AsyncOpenAI(api_key=openai_key)
    sem = asyncio.Semaphore(concurrency)

    total = len(queries)
    print(f"\n{BOLD}GPT-4o-mini Baseline Evaluator{RST}")
    print(f"  Queries : {total}")
    print(f"  Model   : gpt-4o-mini (shopping assistant + same LLM judge)")
    print(f"  Concurrency: {concurrency}\n")

    # ── Phase 1: collect baseline responses ───────────────────────────────
    print(f"  Phase 1 — Calling GPT-4o-mini for {total} queries …")
    t_phase1 = time.perf_counter()

    async def fetch_one(q: Dict):
        text, elapsed_ms = await call_baseline_async(oai, q["message"], sem)
        return {"id": q["id"], "text": text, "elapsed_ms": elapsed_ms}

    raw_responses = await asyncio.gather(*[fetch_one(q) for q in queries])
    phase1_elapsed = time.perf_counter() - t_phase1
    raw_by_id = {r["id"]: r for r in raw_responses}
    print(f"  Done in {phase1_elapsed:.1f}s\n")

    # ── Phase 2: score all responses ──────────────────────────────────────
    print(f"  Phase 2 — Scoring {total} responses …")
    t_phase2 = time.perf_counter()

    query_by_id = {q["id"]: q for q in queries}

    async def score_one(raw: Dict) -> Dict:
        q   = query_by_id[raw["id"]]
        txt = raw["text"]

        rtype = infer_response_type(txt)
        type_score, type_note = check_response_type_baseline(q, rtype)

        # Build a pseudo-response dict for the quality judge
        # (no structured products — GPT gives free text only)
        pseudo_resp = {
            "response_type": rtype,
            "message": txt[:500],
            "recommendations": [],
        }
        quality_score, reason, usage = await score_quality_async(oai, q, pseudo_resp)

        # brand/filter/stock: N/A — GPT has no structured output
        final = compute_final_score(
            type_score=type_score,
            brand_score=None,
            filter_score=None,
            quality_score=quality_score,
            stock_score=None,
        )

        return {
            "id": q["id"],
            "group": q["group"],
            "label": q["label"],
            "message": q["message"],
            "score": final,
            "type_score": type_score,
            "brand_score": None,
            "filter_score": None,
            "stock_score": None,
            "quality_score": quality_score,
            "type_note": type_note,
            "brand_note": "N/A (no structured output)",
            "filter_note": "N/A (no structured output)",
            "stock_note": "N/A (no structured output)",
            "reason": reason,
            "elapsed_ms": raw["elapsed_ms"],
            "response_type": rtype,
            "n_recs": 0,
            "baseline_text": txt[:600],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    scored = await asyncio.gather(*[score_one(r) for r in raw_responses])
    phase2_elapsed = time.perf_counter() - t_phase2
    scored = sorted(scored, key=lambda r: r["id"])
    print(f"  Done in {phase2_elapsed:.1f}s\n")

    # ── Per-query output ──────────────────────────────────────────────────
    print(f"{'─'*74}")
    for r in scored:
        status = f"{GREEN}PASS{RST}" if r["score"] >= PASS_THRESHOLD else f"{RED}FAIL{RST}"
        print(
            f"  Q{r['id']:2d} [{r['group']:15s}] {status}  "
            f"score={color_score(r['score'])}  "
            f"type={r['type_score']:.2f}  "
            f"qual={r['quality_score']:.2f}  "
            f"[{r['response_type']:15s}] {r['elapsed_ms']:5d}ms"
        )
        print(f"       {r['type_note']}")
        if verbose:
            print(f"       reason: {r['reason']}")
            print(f"       GPT: {r['baseline_text'][:120].replace(chr(10), ' ')}")
    print(f"{'─'*74}\n")

    # ── Summary ───────────────────────────────────────────────────────────
    def stats(subset):
        if not subset:
            return 0, 0.0, 0.0, 0.0
        n = len(subset)
        avg = sum(r["score"] for r in subset) / n
        pct = 100.0 * sum(1 for r in subset if r["score"] >= PASS_THRESHOLD) / n
        type_acc = 100.0 * sum(1 for r in subset if r["type_score"] == 1.0) / n
        return n, avg, pct, type_acc

    specified = [r for r in scored if query_by_id[r["id"]].get("expect_recs_on_first")]
    underspec  = [r for r in scored if not query_by_id[r["id"]].get("expect_recs_on_first")]
    ns, as_, ps, ts = stats(specified)
    nu, au, pu, tu  = stats(underspec)
    na, aa, pa, ta  = stats(scored)

    hdr = f"  {'Category':<28} {'N':>4}  {'Avg':>7}  {'Pass%':>7}  {'TypeAcc%':>9}"
    sep = f"  {'─'*60}"
    print(f"\n{BOLD}  {'GPT-4o-mini Baseline — G-Eval':^60}{RST}\n")
    print(f"  Results by Difficulty")
    print(hdr); print(sep)
    print(f"  {'Specified (expect recs)':<28} {ns:>4}  {as_:>7.3f}  {ps:>6.1f}%  {ts:>8.1f}%")
    print(f"  {'Underspecified (vague)':<28} {nu:>4}  {au:>7.3f}  {pu:>6.1f}%  {tu:>8.1f}%")
    print(sep)
    print(f"  {'All':<28} {na:>4}  {aa:>7.3f}  {pa:>6.1f}%  {ta:>8.1f}%")

    # Per-group
    groups = sorted(set(r["group"] for r in scored))
    print(f"\n  Per-group breakdown")
    print(f"  {'Group':<20} {'N':>4}  {'Avg':>7}  {'Pass%':>7}  {'TypeAcc%':>9}")
    print(f"  {'─'*52}")
    for g in groups:
        subset = [r for r in scored if r["group"] == g]
        n, avg, pct, tacc = stats(subset)
        print(f"  {g:<20} {n:>4}  {avg:>7.3f}  {pct:>6.1f}%  {tacc:>8.1f}%")

    # Cost
    total_prompt     = sum(r.get("prompt_tokens", 0) for r in scored)
    total_completion = sum(r.get("completion_tokens", 0) for r in scored)
    cost_usd = (total_prompt * 0.150 + total_completion * 0.600) / 1_000_000
    avg_ms = sum(r.get("elapsed_ms", 0) for r in scored) / len(scored) if scored else 0
    print(f"\n  {'─'*60}")
    print(f"  Judge tokens:  {total_prompt:,} prompt + {total_completion:,} completion")
    print(f"  Total cost:    ${cost_usd:.4f} USD  (gpt-4o-mini rates)")
    print(f"  Baseline latency: avg={avg_ms:.0f}ms")
    print(f"  Total time: baseline={phase1_elapsed:.1f}s + scoring={phase2_elapsed:.1f}s\n")
    print(f"  {YEL}NOTE: brand/filter/stock scores are N/A — GPT has no structured output.{RST}")
    print(f"  {YEL}Compare quality_score and overall score (type+quality only) for fair delta.{RST}\n")

    # ── Save ─────────────────────────────────────────────────────────────
    if save_path:
        output = {
            "version": "gpt_baseline_v1",
            "model": "gpt-4o-mini",
            "threshold": PASS_THRESHOLD,
            "note": (
                "brand/filter/stock scores are N/A — GPT has no structured output. "
                "Compare quality_score column for fair paper delta."
            ),
            "summary": {
                "specified":      {"n": ns, "avg_score": round(as_, 4), "pass_pct": round(ps, 1)},
                "underspecified": {"n": nu, "avg_score": round(au, 4), "pass_pct": round(pu, 1)},
                "all":            {"n": na, "avg_score": round(aa, 4), "pass_pct": round(pa, 1)},
                "type_accuracy":  round(ta, 1),
            },
            "cost": {
                "judge_model": "gpt-4o-mini",
                "baseline_model": "gpt-4o-mini",
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "cost_usd": round(cost_usd, 6),
                "avg_baseline_latency_ms": round(avg_ms),
            },
            "results": scored,
        }
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to: {save_path}")

    return scored


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPT-4o-mini shopping baseline for IDSS G-Eval paper comparison"
    )
    parser.add_argument("--save",        help="Path to save JSON results")
    parser.add_argument("--group",       help="Only run queries from this group")
    parser.add_argument("--query",       type=int, help="Only run query with this ID")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent GPT calls (default 8)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print GPT response text and reason per query")
    args = parser.parse_args()

    queries = list(QUERIES)
    if args.group:
        queries = [q for q in queries if q["group"] == args.group]
    if args.query:
        queries = [q for q in queries if q["id"] == args.query]

    if not queries:
        print("No queries match the given filters.")
        sys.exit(1)

    asyncio.run(run_baseline(
        queries=queries,
        save_path=args.save,
        concurrency=args.concurrency,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
