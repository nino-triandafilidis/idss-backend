"""
shopping_bench/runner.py — Orchestration state machine for IDSS-Shopping-Bench.

Runs one task: simulator ↔ IDSS loop → evaluate → judge.
All I/O goes through httpx (IDSS) and AsyncOpenAI (simulator + judge).

Scoring formula: final_score = 0.55 * constraint_score + 0.45 * judge_score
A task passes when final_score >= 0.5.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

from shopping_bench.evaluator import evaluate_all_constraints, _flatten_recs
from shopping_bench.judge import judge_conversation
from shopping_bench.simulator import get_first_turn, next_simulator_turn
from shopping_bench.tasks import HardConstraint, ShoppingTask

# Scoring weights (must sum to 1.0)
CONSTRAINT_WEIGHT = 0.55
JUDGE_WEIGHT = 0.45
PASS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Per-turn state snapshot."""
    turn_number: int
    user_message: str
    assistant_message: str
    response_type: str
    products: List[Dict]              # up to 6 products, {brand, name, price_value}
    constraint_scores: Dict[str, Tuple[float, str]]  # check_type → (score, note)
    elapsed_s: float


@dataclass
class ScenarioResult:
    """Final aggregated result for one task run."""
    task_id: str
    task_category: str
    turn_results: List[TurnResult]
    final_constraint_score: float    # avg over all HardConstraints at final turn
    judge_score: float               # 0-1 from GPT-4o-mini judge
    final_score: float               # CONSTRAINT_WEIGHT*constraint + JUDGE_WEIGHT*judge
    passed: bool                     # final_score >= PASS_THRESHOLD
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None      # set if the run aborted with an exception


# ---------------------------------------------------------------------------
# IDSS /chat call
# ---------------------------------------------------------------------------

async def _call_idss(
    message: str,
    session_id: str,
    idss_url: str,
    http_client: httpx.AsyncClient,
) -> Dict:
    """POST /chat with session_id and return JSON response dict.

    Returns {"error": str} if the request fails.
    """
    try:
        resp = await http_client.post(
            f"{idss_url.rstrip('/')}/chat",
            json={"message": message, "session_id": session_id},
            timeout=90.0,  # agent calls LLM ~2-3x per turn; each can take 10-20s
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.RequestError as e:
        return {"error": f"Request error: {str(e)[:200]}"}


def _extract_products(response: Dict) -> List[Dict]:
    """Extract top-6 products from an IDSS response for logging."""
    flat = _flatten_recs(response)
    trimmed = []
    for p in flat[:6]:
        trimmed.append({
            "brand":       p.get("brand") or "",
            "name":        p.get("name") or p.get("title") or "",
            "price_value": p.get("price_value"),
        })
    return trimmed


# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------

async def run_task(
    task: ShoppingTask,
    idss_url: str,
    http_client: httpx.AsyncClient,
    oai,                        # AsyncOpenAI instance
    sem: asyncio.Semaphore,
) -> ScenarioResult:
    """Run one ShoppingTask and return a ScenarioResult.

    Uses sem to limit concurrent IDSS requests.
    """
    session_id = str(uuid.uuid4())
    conversation_history: List[Dict] = []
    turn_results: List[TurnResult] = []
    notes: List[str] = []
    last_idss_response: Dict = {}

    async with sem:
        try:
            for turn_idx in range(task.max_turns):
                t_start = time.monotonic()

                # --- Generate user message ---------------------------------
                if turn_idx == 0:
                    user_msg = get_first_turn(task)
                else:
                    user_msg = await next_simulator_turn(
                        task,
                        turn_number=turn_idx,
                        conversation_history=conversation_history,
                        last_idss_response=last_idss_response,
                        oai=oai,
                    )

                conversation_history.append({"role": "user", "content": user_msg})

                # --- Call IDSS --------------------------------------------
                idss_response = await _call_idss(user_msg, session_id, idss_url, http_client)
                last_idss_response = idss_response

                if "error" in idss_response:
                    notes.append(f"Turn {turn_idx}: IDSS error: {idss_response['error']}")
                    break

                assistant_msg = (
                    idss_response.get("message")
                    or idss_response.get("response")
                    or idss_response.get("content")
                    or ""
                )
                conversation_history.append({"role": "assistant", "content": assistant_msg})

                # --- Evaluate intermediate checks -------------------------
                inter_constraints = [
                    c for (turn_num, c) in task.intermediate_checks
                    if turn_num == turn_idx
                ]
                if inter_constraints:
                    _, inter_scores = evaluate_all_constraints(inter_constraints, idss_response)
                    for key, (score, note) in inter_scores.items():
                        if score < 1.0:
                            notes.append(
                                f"Turn {turn_idx} intermediate check FAILED [{key}]: {note}"
                            )

                # --- Record turn -----------------------------------------
                elapsed = time.monotonic() - t_start
                products = _extract_products(idss_response)

                # Run all success_criteria on this turn for the turn-level record
                _, turn_scores = evaluate_all_constraints(
                    task.success_criteria, idss_response
                )

                turn_results.append(TurnResult(
                    turn_number=turn_idx,
                    user_message=user_msg,
                    assistant_message=assistant_msg,
                    response_type=idss_response.get("response_type") or "",
                    products=products,
                    constraint_scores=turn_scores,
                    elapsed_s=elapsed,
                ))

                # --- Check terminal condition -----------------------------
                # Stop early if all hard constraints are already satisfied
                if _all_constraints_met(task.success_criteria, idss_response):
                    notes.append(f"All constraints satisfied at turn {turn_idx} — stopping early")
                    break

        except Exception as e:
            notes.append(f"Runner exception: {type(e).__name__}: {str(e)[:200]}")
            # Return a zero-score result on unhandled exception
            return ScenarioResult(
                task_id=task.id,
                task_category=task.category,
                turn_results=turn_results,
                final_constraint_score=0.0,
                judge_score=0.0,
                final_score=0.0,
                passed=False,
                notes=notes,
                error=str(e),
            )

    # --- Final evaluation (last IDSS response) ----------------------------
    if not turn_results:
        return ScenarioResult(
            task_id=task.id,
            task_category=task.category,
            turn_results=[],
            final_constraint_score=0.0,
            judge_score=0.0,
            final_score=0.0,
            passed=False,
            notes=notes + ["No turns completed"],
        )

    final_response = last_idss_response
    final_constraint_score, final_scores = evaluate_all_constraints(
        task.success_criteria, final_response
    )

    for key, (score, note) in final_scores.items():
        if score < 1.0:
            notes.append(f"Final constraint FAILED [{key}]: {note}")

    # --- Judge scoring ----------------------------------------------------
    try:
        judge_score, judge_reason, raw_score = await judge_conversation(
            conversation_history, task.description, oai
        )
        notes.append(f"Judge: {raw_score}/10 — {judge_reason}")
    except Exception as e:
        judge_score = 0.0
        notes.append(f"Judge error: {str(e)[:100]}")

    # --- Final score ------------------------------------------------------
    final_score = CONSTRAINT_WEIGHT * final_constraint_score + JUDGE_WEIGHT * judge_score
    passed = final_score >= PASS_THRESHOLD

    return ScenarioResult(
        task_id=task.id,
        task_category=task.category,
        turn_results=turn_results,
        final_constraint_score=final_constraint_score,
        judge_score=judge_score,
        final_score=final_score,
        passed=passed,
        notes=notes,
    )


def _all_constraints_met(constraints: List[HardConstraint], response: Dict) -> bool:
    """Return True if all HardConstraints are satisfied by this response."""
    for c in constraints:
        from shopping_bench.evaluator import evaluate_response
        score, _ = evaluate_response(c, response)
        if score < 1.0:
            return False
    return True
