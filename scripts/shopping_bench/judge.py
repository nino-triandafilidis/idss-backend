"""
shopping_bench/judge.py — GPT-4o-mini quality judge for IDSS-Shopping-Bench.

Evaluates the full conversation transcript on three qualitative criteria:
  1. Interview intelligence (0-4): asked smart questions when vague; skipped when fully specified
  2. Recommendation relevance (0-3): products matched stated need
  3. Helpfulness & tone (0-3): clear, concise, actionable

Total: 0-10 (normalized to 0-1 for scoring).

Hard constraints are already checked deterministically in evaluator.py and are
NOT re-assessed here.  The judge focuses only on what the LLM can judge better.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

# Judge always uses gpt-4o-mini — changing this would break cross-run comparability.
JUDGE_MODEL = "gpt-4o-mini"

SHOPPING_BENCH_JUDGE_SYSTEM = """\
You are an expert evaluator for a laptop shopping assistant (IDSS).
You will be given a multi-turn conversation between a user and IDSS.
Evaluate the conversation on exactly three criteria. Hard constraints (brand, budget) \
are verified separately — do NOT penalise for those here.

CRITERIA:

1. Interview Intelligence (0–4 points)
   4 = Asked exactly the right clarifying questions when the query was vague; \
skipped questions when user gave full details upfront
   3 = Asked mostly relevant questions; minor gaps
   2 = Asked some questions but missed key ones, or asked unnecessary questions \
when not needed
   1 = Asked irrelevant questions or always asked even when unnecessary
   0 = Did not ask any questions despite a vague query requiring clarification

2. Recommendation Relevance (0–3 points)
   3 = Products clearly match the user's use case, tier, and preferences
   2 = Products mostly match; minor misalignment
   1 = Products only weakly match (wrong category, wrong tier)
   0 = Products do not match user needs, or no products given when they should be

3. Helpfulness & Tone (0–3 points)
   3 = Clear, concise, actionable; no filler; builds confidence
   2 = Mostly helpful; some verbosity or vagueness
   1 = Somewhat helpful but unclear or overly verbose
   0 = Unhelpful, confusing, or off-topic

RULES:
- Score each criterion independently
- Do not penalise for hard-constraint violations (brand/price) — those are checked elsewhere
- If IDSS appropriately entered interview mode for a vague query, that is correct behaviour
- Output ONLY valid JSON: {"score": <0-10>, "reason": "<10 words or fewer>"}
- No markdown, no extra keys, no explanation outside the JSON object
"""

SHOPPING_BENCH_JUDGE_USER = """\
Conversation transcript:
{transcript}

Task description: {task_description}

Score the conversation. Output only JSON: {{"score": <0-10>, "reason": "<10 words>"}}\
"""


def format_transcript(conversation_history: List[Dict]) -> str:
    """Format a conversation history list into a readable transcript string.

    conversation_history: [{"role": "user"|"assistant", "content": str}, ...]
    """
    lines = []
    for turn in conversation_history:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


async def judge_conversation(
    conversation_history: List[Dict],
    task_description: str,
    oai,  # AsyncOpenAI instance — passed in to avoid import-time dependency
) -> Tuple[float, str, int]:
    """Call GPT-4o-mini to judge a full conversation transcript.

    Returns:
      (normalized_score, reason, raw_score_0_to_10)
      normalized_score is raw_score / 10.0
    """
    transcript = format_transcript(conversation_history)
    user_content = SHOPPING_BENCH_JUDGE_USER.format(
        transcript=transcript,
        task_description=task_description,
    )

    response = await oai.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SHOPPING_BENCH_JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        max_tokens=60,
        temperature=0.0,  # deterministic — same judge model as G-Eval for consistency
    )

    raw_text = response.choices[0].message.content or ""
    score, reason = _parse_judge_output(raw_text)
    return score / 10.0, reason, score


def _parse_judge_output(raw: str) -> Tuple[int, str]:
    """Parse JSON judge output into (score, reason).

    Returns (0, "parse error") if parsing fails.
    """
    raw = raw.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        score = int(data.get("score", 0))
        score = max(0, min(10, score))  # clamp to [0, 10]
        reason = str(data.get("reason", ""))[:120]
        return score, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        # Try extracting a number from the text as fallback
        m = re.search(r'"score"\s*:\s*(\d+)', raw)
        if m:
            return min(10, int(m.group(1))), "parse fallback"
        return 0, f"parse error: {raw[:60]}"
