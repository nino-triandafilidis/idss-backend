"""
shopping_bench/simulator.py — LLM user simulator for IDSS-Shopping-Bench.

Simulates a human shopper interacting with IDSS across multiple turns.
Turn 0 always returns the task's initial_message (no LLM call).
Turns 1+ call GPT-4o-mini to generate the next user message based on:
  - The task persona and constraints
  - The conversation history
  - The last IDSS response (was it a question? did it recommend products?)

Design principles:
  - Keep turns short (≤2 sentences) to mimic a real chat user
  - Only reveal constraints when the task's clarification_answers suggest it
  - If agent asks a clarifying question, answer from clarification_answers
  - If agent gave good recommendations, express satisfaction or ask to add to cart
  - Never break character or mention being a test
"""

from __future__ import annotations

from typing import Dict, List, Optional

from shopping_bench.tasks import ShoppingTask

# Simulator always uses gpt-4o-mini regardless of the system under test.
# This keeps simulator behavior constant across different IDSS model configurations.
SIMULATOR_MODEL = "gpt-4o-mini"

SIMULATOR_SYSTEM = """\
You are simulating a real human user chatting with a laptop shopping assistant.
Your job is to respond naturally as the user described in the persona below.

RULES:
- Stay completely in character — never mention you are a test or simulation
- Keep each response to 1-2 sentences maximum
- Only share information when the assistant asks for it or when it would naturally come up
- If the assistant asks a clarifying question, give ONE of the answers from the hints provided
- If the assistant recommends laptops that seem good, express interest or satisfaction
- If you are satisfied with recommendations, you may ask to add one to your cart
- If the assistant asks something not covered in your hints, say "I'm flexible on that"
- Do not ask questions back unless you genuinely need clarification about the assistant's response
- Natural typos and casual language are fine — you are a real user

PERSONA:
{persona}

YOUR CONSTRAINTS (do NOT reveal all at once — share naturally as the conversation unfolds):
{constraints_summary}

CLARIFICATION HINTS (use these when the assistant asks):
{clarification_hints}
"""

SIMULATOR_USER_PROMPT = """\
CONVERSATION SO FAR:
{history_text}

ASSISTANT'S LAST MESSAGE:
{last_message}

TASK CONTEXT:
Turn number: {turn_number}
The assistant seems to be: {last_response_type}

What does the user say next? Write ONLY the user's reply (1-2 sentences, no quotation marks).
"""


def get_first_turn(task: ShoppingTask) -> str:
    """Return the initial user message. Always deterministic — no LLM call."""
    return task.initial_message


def _build_constraints_summary(task: ShoppingTask) -> str:
    """Build a readable summary of the task's hard constraints for the simulator prompt."""
    parts = []
    for c in task.success_criteria:
        if c.check_type == "max_price_cents":
            parts.append(f"- Budget limit: ${c.value / 100:.0f}")
        elif c.check_type == "excluded_brand":
            parts.append(f"- Do NOT want any {c.value} products")
        elif c.check_type == "min_ram_gb":
            parts.append(f"- Need at least {c.value}GB RAM")
        elif c.check_type == "min_storage_gb":
            parts.append(f"- Need at least {c.value}GB storage")
        elif c.check_type == "cart_action":
            parts.append("- Plan to add a product to cart once satisfied with recommendations")
        elif c.check_type == "response_type":
            if c.value == "question":
                parts.append("- Your initial query is intentionally vague")
    return "\n".join(parts) if parts else "- No hard constraints (general shopping query)"


def _build_clarification_hints(task: ShoppingTask) -> str:
    """Format clarification_answers as hints for the simulator prompt."""
    if not task.clarification_answers:
        return "- (no specific hints; say 'I'm flexible' if asked)"
    lines = []
    for question_hint, answer in task.clarification_answers.items():
        lines.append(f"- If asked about {question_hint}: say '{answer}'")
    return "\n".join(lines)


def _format_history(conversation_history: List[Dict]) -> str:
    """Format conversation history list for the simulator prompt."""
    lines = []
    for turn in conversation_history:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        prefix = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines) if lines else "(no previous turns)"


def _describe_response_type(last_response: Dict) -> str:
    """Return a short description of what the agent's last response was doing."""
    rt = (last_response.get("response_type") or "").lower()
    descriptions = {
        "question":       "asking you a clarifying question",
        "recommendation": "recommending specific laptop products",
        "comparison":     "comparing multiple laptops",
        "cart":           "confirming a cart action",
        "answer":         "answering a question",
    }
    return descriptions.get(rt, f"responding (type: {rt or 'unknown'})")


def build_simulator_prompt(
    task: ShoppingTask,
    turn_number: int,
    conversation_history: List[Dict],
    last_idss_response: Dict,
) -> List[Dict]:
    """Build the GPT-4o-mini messages list for the simulator.

    Returns a list of {"role": ..., "content": ...} dicts.
    Exposed as a module-level function so tests can verify the prompt structure
    without making any LLM calls.
    """
    system_content = SIMULATOR_SYSTEM.format(
        persona=task.persona,
        constraints_summary=_build_constraints_summary(task),
        clarification_hints=_build_clarification_hints(task),
    )

    # Extract text from last IDSS response
    last_message = (
        last_idss_response.get("message")
        or last_idss_response.get("response")
        or last_idss_response.get("content")
        or "(no message)"
    )
    # Truncate very long messages to keep the prompt focused
    if len(last_message) > 800:
        last_message = last_message[:800] + "..."

    user_content = SIMULATOR_USER_PROMPT.format(
        history_text=_format_history(conversation_history),
        last_message=last_message,
        turn_number=turn_number,
        last_response_type=_describe_response_type(last_idss_response),
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


async def next_simulator_turn(
    task: ShoppingTask,
    turn_number: int,
    conversation_history: List[Dict],
    last_idss_response: Dict,
    oai,  # AsyncOpenAI instance — passed in to avoid import-time dependency
) -> str:
    """Generate the next simulated user message using GPT-4o-mini.

    turn_number is the index of the turn being generated (1-based after turn 0).
    conversation_history contains all turns so far (user and assistant alternating).
    last_idss_response is the raw IDSS /chat JSON response from the previous turn.
    """
    messages = build_simulator_prompt(task, turn_number, conversation_history, last_idss_response)

    response = await oai.chat.completions.create(
        model=SIMULATOR_MODEL,
        messages=messages,
        max_tokens=80,
        temperature=0.3,  # slight randomness to mimic human variation
    )

    text = response.choices[0].message.content or ""
    # Strip quotes if the model wrapped the response in them
    text = text.strip().strip('"').strip("'")
    return text
