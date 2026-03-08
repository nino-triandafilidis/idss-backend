"""
query_rewriter.py — Disambiguates and expands user queries before slot extraction.

Responsibilities:
1. Accessory disambiguation (moved from universal_agent.py)
2. Context-aware expansion: fills vague references using known session filters
3. Common-sense enrichment: "for my son" → annotates [use_case: school]

Called at the top of UniversalAgent.process_message() before domain detection
and slot extraction.  Returns either a rewritten query string or a clarifying
question to show the user (is_clarification=True).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewriteResult:
    rewritten: str                                  # expanded / clarified query (may equal original)
    clarifying_question: Optional[str] = None       # if set, show this to user instead of proceeding
    quick_replies: Optional[List[str]] = field(default_factory=list)
    is_clarification: bool = False


# ---------------------------------------------------------------------------
# Keyword sets (mirrored from former universal_agent.py inline block)
# ---------------------------------------------------------------------------

_ACCESSORY_KEYWORDS: frozenset = frozenset({
    "bag", "sleeve", "stand", "dock", "docking", "charger", "adapter",
    "cable", "mouse", "webcam", "hub", "port", "hdmi", "usb",
    "upgrade", "parts", "peripheral", "accessories", "case", "cover",
})

_SPEC_SIGNALS: frozenset = frozenset({
    "ram", "gb", "tb", "ssd", "nvme", "cpu", "gpu", "processor",
    "battery", "storage", "performance", "budget", "price",
    "gaming", "coding", "development", "programming", "editing",
    "under", "laptop", "notebook", "chromebook",
})

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite(
    message: str,
    session_history: List[Dict[str, Any]],
    domain: str,
    current_filters: Dict[str, Any],
    question_count: int = 0,
) -> RewriteResult:
    """
    Main entry point.  Returns a RewriteResult with the (possibly expanded)
    query or a clarifying question to ask the user.

    Args:
        message:         Raw user message.
        session_history: Prior conversation turns (dicts with 'role'/'content').
        domain:          Active domain ('laptops', 'vehicles', 'books', or '').
        current_filters: Slot values extracted so far (e.g. {'budget': 1200}).
        question_count:  How many questions the agent has already asked.
    """
    msg_lower = message.lower().strip()
    msg_words = set(re.sub(r"[^a-z0-9 ]", " ", msg_lower).split())

    # 1. Accessory / subtype ambiguity check
    accessory_hit = msg_words & _ACCESSORY_KEYWORDS
    spec_hit = msg_words & _SPEC_SIGNALS
    domain_label = {"laptops": "laptop", "vehicles": "vehicle", "books": "book"}.get(domain, domain)

    if (accessory_hit
            and question_count == 0
            and not spec_hit
            and len(msg_words) < 20
            and domain_label):
        example = next(iter(accessory_hit))
        clarify = (
            f"Are you looking for a **{domain_label}** itself, or a "
            f"**{domain_label} accessory** (like a {example})?"
        )
        return RewriteResult(
            rewritten=message,
            clarifying_question=clarify,
            quick_replies=[f"The {domain_label} itself", f"A {domain_label} accessory"],
            is_clarification=True,
        )

    # 2. Context-aware expansion: fill vague references from known filters
    expanded = _expand_with_context(message, msg_lower, current_filters)

    # 3. Common-sense enrichment
    expanded = _commonsense_enrich(expanded, msg_lower)

    return RewriteResult(rewritten=expanded)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _expand_with_context(
    message: str,
    msg_lower: str,
    filters: Dict[str, Any],
) -> str:
    """
    Expand vague comparative references using slot values already known.

    Examples:
        "show me cheaper ones" + {price_max_cents: 120000}
            → "show me cheaper ones [under $1200]"
        "different brand" + {brand: "Dell"}
            → "different brand [not Dell]"
    """
    expanded = message

    # "cheaper / more affordable" → append current budget cap
    if re.search(r"\b(cheaper|less expensive|more affordable)\b", msg_lower):
        budget = filters.get("price_max_cents") or filters.get("budget")
        if budget:
            try:
                raw = int(budget)
                price = raw // 100 if raw > 10_000 else raw   # cents → dollars if large
                expanded += f" [under ${price}]"
            except (TypeError, ValueError):
                pass

    # "different / another brand" → append the brand to avoid
    if re.search(r"\b(different brand|another brand|not that brand)\b", msg_lower):
        brand = filters.get("brand")
        if brand:
            expanded += f" [not {brand}]"

    return expanded


def _commonsense_enrich(message: str, msg_lower: str) -> str:
    """
    Annotate implicit intent signals so the downstream LLM extractor
    can pick them up even without explicit phrasing.

    These annotations are enclosed in [] and will be parsed by the
    criteria extractor just like any other slot hint.
    """
    enriched = message

    # "for my son / daughter / kid / child / student" → school use case
    if re.search(
        r"\bfor\s+(my\s+)?(son|daughter|kid|child|nephew|niece|student)\b",
        msg_lower,
    ):
        enriched += " [use_case: school]"

    # "for work / office / business" (only if not already explicit)
    if (re.search(r"\bfor\s+(work|office|business)\b", msg_lower)
            and "use_case" not in msg_lower):
        enriched += " [use_case: business]"

    return enriched
