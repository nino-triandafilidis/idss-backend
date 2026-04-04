"""
shopping_bench/tasks.py — Task schema and 20 task definitions for IDSS-Shopping-Bench.

Task categories (4 each):
  budget              — price constraint enforcement
  brand_exclusion     — excluded brand enforcement (incl. mid-turn changes)
  multi_constraint    — 3+ constraints accumulated across turns
  interview_elicitation — vague query → IDSS must ask → user answers
  cart_action         — final: user requests add-to-cart
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

VALID_CHECK_TYPES = {
    "max_price_cents",   # all recommended products must be ≤ value (int, cents)
    "excluded_brand",    # none of the recommended products may be this brand (str)
    "min_ram_gb",        # all recommended products must have RAM ≥ value (int)
    "min_storage_gb",    # all recommended products must have storage ≥ value (int)
    "response_type",     # IDSS response_type field must equal value (str, e.g. "question")
    "cart_action",       # response must contain a cart action (bool True means present)
}


@dataclass
class HardConstraint:
    """Deterministically checkable requirement for a task.

    check_type values:
      - "max_price_cents":  all products must have price ≤ value (int cents)
      - "excluded_brand":   no product may match this brand (str, case-insensitive)
      - "min_ram_gb":       all products must have RAM ≥ value (int GB)
      - "min_storage_gb":   all products must have storage ≥ value (int GB)
      - "response_type":    IDSS response_type field must equal value (str)
      - "cart_action":      response contains a cart confirmation (bool True)
    """
    check_type: str
    value: Any

    def __post_init__(self) -> None:
        if self.check_type not in VALID_CHECK_TYPES:
            raise ValueError(
                f"Unknown check_type {self.check_type!r}. Valid: {sorted(VALID_CHECK_TYPES)}"
            )


@dataclass
class SoftGoal:
    """Qualitative goal passed verbatim to the LLM judge prompt."""
    description: str


@dataclass
class ShoppingTask:
    """One IDSS-Shopping-Bench scenario.

    Fields
    ------
    id                    Unique identifier (e.g. "brand_excl_01")
    category              One of: budget | brand_exclusion | multi_constraint |
                          interview_elicitation | cart_action
    description           Human-readable scenario summary
    persona               1-2 sentence persona passed to the LLM user simulator
    initial_message       Deterministic first user turn (no LLM call needed)
    max_turns             Safety limit for the simulator loop (default 6)
    success_criteria      Hard constraints checked deterministically at final turn
    soft_goals            Qualitative goals passed to the LLM judge
    clarification_answers Dict mapping expected agent question → answer to give
                          (used by simulator when IDSS asks a clarifying question)
    intermediate_checks   [(turn_number, HardConstraint)] — constraints to check
                          at a specific turn, not just the final turn
    """
    id: str
    category: str
    description: str
    persona: str
    initial_message: str
    success_criteria: List[HardConstraint]
    soft_goals: List[SoftGoal] = field(default_factory=list)
    max_turns: int = 6
    clarification_answers: Dict[str, str] = field(default_factory=dict)
    intermediate_checks: List[Tuple[int, HardConstraint]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 20 Task definitions
# ---------------------------------------------------------------------------

TASKS: List[ShoppingTask] = [

    # ---- BUDGET (4 tasks) --------------------------------------------------

    ShoppingTask(
        id="budget_01",
        category="budget",
        description="Strict $800 ceiling; verify all recommended products are under budget.",
        persona="College student on a tight budget who needs a reliable laptop for classes.",
        initial_message="I need a laptop for college, my budget is $800. What do you recommend?",
        success_criteria=[
            HardConstraint("max_price_cents", 80_000),
        ],
        soft_goals=[
            SoftGoal("Agent recommends at least one product suitable for student use"),
            SoftGoal("Agent explains why the product fits the budget"),
        ],
    ),

    ShoppingTask(
        id="budget_02",
        category="budget",
        description="Low $500 ceiling; tests that IDSS doesn't recommend mid-range products.",
        persona="Budget-conscious buyer who must stay under $500 absolutely.",
        initial_message="What's the best laptop I can get for under $500?",
        success_criteria=[
            HardConstraint("max_price_cents", 50_000),
        ],
        soft_goals=[
            SoftGoal("Agent acknowledges the tight budget and sets realistic expectations"),
        ],
    ),

    ShoppingTask(
        id="budget_03",
        category="budget",
        description="Budget revealed mid-conversation; tests that IDSS applies constraint after update.",
        persona="Young professional who initially asks broadly then reveals a budget limit.",
        initial_message="I'm looking for a good laptop for working from home.",
        clarification_answers={
            "budget": "I can spend up to $1200",
            "use": "Mainly email, video calls, and documents",
            "brand": "I'm flexible on brand",
        },
        max_turns=5,
        success_criteria=[
            HardConstraint("max_price_cents", 120_000),
        ],
        soft_goals=[
            SoftGoal("Agent asks clarifying questions before recommending"),
            SoftGoal("Recommendations align with work-from-home use case"),
        ],
    ),

    ShoppingTask(
        id="budget_04",
        category="budget",
        description="'Over $X' direction — user wants something ABOVE $1000 (not below).",
        persona="Power user willing to invest in a high-end machine.",
        initial_message="I want a premium laptop, something over $1000. What's good?",
        success_criteria=[
            # IDSS returns "recommendations" (plural) for product results — see chat_endpoint.py.
            # Constraint: IDSS should recommend directly, not ask, since the intent is clear.
            HardConstraint("response_type", "recommendations"),
        ],
        soft_goals=[
            SoftGoal("Agent recommends higher-end products, not budget laptops"),
            SoftGoal("Agent does not cap recommendations at $1000"),
        ],
    ),

    # ---- BRAND EXCLUSION (4 tasks) ----------------------------------------

    ShoppingTask(
        id="brand_excl_01",
        category="brand_exclusion",
        description="Simple HP exclusion — verify no HP products recommended.",
        persona="Office worker who had bad experience with HP and refuses to buy HP again.",
        initial_message="I need a laptop for work, budget $900, but please no HP.",
        success_criteria=[
            HardConstraint("excluded_brand", "HP"),
            HardConstraint("max_price_cents", 90_000),
        ],
        soft_goals=[
            SoftGoal("Agent explicitly acknowledges the HP exclusion"),
            SoftGoal("All recommendations are non-HP brands"),
        ],
    ),

    ShoppingTask(
        id="brand_excl_02",
        category="brand_exclusion",
        description="Dell exclusion with budget — tests SQL-level brand filtering.",
        persona="Tech-savvy buyer who dislikes Dell for quality reasons.",
        initial_message="Looking for a laptop under $700, nothing from Dell please.",
        success_criteria=[
            HardConstraint("excluded_brand", "Dell"),
            HardConstraint("max_price_cents", 70_000),
        ],
        soft_goals=[
            SoftGoal("No Dell products appear in recommendations"),
        ],
    ),

    ShoppingTask(
        id="brand_excl_03",
        category="brand_exclusion",
        description="Mid-turn brand re-inclusion — user initially excludes HP, then says HP is fine.",
        persona="Indecisive shopper who changes their mind about HP after seeing options.",
        initial_message="I want a gaming laptop under $1000, no HP please.",
        clarification_answers={
            "brand": "Actually, HP is fine, I changed my mind.",
        },
        max_turns=6,
        success_criteria=[
            # After user says HP is fine, HP should no longer be excluded
            # We check that response_type is recommendation (not blocked by brand logic)
            HardConstraint("response_type", "recommendation"),
            HardConstraint("max_price_cents", 100_000),
        ],
        soft_goals=[
            SoftGoal("Agent accepts the brand un-exclusion gracefully"),
            SoftGoal("Final recommendations may include HP products"),
        ],
    ),

    ShoppingTask(
        id="brand_excl_04",
        category="brand_exclusion",
        description="Apple/Mac exclusion phrased as 'no Mac' — tests alias resolution.",
        persona="Windows user who specifically does not want a Mac.",
        initial_message="Need a laptop for video editing, budget $1500, but no Mac.",
        success_criteria=[
            HardConstraint("excluded_brand", "Apple"),
            HardConstraint("max_price_cents", 150_000),
        ],
        soft_goals=[
            SoftGoal("No Apple/MacBook products appear in recommendations"),
            SoftGoal("Agent recommends Windows laptops suitable for video editing"),
        ],
    ),

    # ---- MULTI CONSTRAINT (4 tasks) ----------------------------------------

    ShoppingTask(
        id="multi_01",
        category="multi_constraint",
        description="Budget + RAM + brand exclusion all given in one turn.",
        persona="Software developer who needs a powerful machine at a reasonable price.",
        initial_message=(
            "I'm a developer looking for a laptop: budget $1200, at least 16GB RAM, "
            "and no Acer please."
        ),
        success_criteria=[
            HardConstraint("max_price_cents", 120_000),
            HardConstraint("min_ram_gb", 16),
            HardConstraint("excluded_brand", "Acer"),
        ],
        soft_goals=[
            SoftGoal("Agent confirms all three constraints"),
            SoftGoal("Recommendations are developer-appropriate (coding, multiple tabs)"),
        ],
    ),

    ShoppingTask(
        id="multi_02",
        category="multi_constraint",
        description="Constraints added across 3 turns: first budget, then brand, then RAM.",
        persona="Methodical shopper who reveals requirements one at a time.",
        initial_message="I'm looking for a laptop under $1000.",
        clarification_answers={
            "brand": "No Lenovo, I've had issues with them.",
            "ram": "I need at least 16GB for my work.",
            "storage": "500GB or more would be great.",
        },
        max_turns=7,
        success_criteria=[
            HardConstraint("max_price_cents", 100_000),
            HardConstraint("excluded_brand", "Lenovo"),
        ],
        soft_goals=[
            SoftGoal("Agent accumulates constraints across turns correctly"),
            SoftGoal("No Lenovo products appear despite progressive constraint revelation"),
        ],
    ),

    ShoppingTask(
        id="multi_03",
        category="multi_constraint",
        description="Budget overwrite — user changes $1500 to $800 in turn 3.",
        persona="Shopper who gets a reality check on budget mid-conversation.",
        initial_message="I need a high-performance laptop, budget around $1500.",
        clarification_answers={
            "use": "Gaming and programming both.",
            "budget_update": "Actually, I can only spend $800. Can you redo with that?",
        },
        max_turns=6,
        success_criteria=[
            HardConstraint("max_price_cents", 80_000),  # updated budget must be enforced
        ],
        soft_goals=[
            SoftGoal("Agent acknowledges the budget reduction explicitly"),
            SoftGoal("Final recommendations all fit within the new $800 limit"),
        ],
    ),

    ShoppingTask(
        id="multi_04",
        category="multi_constraint",
        description="Three-constraint scenario: two brand exclusions + budget.",
        persona="Particular shopper with negative opinions on multiple brands.",
        initial_message=(
            "Help me find a laptop under $1100. I don't want HP or Dell — "
            "bad experiences with both."
        ),
        success_criteria=[
            HardConstraint("max_price_cents", 110_000),
            HardConstraint("excluded_brand", "HP"),
            HardConstraint("excluded_brand", "Dell"),
        ],
        soft_goals=[
            SoftGoal("Neither HP nor Dell products appear in recommendations"),
            SoftGoal("Agent acknowledges both exclusions"),
        ],
    ),

    # ---- INTERVIEW ELICITATION (4 tasks) ------------------------------------

    ShoppingTask(
        id="interview_01",
        category="interview_elicitation",
        description="Vague 'good laptop' query — IDSS should ask clarifying questions.",
        persona="First-time laptop buyer who doesn't know technical specs.",
        initial_message="I want to buy a good laptop.",
        clarification_answers={
            "use": "Mostly browsing, Netflix, and light work.",
            "budget": "Around $600.",
            "brand": "No preference.",
        },
        max_turns=6,
        success_criteria=[
            # First response should be a question, not a recommendation
            HardConstraint("response_type", "question"),
        ],
        soft_goals=[
            SoftGoal("Agent asks smart clarifying questions (use case and budget)"),
            SoftGoal("After clarifications, final recommendations match stated needs"),
        ],
        intermediate_checks=[
            (0, HardConstraint("response_type", "question")),  # turn 0 = first IDSS response
        ],
    ),

    ShoppingTask(
        id="interview_02",
        category="interview_elicitation",
        description="Gaming query without budget — IDSS should ask about budget.",
        persona="Teen gamer who knows they want gaming but hasn't thought about price.",
        initial_message="I want a gaming laptop.",
        clarification_answers={
            "budget": "I have about $900 to spend.",
            "brand": "Whatever is best for gaming.",
        },
        max_turns=5,
        success_criteria=[
            HardConstraint("response_type", "question"),
        ],
        soft_goals=[
            SoftGoal("Agent asks about budget or other specs before recommending"),
        ],
        intermediate_checks=[
            (0, HardConstraint("response_type", "question")),
        ],
    ),

    ShoppingTask(
        id="interview_03",
        category="interview_elicitation",
        description="'For my kid' query — age/use elicitation before recommending.",
        persona="Parent buying a laptop for their child starting high school.",
        initial_message="I need a laptop for my kid who is starting high school.",
        clarification_answers={
            "use": "School work, some YouTube and games.",
            "budget": "Under $500 please.",
            "age": "14 years old.",
        },
        max_turns=6,
        success_criteria=[
            HardConstraint("response_type", "question"),
        ],
        soft_goals=[
            SoftGoal("Agent asks clarifying questions about use case and budget"),
            SoftGoal("Final product suggestions are age-appropriate and durable"),
        ],
        intermediate_checks=[
            (0, HardConstraint("response_type", "question")),
        ],
    ),

    ShoppingTask(
        id="interview_04",
        category="interview_elicitation",
        description="Fully specified query after interview — IDSS should skip questions.",
        persona="Experienced tech buyer who gives all details upfront.",
        initial_message=(
            "I want a Windows laptop for software development, at least 16GB RAM, "
            "512GB SSD, budget $1000, no HP."
        ),
        success_criteria=[
            # IDSS returns "recommendations" (plural) — see chat_endpoint.py.
            # With 5 explicit criteria, IDSS should recommend on turn 0, not ask questions.
            HardConstraint("response_type", "recommendations"),
            HardConstraint("max_price_cents", 100_000),
            HardConstraint("excluded_brand", "HP"),
        ],
        soft_goals=[
            SoftGoal("Agent does not ask unnecessary clarifying questions"),
            SoftGoal("Recommendations match all stated specs"),
        ],
        # Check turn 0 immediately: if IDSS asks instead of recommending, flag it early
        # so the runner stops before the simulator can go off-script with checkout questions.
        intermediate_checks=[
            (0, HardConstraint("response_type", "recommendations")),
        ],
    ),

    # ---- CART ACTION (4 tasks) ---------------------------------------------

    ShoppingTask(
        id="cart_01",
        category="cart_action",
        description="User asks to add the first recommended laptop to cart.",
        persona="Decisive buyer ready to purchase after seeing the first recommendation.",
        initial_message="I need a laptop for office work under $700. What do you recommend?",
        clarification_answers={
            "add_to_cart": "Yes, add the first one to my cart please.",
        },
        max_turns=5,
        success_criteria=[
            HardConstraint("max_price_cents", 70_000),
            HardConstraint("cart_action", True),
        ],
        soft_goals=[
            SoftGoal("Agent adds the correct product to cart and confirms"),
        ],
    ),

    ShoppingTask(
        id="cart_02",
        category="cart_action",
        description="User selects specific product by name to add to cart.",
        persona="Shopper who compared options and chose a specific model.",
        initial_message="Show me laptops under $800 for light programming.",
        clarification_answers={
            "add_to_cart": "I'll take the second option. Please add it to my cart.",
        },
        max_turns=5,
        success_criteria=[
            HardConstraint("max_price_cents", 80_000),
            HardConstraint("cart_action", True),
        ],
        soft_goals=[
            SoftGoal("Agent confirms which product was added to cart by name"),
        ],
    ),

    ShoppingTask(
        id="cart_03",
        category="cart_action",
        description="Cart action after brand exclusion — HP-free laptop added to cart.",
        persona="Careful buyer who filters brands before purchasing.",
        initial_message="Find me a laptop under $900, no HP, for video calls and documents.",
        clarification_answers={
            "add_to_cart": "Perfect, add the top recommendation to my cart.",
        },
        max_turns=5,
        success_criteria=[
            HardConstraint("max_price_cents", 90_000),
            HardConstraint("excluded_brand", "HP"),
            HardConstraint("cart_action", True),
        ],
        soft_goals=[
            SoftGoal("Agent confirms the cart addition includes a non-HP product"),
        ],
    ),

    ShoppingTask(
        id="cart_04",
        category="cart_action",
        description="Multi-turn: user refines then adds to cart.",
        persona="Thoughtful buyer who refines requirements then commits.",
        initial_message="I need a laptop for data science work, budget $1300.",
        clarification_answers={
            "refine": "Actually, make sure it has at least 16GB RAM.",
            "add_to_cart": "Great, add the first one to my cart.",
        },
        max_turns=6,
        success_criteria=[
            HardConstraint("max_price_cents", 130_000),
            HardConstraint("cart_action", True),
        ],
        soft_goals=[
            SoftGoal("Agent incorporates the RAM refinement before cart add"),
            SoftGoal("Agent confirms the cart addition with product name"),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

TASKS_BY_ID: Dict[str, ShoppingTask] = {t.id: t for t in TASKS}
TASKS_BY_CATEGORY: Dict[str, List[ShoppingTask]] = {}
for _task in TASKS:
    TASKS_BY_CATEGORY.setdefault(_task.category, []).append(_task)

VALID_CATEGORIES = set(TASKS_BY_CATEGORY.keys())


def get_task(task_id: str) -> ShoppingTask:
    """Return task by ID, raise KeyError with helpful message if not found."""
    if task_id not in TASKS_BY_ID:
        raise KeyError(
            f"Unknown task ID {task_id!r}. Valid IDs: {sorted(TASKS_BY_ID)}"
        )
    return TASKS_BY_ID[task_id]


def get_tasks_for_category(category: str) -> List[ShoppingTask]:
    """Return all tasks in a category, raise KeyError if category unknown."""
    if category not in TASKS_BY_CATEGORY:
        raise KeyError(
            f"Unknown category {category!r}. Valid: {sorted(VALID_CATEGORIES)}"
        )
    return TASKS_BY_CATEGORY[category]
