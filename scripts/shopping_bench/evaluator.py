"""
shopping_bench/evaluator.py — Deterministic constraint checks for IDSS-Shopping-Bench.

All handlers are pure functions (no I/O, no LLM).
evaluate_response() dispatches to the right handler based on HardConstraint.check_type.

IDSS /chat response shape (relevant fields):
  {
    "response_type": "recommendations" | "question" | "research" | "compare",
    "recommendations": [
      {"brand": "HP", "name": "HP Envy 15", "price_value": 799.99, ...},
      ...
    ],
    "cart_items": [{"product_id": "...", "name": "...", "action": "add", ...}],
    ...
  }

NOTE on response_type values (confirmed from chat_endpoint.py line 286, 1375, 1453, 1615):
  IDSS always returns the PLURAL form "recommendations" for product results.
  Human-written tasks and specs commonly use the singular "recommendation".
  The evaluator normalizes both forms so either string works in HardConstraint.value.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from shopping_bench.tasks import HardConstraint

# ---------------------------------------------------------------------------
# Response-type normalization aliases.
# IDSS returns "recommendations" (plural); task specs often write "recommendation".
# Map both directions so HardConstraint.value can use either form.
# ---------------------------------------------------------------------------
_RESPONSE_TYPE_ALIASES: Dict[str, str] = {
    "recommendations": "recommendation",  # IDSS plural → canonical singular
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_response(
    constraint: HardConstraint,
    response: Dict,
) -> Tuple[float, str]:
    """Return (score, explanation) for one HardConstraint against an IDSS response.

    score is 0.0 or 1.0.  explanation is a short human-readable note.
    """
    handlers = {
        "max_price_cents":  _check_max_price,
        "excluded_brand":   _check_excluded_brand,
        "min_ram_gb":       _check_min_ram,
        "min_storage_gb":   _check_min_storage,
        "response_type":    _check_response_type,
        "cart_action":      _check_cart_action,
    }
    handler = handlers.get(constraint.check_type)
    if handler is None:
        return 0.0, f"Unknown check_type: {constraint.check_type!r}"
    return handler(constraint, response)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _flatten_recs(response: Dict) -> List[Dict]:
    """Flatten IDSS recommendations into a flat list of product dicts.

    IDSS sometimes returns grouped lists [[p1, p2], [p3]] or a flat list [p1, p2, p3].
    Returns [] if the field is absent or None.
    """
    raw = response.get("recommendations") or []
    flat: List[Dict] = []
    for item in raw:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, dict):
            flat.append(item)
    return flat


def _price_cents(product: Dict) -> Optional[int]:
    """Convert a product's price to integer cents.

    IDSS stores price as price_value (float dollars).
    Returns None if the field is absent or not numeric.
    """
    raw = product.get("price_value")
    if raw is None:
        return None
    try:
        dollars = float(raw)
        return int(round(dollars * 100))
    except (TypeError, ValueError):
        return None


def _brand_of(product: Dict) -> str:
    """Return lower-cased brand string from a product dict.

    Checks 'brand' field first, then falls back to scanning 'name'/'title'.
    Returns "" if neither is present.
    """
    brand = product.get("brand") or ""
    if brand:
        return brand.lower().strip()
    # Fallback: first word of name often is brand
    name = product.get("name") or product.get("title") or ""
    return name.split()[0].lower() if name else ""


def _brand_matches(product: Dict, excluded: str) -> bool:
    """Return True if the product is from the excluded brand.

    Case-insensitive. Also checks if the brand name appears anywhere in the
    product title (handles 'Recertified HP Envy' where brand field = 'Recertified').
    """
    excluded_lower = excluded.lower().strip()

    # Direct brand field match
    brand_field = (product.get("brand") or "").lower()
    if excluded_lower in brand_field or brand_field in excluded_lower:
        return True

    # Substring match in product name/title — catches 'Recertified HP Envy'
    name = (product.get("name") or product.get("title") or "").lower()
    if excluded_lower in name:
        return True

    return False


# ---------------------------------------------------------------------------
# Check handlers — each returns (score, explanation)
# ---------------------------------------------------------------------------

def _check_max_price(constraint: HardConstraint, response: Dict) -> Tuple[float, str]:
    """All recommended products must be priced ≤ constraint.value (int cents)."""
    limit_cents: int = int(constraint.value)
    products = _flatten_recs(response)

    if not products:
        return 1.0, "No recommendations — vacuously within budget"

    violations = []
    for p in products:
        cents = _price_cents(p)
        if cents is not None and cents > limit_cents:
            name = p.get("name") or p.get("title") or "unknown"
            violations.append(f"{name} (${cents / 100:.0f} > ${limit_cents / 100:.0f})")

    if violations:
        return 0.0, f"Over-budget products: {'; '.join(violations)}"
    return 1.0, f"All products within ${limit_cents / 100:.0f} budget"


def _check_excluded_brand(constraint: HardConstraint, response: Dict) -> Tuple[float, str]:
    """No recommended product may be from the excluded brand."""
    excluded: str = str(constraint.value)
    products = _flatten_recs(response)

    if not products:
        return 1.0, f"No recommendations — brand exclusion ({excluded}) vacuously satisfied"

    violations = []
    for p in products:
        if _brand_matches(p, excluded):
            name = p.get("name") or p.get("title") or "unknown"
            violations.append(name)

    if violations:
        return 0.0, f"Excluded brand {excluded!r} found in: {'; '.join(violations)}"
    return 1.0, f"No {excluded!r} products in recommendations"


def _check_min_ram(constraint: HardConstraint, response: Dict) -> Tuple[float, str]:
    """All recommended products must have RAM ≥ constraint.value GB.

    Checks the 'ram_gb' / 'specs.ram_gb' fields if present.
    If the field is absent for a product, that product is skipped (insufficient data).
    """
    min_gb: int = int(constraint.value)
    products = _flatten_recs(response)

    violations = []
    checked = 0
    for p in products:
        ram = p.get("ram_gb") or (p.get("specs") or {}).get("ram_gb")
        if ram is None:
            continue  # can't verify, skip
        checked += 1
        try:
            if float(ram) < min_gb:
                name = p.get("name") or p.get("title") or "unknown"
                violations.append(f"{name} ({ram}GB < {min_gb}GB)")
        except (TypeError, ValueError):
            pass

    if violations:
        return 0.0, f"Insufficient RAM: {'; '.join(violations)}"
    if checked == 0:
        return 1.0, f"RAM data unavailable — constraint unverifiable (pass by default)"
    return 1.0, f"All verified products have ≥{min_gb}GB RAM"


def _check_min_storage(constraint: HardConstraint, response: Dict) -> Tuple[float, str]:
    """All recommended products must have storage ≥ constraint.value GB."""
    min_gb: int = int(constraint.value)
    products = _flatten_recs(response)

    violations = []
    checked = 0
    for p in products:
        storage = p.get("storage_gb") or (p.get("specs") or {}).get("storage_gb")
        if storage is None:
            continue
        checked += 1
        try:
            if float(storage) < min_gb:
                name = p.get("name") or p.get("title") or "unknown"
                violations.append(f"{name} ({storage}GB < {min_gb}GB)")
        except (TypeError, ValueError):
            pass

    if violations:
        return 0.0, f"Insufficient storage: {'; '.join(violations)}"
    if checked == 0:
        return 1.0, "Storage data unavailable — constraint unverifiable (pass by default)"
    return 1.0, f"All verified products have ≥{min_gb}GB storage"


def _check_response_type(constraint: HardConstraint, response: Dict) -> Tuple[float, str]:
    """IDSS response_type field must equal constraint.value.

    Comparison is case-insensitive.
    "recommendations" (IDSS plural form) is treated as equivalent to "recommendation"
    because IDSS always emits the plural form in ChatResponse but human-written task
    specs commonly write the singular.  _RESPONSE_TYPE_ALIASES normalises both sides
    before comparison so either spelling works in HardConstraint.value.
    """
    # Lower-case both sides so "Question" == "question" etc.
    expected_raw: str = str(constraint.value).lower().strip()
    actual_raw: str = str(response.get("response_type") or "").lower().strip()

    # Apply plural/singular aliases to both sides
    expected_norm = _RESPONSE_TYPE_ALIASES.get(expected_raw, expected_raw)
    actual_norm   = _RESPONSE_TYPE_ALIASES.get(actual_raw,   actual_raw)

    if actual_norm == expected_norm:
        # Include the raw value in the note so callers can see what IDSS actually returned
        return 1.0, (
            f"response_type is {actual_raw!r}"
            + (f" → normalized {actual_norm!r}" if actual_raw != actual_norm else "")
            + f" (expected {expected_raw!r})"
        )
    return 0.0, (
        f"response_type is {actual_raw!r}"
        + (f" (normalized: {actual_norm!r})" if actual_raw != actual_norm else "")
        + f", expected {expected_raw!r}"
        + (f" (normalized: {expected_norm!r})" if expected_raw != expected_norm else "")
    )


def _check_cart_action(constraint: HardConstraint, response: Dict) -> Tuple[float, str]:
    """Response must contain a cart action (cart_items present with action='add')."""
    # Check top-level cart_items list
    cart_items = response.get("cart_items") or []
    for item in cart_items:
        if isinstance(item, dict) and item.get("action") in ("add", "added"):
            return 1.0, f"Cart action present: {item.get('name') or item.get('product_id')}"

    # Also check response_type = "cart" as a secondary signal
    if response.get("response_type") == "cart":
        return 1.0, "response_type='cart' indicates cart action"

    # Check if the assistant message text confirms an add
    message = (response.get("message") or response.get("response") or "").lower()
    cart_phrases = [
        "added to your cart", "added to cart", "i've added", "i have added",
        "successfully added", "placed in your cart",
    ]
    for phrase in cart_phrases:
        if phrase in message:
            return 1.0, f"Cart confirmation phrase found: '{phrase}'"

    return 0.0, "No cart action found in response"


# ---------------------------------------------------------------------------
# Aggregate evaluator — run all constraints for a task's success_criteria
# ---------------------------------------------------------------------------

def evaluate_all_constraints(
    constraints: List[HardConstraint],
    response: Dict,
) -> Tuple[float, Dict[str, Tuple[float, str]]]:
    """Run evaluate_response for each constraint.

    Returns:
      (average_score, {check_type_key: (score, explanation), ...})

    When multiple constraints share a check_type (e.g. two excluded_brand constraints),
    keys are disambiguated as "excluded_brand_0", "excluded_brand_1", etc.
    """
    results: Dict[str, Tuple[float, str]] = {}
    scores: List[float] = []

    type_count: Dict[str, int] = {}
    for c in constraints:
        count = type_count.get(c.check_type, 0)
        key = c.check_type if count == 0 else f"{c.check_type}_{count}"
        type_count[c.check_type] = count + 1

        score, explanation = evaluate_response(c, response)
        results[key] = (score, explanation)
        scores.append(score)

    avg = sum(scores) / len(scores) if scores else 1.0
    return avg, results
