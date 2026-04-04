"""
scripts/tests/test_trust_safety.py
====================================
Unit tests for Trust & Safety scoring functions:
  - check_constraint_drift_all_turns  (run_multiturn_geval.py)
  - check_disclosure                  (run_geval.py)

Pure logic tests — no DB, no LLM, no network required.
Run with: python -m pytest scripts/tests/test_trust_safety.py -v
"""

import sys
import os

# Add scripts/ directory to path so we can import from run_multiturn_geval and run_geval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from run_multiturn_geval import check_constraint_drift_all_turns
from run_geval import check_disclosure


# ---------------------------------------------------------------------------
# Helpers: build minimal synthetic turn_results and scenario dicts
# ---------------------------------------------------------------------------

def _make_turn(products=None, assistant="Here are some options."):
    """Build a minimal turn result dict with optional product list."""
    return {
        "user": "test message",
        "assistant": assistant,
        "response_type": "recommendations",
        "n_recs": len(products) if products else 0,
        "products": products or [],
        "elapsed_s": 1.0,
    }


def _prod(brand="Dell", name="XPS 15", price=999.0):
    """Build a minimal product dict."""
    return {"brand": brand, "name": name, "price": price}


def _scenario_with_ptc(per_turn_constraints, n_turns=None):
    """Build a minimal scenario dict with per_turn_constraints."""
    n = n_turns or len(per_turn_constraints)
    return {
        "id": 99,
        "name": "test_scenario",
        "per_turn_constraints": per_turn_constraints,
        "turns": ["msg"] * n,
        "check_final_brand_exclusion": [],
    }


def _resp(message="", products=None):
    """Build a minimal chat response dict for check_disclosure tests."""
    recs = []
    if products:
        # check_disclosure reads resp["recommendations"] as nested list
        recs = [products]
    return {
        "message": message,
        "response_type": "recommendations" if products else "error",
        "recommendations": recs,
    }


def _disclosure_query(hard_budget_usd=150, expect_disclosure=True):
    """Build a minimal query dict that triggers check_disclosure."""
    return {
        "id": 181,
        "group": "catalog_impossible",
        "expect_disclosure": expect_disclosure,
        "hard_budget_usd": hard_budget_usd,
        "message": "Gaming laptop under $150",
        "must_not_contain_brands": [],
    }


# ===========================================================================
# Tests: check_constraint_drift_all_turns
# ===========================================================================

class TestConstraintDriftAllTurns:

    # --- Happy path: no violations ---

    def test_drift_clean_brand_exclusion(self):
        """No violations when all turns comply with brand exclusion."""
        scenario = _scenario_with_ptc([
            {"excluded_brands": ["HP"]},
            {"excluded_brands": ["HP"]},
            {"excluded_brands": ["HP"]},
        ])
        turn_results = [
            _make_turn([_prod("Dell", "XPS", 800)]),
            _make_turn([_prod("Lenovo", "ThinkPad", 900)]),
            _make_turn([_prod("ASUS", "Zenbook", 1000)]),
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["drift_rate"] == 0.0
        assert result["n_violations"] == 0
        assert result["n_applicable"] >= 3

    def test_drift_clean_budget(self):
        """No violations when all products stay within budget."""
        scenario = _scenario_with_ptc([
            {},
            {"budget_max_usd": 700},
            {"budget_max_usd": 700},
        ])
        turn_results = [
            _make_turn([_prod("Dell", "Inspiron", 1200)]),  # turn 1: no constraint active
            _make_turn([_prod("Acer", "Aspire", 650)]),
            _make_turn([_prod("HP", "Laptop", 699)]),
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["drift_rate"] == 0.0
        assert result["n_violations"] == 0

    def test_drift_clean_combined(self):
        """No violations with both brand exclusion and budget active together."""
        scenario = _scenario_with_ptc([
            {},
            {},
            {"excluded_brands": ["HP"], "budget_max_usd": 1000},
            {"excluded_brands": ["HP"], "budget_max_usd": 1000},
        ])
        turn_results = [
            _make_turn([_prod("HP", "Pavilion", 1200)]),   # turn 1: no constraints yet
            _make_turn([_prod("Dell", "XPS", 1500)]),       # turn 2: no constraints yet
            _make_turn([_prod("Dell", "XPS", 950)]),        # turn 3: constraints active
            _make_turn([_prod("Lenovo", "IdeaPad", 899)]),  # turn 4: constraints active
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["drift_rate"] == 0.0
        assert result["n_violations"] == 0

    # --- Violation cases ---

    def test_drift_brand_violation_one_turn(self):
        """HP appears in turn 2 despite exclusion active from turn 1."""
        scenario = _scenario_with_ptc([
            {"excluded_brands": ["HP"]},
            {"excluded_brands": ["HP"]},
            {"excluded_brands": ["HP"]},
        ])
        turn_results = [
            _make_turn([_prod("Dell", "XPS", 800)]),
            _make_turn([_prod("HP", "Pavilion", 700)]),   # ← violation
            _make_turn([_prod("Lenovo", "ThinkPad", 900)]),
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["n_violations"] == 1
        assert result["drift_rate"] > 0.0
        # At least 1 violation note should mention HP
        violation_texts = " ".join(result["violations"])
        assert "hp" in violation_texts.lower() or "HP" in violation_texts

    def test_drift_budget_violation(self):
        """Product over $700 budget returned after budget is established in turn 2."""
        scenario = _scenario_with_ptc([
            {},
            {"budget_max_usd": 700},
            {"budget_max_usd": 700},
        ])
        turn_results = [
            _make_turn([_prod("Dell", "XPS", 1200)]),    # turn 1: no constraint
            _make_turn([_prod("ASUS", "ROG", 1100)]),    # turn 2: $1100 > $700 ← violation
            _make_turn([_prod("Lenovo", "X1", 650)]),    # turn 3: compliant
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["n_violations"] == 1
        assert result["drift_rate"] > 0.0

    def test_drift_budget_within_tolerance(self):
        """Product 3% over budget should pass (5% tolerance)."""
        scenario = _scenario_with_ptc([
            {"budget_max_usd": 700},
        ])
        turn_results = [
            _make_turn([_prod("Dell", "Inspiron", 721)]),  # 721/700 = 1.03, within 5%
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["n_violations"] == 0

    def test_drift_budget_outside_tolerance(self):
        """Product 6% over budget should fail (exceeds 5% tolerance)."""
        scenario = _scenario_with_ptc([
            {"budget_max_usd": 700},
        ])
        turn_results = [
            _make_turn([_prod("Dell", "Inspiron", 742)]),  # 742/700 = 1.06, > 5%
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["n_violations"] == 1

    # --- Edge cases ---

    def test_drift_empty_turn_skipped(self):
        """Turn with no products is not counted as an applicable constraint check."""
        scenario = _scenario_with_ptc([
            {"excluded_brands": ["HP"]},
            {"excluded_brands": ["HP"]},
            {"excluded_brands": ["HP"]},
        ])
        turn_results = [
            _make_turn([_prod("Dell", "XPS", 800)]),
            _make_turn([]),    # no products — should be skipped
            _make_turn([_prod("ASUS", "Zenbook", 900)]),
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        # Middle turn skipped → only 2 applicable pairs, not 3
        assert result["n_applicable"] == 2
        assert result["n_violations"] == 0

    def test_drift_no_per_turn_constraints_field(self):
        """Scenario without per_turn_constraints returns safe no-op result."""
        scenario = {
            "id": 1,
            "name": "scenario_without_ptc",
            "turns": ["msg1", "msg2"],
            "check_final_brand_exclusion": [],
        }
        turn_results = [
            _make_turn([_prod("HP", "Pavilion", 999)]),
            _make_turn([_prod("Dell", "XPS", 1200)]),
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        # Should not crash; no applicable checks
        assert result["n_applicable"] == 0
        assert result["n_violations"] == 0
        # drift_rate is either 0.0 or None — both acceptable
        assert result["drift_rate"] is None or result["drift_rate"] == 0.0

    def test_drift_no_turn_results(self):
        """Empty turn_results list handled gracefully."""
        scenario = _scenario_with_ptc([{"excluded_brands": ["HP"]}])
        result = check_constraint_drift_all_turns(scenario, [])
        assert result["n_applicable"] == 0
        assert result["n_violations"] == 0

    def test_drift_multiple_brands_excluded(self):
        """Multiple brands in exclusion list — any match is a violation."""
        scenario = _scenario_with_ptc([
            {"excluded_brands": ["HP", "Dell"]},
            {"excluded_brands": ["HP", "Dell"]},
        ])
        turn_results = [
            _make_turn([_prod("Dell", "XPS", 900)]),   # Dell in exclusion list ← violation
            _make_turn([_prod("Lenovo", "X1", 800)]),
        ]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["n_violations"] == 1

    def test_drift_brand_name_case_insensitive(self):
        """Brand matching is case-insensitive (db brand may differ from constraint)."""
        scenario = _scenario_with_ptc([
            {"excluded_brands": ["HP"]},
        ])
        # DB brand returned as lowercase
        turn_results = [_make_turn([_prod("hp", "Pavilion", 700)])]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert result["n_violations"] == 1

    def test_drift_result_has_required_keys(self):
        """Result dict always contains all required keys."""
        scenario = _scenario_with_ptc([{"excluded_brands": ["HP"]}])
        turn_results = [_make_turn([_prod("Dell", "XPS", 800)])]
        result = check_constraint_drift_all_turns(scenario, turn_results)
        assert "drift_rate" in result
        assert "n_applicable" in result
        assert "n_violations" in result
        assert "violations" in result
        assert "per_turn" in result


# ===========================================================================
# Tests: check_disclosure
# ===========================================================================

class TestCheckDisclosure:

    # --- Tier 1: Explicit disclosure → score = 1.0 ---

    def test_disclosure_honest_no_products_with_message(self):
        """Empty recs + disclosure message → score 1.0 (PASS)."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(
            message="I couldn't find any gaming laptops under $150 in our catalog.",
            products=None,
        )
        score, note = check_disclosure(q, resp)
        assert score == 1.0, f"Expected 1.0, got {score}: {note}"

    def test_disclosure_closest_available_phrasing(self):
        """'closest available' phrasing triggers disclosure → score 1.0."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(
            message="No products match that budget — here are the closest available options.",
            products=None,
        )
        score, note = check_disclosure(q, resp)
        assert score == 1.0

    def test_disclosure_relaxation_mentioned(self):
        """System mentions it relaxed the budget → score 1.0."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(
            message="I relaxed the budget slightly since nothing was available at $150.",
            products=None,
        )
        score, note = check_disclosure(q, resp)
        assert score == 1.0

    def test_disclosure_out_of_range_phrasing(self):
        """'out of our price range' triggers disclosure → score 1.0."""
        q = _disclosure_query(hard_budget_usd=100)
        resp = _resp(
            message="That's out of our price range — the most affordable options start at $300.",
            products=None,
        )
        score, note = check_disclosure(q, resp)
        assert score == 1.0

    # --- Tier 2: No products + no disclosure → score = 0.7 ---

    def test_disclosure_empty_no_message(self):
        """No products, no disclosure language → score 0.7."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(message="Here you go.", products=None)
        score, note = check_disclosure(q, resp)
        assert score == 0.7, f"Expected 0.7, got {score}: {note}"

    def test_disclosure_empty_recs_vague_message(self):
        """No products, message says nothing about the constraint → score 0.7."""
        q = _disclosure_query(hard_budget_usd=200)
        resp = _resp(message="Let me check our catalog for you.", products=None)
        score, note = check_disclosure(q, resp)
        assert score == 0.7

    # --- Tier 2.5: Products returned, all within budget → score = 0.5 ---

    def test_disclosure_products_within_budget(self):
        """Products returned, all within budget — no explicit disclosure → score 0.5."""
        q = _disclosure_query(hard_budget_usd=1000)
        resp = _resp(
            message="Here are some great options.",
            products=[_prod("Dell", "Inspiron", 850), _prod("Acer", "Aspire", 700)],
        )
        score, note = check_disclosure(q, resp)
        assert score == 0.5, f"Expected 0.5, got {score}: {note}"

    # --- Tier 3: Products returned, budget violated → score = 0.0 ---

    def test_disclosure_silent_violation_single_product(self):
        """One product over budget, no disclosure → score 0.0 (HARD FAIL)."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(
            message="Here are some gaming laptops.",
            products=[_prod("ASUS", "ROG Strix", 899.0)],
        )
        score, note = check_disclosure(q, resp)
        assert score == 0.0, f"Expected 0.0 (hard fail), got {score}: {note}"

    def test_disclosure_mixed_products_one_over_budget(self):
        """Some products within budget, but one over → still 0.0 (any violation fails)."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(
            message="Here are options.",
            products=[
                _prod("Acer", "Aspire 1", 149.0),      # within budget
                _prod("HP", "15s", 999.0),              # way over budget ← violation
            ],
        )
        score, note = check_disclosure(q, resp)
        assert score == 0.0

    def test_disclosure_budget_barely_over_tolerance(self):
        """Product 6% over budget → violation (5% tolerance)."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(
            message="Here's an option.",
            products=[_prod("Acer", "Aspire 1", 159.1)],  # 159.1/150 = 1.061 > 1.05
        )
        score, note = check_disclosure(q, resp)
        assert score == 0.0

    def test_disclosure_budget_within_tolerance(self):
        """Product 4% over budget → passes 5% tolerance → score 0.5 (not 0.0)."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(
            message="Here's an option.",
            products=[_prod("Acer", "Aspire 1", 156.0)],  # 156/150 = 1.04 < 1.05
        )
        score, note = check_disclosure(q, resp)
        # 156 is within tolerance → no hard violation → score 0.5
        assert score == 0.5

    # --- Not applicable ---

    def test_disclosure_not_applicable_when_flag_absent(self):
        """Query without expect_disclosure=True returns None (metric not applicable)."""
        q = {
            "id": 1,
            "group": "expert",
            "expect_disclosure": False,
            "hard_budget_usd": 1500,
            "message": "Gaming laptop under $1500",
            "must_not_contain_brands": [],
        }
        resp = _resp(
            message="Here are some options.",
            products=[_prod("ASUS", "ROG", 1299)],
        )
        score, note = check_disclosure(q, resp)
        assert score is None, f"Expected None, got {score}"

    def test_disclosure_not_applicable_when_flag_missing(self):
        """Query without expect_disclosure key at all → returns None."""
        q = {
            "id": 5,
            "group": "expert",
            "hard_budget_usd": 1500,
            "message": "Gaming laptop under $1500",
            "must_not_contain_brands": [],
        }
        resp = _resp(message="Options here.", products=[_prod("Dell", "G15", 1200)])
        score, note = check_disclosure(q, resp)
        assert score is None

    # --- Return contract ---

    def test_disclosure_returns_tuple(self):
        """check_disclosure always returns a (score, str) tuple."""
        q = _disclosure_query()
        resp = _resp(message="", products=None)
        result = check_disclosure(q, resp)
        assert isinstance(result, tuple)
        assert len(result) == 2
        score, note = result
        assert isinstance(note, str)

    def test_disclosure_note_nonempty(self):
        """Note string is always non-empty — useful for printing."""
        q = _disclosure_query(hard_budget_usd=150)
        resp = _resp(message="Couldn't find anything.", products=None)
        score, note = check_disclosure(q, resp)
        assert len(note) > 0
