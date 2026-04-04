"""
tests/test_evaluator.py — Unit tests for shopping_bench/evaluator.py.
Pure logic — no DB, no LLM, no network.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from shopping_bench.tasks import HardConstraint
from shopping_bench.evaluator import (
    _flatten_recs,
    _price_cents,
    _brand_matches,
    _check_max_price,
    _check_excluded_brand,
    _check_min_ram,
    _check_min_storage,
    _check_response_type,
    _check_cart_action,
    evaluate_response,
    evaluate_all_constraints,
)


# ---------------------------------------------------------------------------
# _flatten_recs
# ---------------------------------------------------------------------------

class TestFlattenRecs:
    def test_flat_list(self):
        recs = [{"name": "A"}, {"name": "B"}]
        assert _flatten_recs({"recommendations": recs}) == recs

    def test_nested_list(self):
        recs = [[{"name": "A"}, {"name": "B"}], [{"name": "C"}]]
        result = _flatten_recs({"recommendations": recs})
        assert result == [{"name": "A"}, {"name": "B"}, {"name": "C"}]

    def test_empty_recommendations(self):
        assert _flatten_recs({"recommendations": []}) == []

    def test_none_recommendations(self):
        assert _flatten_recs({"recommendations": None}) == []

    def test_missing_recommendations_key(self):
        assert _flatten_recs({}) == []

    def test_mixed_flat_and_nested(self):
        recs = [{"name": "A"}, [{"name": "B"}, {"name": "C"}]]
        result = _flatten_recs({"recommendations": recs})
        assert len(result) == 3
        assert result[0]["name"] == "A"


# ---------------------------------------------------------------------------
# _price_cents
# ---------------------------------------------------------------------------

class TestPriceCents:
    def test_dollars_float_converted_to_cents(self):
        assert _price_cents({"price_value": 12.99}) == 1299

    def test_integer_dollars_converted(self):
        assert _price_cents({"price_value": 800}) == 80_000

    def test_exact_boundary_price(self):
        assert _price_cents({"price_value": 999.99}) == 99999

    def test_none_price_returns_none(self):
        assert _price_cents({"price_value": None}) is None

    def test_missing_price_field_returns_none(self):
        assert _price_cents({}) is None

    def test_string_price_returns_none(self):
        # Non-numeric strings should return None gracefully
        assert _price_cents({"price_value": "N/A"}) is None

    def test_zero_price(self):
        assert _price_cents({"price_value": 0.0}) == 0

    def test_rounding(self):
        # 5.005 rounds to 501 (float arithmetic edge case)
        result = _price_cents({"price_value": 5.005})
        assert result in (500, 501)  # either is acceptable


# ---------------------------------------------------------------------------
# _brand_matches
# ---------------------------------------------------------------------------

class TestBrandMatches:
    def test_exact_brand_field_match(self):
        assert _brand_matches({"brand": "HP"}, "HP") is True

    def test_case_insensitive_match(self):
        assert _brand_matches({"brand": "hp"}, "HP") is True
        assert _brand_matches({"brand": "HP"}, "hp") is True

    def test_no_match(self):
        assert _brand_matches({"brand": "Dell"}, "HP") is False

    def test_brand_in_title(self):
        # "Recertified HP Envy" — brand field may say "Recertified"
        assert _brand_matches({"brand": "Recertified", "name": "Recertified HP Envy 15"}, "HP") is True

    def test_brand_not_in_title(self):
        assert _brand_matches({"brand": "Dell", "name": "Dell Inspiron 15"}, "HP") is False

    def test_empty_brand_field_checks_name(self):
        assert _brand_matches({"brand": "", "name": "HP Spectre"}, "HP") is True

    def test_partial_brand_name_not_in_field(self):
        # "Acer" should not match "HP"
        assert _brand_matches({"brand": "Acer", "name": "Acer Aspire 5"}, "HP") is False

    def test_apple_exclusion_macbook(self):
        # MacBook has Apple in brand; exclusion of "Apple"
        assert _brand_matches({"brand": "Apple", "name": "MacBook Air M2"}, "Apple") is True


# ---------------------------------------------------------------------------
# _check_max_price
# ---------------------------------------------------------------------------

class TestCheckMaxPrice:
    CONSTRAINT = HardConstraint("max_price_cents", 80_000)

    def _response(self, *prices):
        """Build a minimal response with products at given prices (dollars)."""
        return {
            "recommendations": [
                {"name": f"Product {i}", "price_value": p}
                for i, p in enumerate(prices)
            ]
        }

    def test_all_within_budget_passes(self):
        score, note = _check_max_price(self.CONSTRAINT, self._response(699.99, 750.00))
        assert score == 1.0
        assert "within" in note.lower()

    def test_one_over_budget_fails(self):
        score, note = _check_max_price(self.CONSTRAINT, self._response(700, 850))
        assert score == 0.0
        assert "over-budget" in note.lower() or "850" in note

    def test_exact_boundary_passes(self):
        score, _ = _check_max_price(self.CONSTRAINT, self._response(800.00))
        assert score == 1.0

    def test_one_cent_over_fails(self):
        # $800.01 > $800.00
        score, _ = _check_max_price(self.CONSTRAINT, self._response(800.01))
        assert score == 0.0

    def test_no_recommendations_vacuously_passes(self):
        score, note = _check_max_price(self.CONSTRAINT, {})
        assert score == 1.0
        assert "vacuously" in note.lower()

    def test_product_without_price_ignored(self):
        # Product with no price_value should not cause a failure
        response = {"recommendations": [{"name": "Mystery Item"}]}
        score, _ = _check_max_price(self.CONSTRAINT, response)
        assert score == 1.0

    def test_multiple_violations_listed(self):
        score, note = _check_max_price(self.CONSTRAINT, self._response(900, 1000))
        assert score == 0.0
        # Both over-budget products should be mentioned
        assert "Product 0" in note or "900" in note
        assert "Product 1" in note or "1000" in note


# ---------------------------------------------------------------------------
# _check_excluded_brand
# ---------------------------------------------------------------------------

class TestCheckExcludedBrand:
    CONSTRAINT = HardConstraint("excluded_brand", "HP")

    def _response(self, *brands_and_names):
        """Build a response with [(brand, name)] pairs."""
        return {
            "recommendations": [
                {"brand": b, "name": n}
                for b, n in brands_and_names
            ]
        }

    def test_excluded_brand_absent_passes(self):
        score, note = self._constraint_check(("Dell", "Dell XPS 15"))
        assert score == 1.0

    def test_excluded_brand_present_fails(self):
        score, note = self._constraint_check(("HP", "HP Envy 15"))
        assert score == 0.0
        assert "HP" in note

    def test_case_insensitive_brand_match(self):
        score, _ = self._constraint_check(("hp", "hp laptop"))
        assert score == 0.0

    def test_excluded_brand_in_name_only_fails(self):
        # Brand field says "Recertified" but name contains "HP"
        score, note = _check_excluded_brand(
            self.CONSTRAINT,
            {"recommendations": [{"brand": "Recertified", "name": "Recertified HP Envy"}]},
        )
        assert score == 0.0

    def test_no_recommendations_vacuously_passes(self):
        score, _ = _check_excluded_brand(self.CONSTRAINT, {})
        assert score == 1.0

    def _constraint_check(self, *products):
        return _check_excluded_brand(self.CONSTRAINT, self._response(*products))


# ---------------------------------------------------------------------------
# _check_response_type
# ---------------------------------------------------------------------------

class TestCheckResponseType:
    def test_matching_type_passes(self):
        c = HardConstraint("response_type", "question")
        score, note = _check_response_type(c, {"response_type": "question"})
        assert score == 1.0

    def test_mismatching_type_fails(self):
        # "question" vs "recommendations" must fail
        c = HardConstraint("response_type", "question")
        score, note = _check_response_type(c, {"response_type": "recommendations"})
        assert score == 0.0

    def test_recommendation_singular_passes(self):
        c = HardConstraint("response_type", "recommendation")
        score, _ = _check_response_type(c, {"response_type": "recommendation"})
        assert score == 1.0

    def test_missing_response_type_fails(self):
        c = HardConstraint("response_type", "question")
        score, _ = _check_response_type(c, {})
        assert score == 0.0

    # --- Normalization tests (root cause of budget_04 and interview_04 failures) ---

    def test_recommendations_plural_matches_recommendation_singular(self):
        """IDSS always returns 'recommendations' (plural); tasks often write 'recommendation'.
        Both must be treated as equivalent — this was the root cause of v1 failures."""
        c = HardConstraint("response_type", "recommendation")
        score, note = _check_response_type(c, {"response_type": "recommendations"})
        assert score == 1.0, "plural 'recommendations' must match singular 'recommendation'"
        assert "normalized" in note  # note must explain the normalization

    def test_recommendations_plural_to_plural_matches(self):
        """Task written with plural — IDSS returns plural — must pass."""
        c = HardConstraint("response_type", "recommendations")
        score, _ = _check_response_type(c, {"response_type": "recommendations"})
        assert score == 1.0

    def test_question_does_not_match_recommendation_after_normalization(self):
        """Normalization must NOT accidentally make 'question' pass 'recommendation'."""
        c = HardConstraint("response_type", "recommendation")
        score, _ = _check_response_type(c, {"response_type": "question"})
        assert score == 0.0

    def test_comparison_is_case_insensitive(self):
        """Case must not matter: 'Question' and 'question' are the same."""
        c = HardConstraint("response_type", "Question")
        score, _ = _check_response_type(c, {"response_type": "question"})
        assert score == 1.0


# ---------------------------------------------------------------------------
# _check_cart_action
# ---------------------------------------------------------------------------

class TestCheckCartAction:
    CONSTRAINT = HardConstraint("cart_action", True)

    def test_cart_items_with_add_action_passes(self):
        response = {
            "cart_items": [{"product_id": "abc", "name": "HP Envy", "action": "add"}]
        }
        score, _ = _check_cart_action(self.CONSTRAINT, response)
        assert score == 1.0

    def test_cart_items_with_added_action_passes(self):
        response = {
            "cart_items": [{"product_id": "abc", "action": "added"}]
        }
        score, _ = _check_cart_action(self.CONSTRAINT, response)
        assert score == 1.0

    def test_cart_items_with_remove_action_fails(self):
        response = {
            "cart_items": [{"product_id": "abc", "action": "remove"}]
        }
        score, _ = _check_cart_action(self.CONSTRAINT, response)
        assert score == 0.0

    def test_empty_cart_items_fails(self):
        score, _ = _check_cart_action(self.CONSTRAINT, {"cart_items": []})
        assert score == 0.0

    def test_missing_cart_items_fails(self):
        score, _ = _check_cart_action(self.CONSTRAINT, {})
        assert score == 0.0

    def test_response_type_cart_passes(self):
        score, _ = _check_cart_action(self.CONSTRAINT, {"response_type": "cart"})
        assert score == 1.0

    def test_confirmation_phrase_in_message_passes(self):
        response = {"message": "I've added the Dell XPS to your cart."}
        score, _ = _check_cart_action(self.CONSTRAINT, response)
        assert score == 1.0

    def test_no_cart_signal_fails(self):
        response = {"response_type": "recommendation", "message": "Here are some options."}
        score, _ = _check_cart_action(self.CONSTRAINT, response)
        assert score == 0.0


# ---------------------------------------------------------------------------
# _check_min_ram
# ---------------------------------------------------------------------------

class TestCheckMinRam:
    CONSTRAINT = HardConstraint("min_ram_gb", 16)

    def test_all_meet_requirement_passes(self):
        response = {"recommendations": [
            {"name": "A", "ram_gb": 16},
            {"name": "B", "ram_gb": 32},
        ]}
        score, _ = _check_min_ram(self.CONSTRAINT, response)
        assert score == 1.0

    def test_one_below_fails(self):
        response = {"recommendations": [
            {"name": "A", "ram_gb": 8},
            {"name": "B", "ram_gb": 16},
        ]}
        score, note = _check_min_ram(self.CONSTRAINT, response)
        assert score == 0.0
        assert "8" in note

    def test_no_ram_data_passes_by_default(self):
        response = {"recommendations": [{"name": "A"}]}
        score, note = _check_min_ram(self.CONSTRAINT, response)
        assert score == 1.0
        assert "unavailable" in note.lower()

    def test_exact_boundary_passes(self):
        response = {"recommendations": [{"name": "A", "ram_gb": 16}]}
        score, _ = _check_min_ram(self.CONSTRAINT, response)
        assert score == 1.0


# ---------------------------------------------------------------------------
# _check_min_storage
# ---------------------------------------------------------------------------

class TestCheckMinStorage:
    CONSTRAINT = HardConstraint("min_storage_gb", 512)

    def test_all_meet_requirement_passes(self):
        response = {"recommendations": [
            {"name": "A", "storage_gb": 512},
            {"name": "B", "storage_gb": 1024},
        ]}
        score, _ = _check_min_storage(self.CONSTRAINT, response)
        assert score == 1.0

    def test_one_below_fails(self):
        response = {"recommendations": [{"name": "A", "storage_gb": 256}]}
        score, note = _check_min_storage(self.CONSTRAINT, response)
        assert score == 0.0
        assert "256" in note

    def test_no_storage_data_passes(self):
        response = {"recommendations": [{"name": "A"}]}
        score, note = _check_min_storage(self.CONSTRAINT, response)
        assert score == 1.0


# ---------------------------------------------------------------------------
# evaluate_response — dispatch
# ---------------------------------------------------------------------------

class TestEvaluateResponse:
    def test_dispatches_to_max_price(self):
        c = HardConstraint("max_price_cents", 80_000)
        resp = {"recommendations": [{"name": "Laptop", "price_value": 700}]}
        score, _ = evaluate_response(c, resp)
        assert score == 1.0

    def test_dispatches_to_excluded_brand(self):
        c = HardConstraint("excluded_brand", "HP")
        resp = {"recommendations": [{"brand": "HP", "name": "HP X"}]}
        score, _ = evaluate_response(c, resp)
        assert score == 0.0

    def test_dispatches_to_response_type(self):
        c = HardConstraint("response_type", "recommendation")
        score, _ = evaluate_response(c, {"response_type": "recommendation"})
        assert score == 1.0

    def test_dispatches_to_cart_action(self):
        c = HardConstraint("cart_action", True)
        score, _ = evaluate_response(c, {"response_type": "cart"})
        assert score == 1.0


# ---------------------------------------------------------------------------
# evaluate_all_constraints
# ---------------------------------------------------------------------------

class TestEvaluateAllConstraints:
    def test_single_constraint_passing(self):
        constraints = [HardConstraint("max_price_cents", 80_000)]
        resp = {"recommendations": [{"price_value": 700}]}
        avg, results = evaluate_all_constraints(constraints, resp)
        assert avg == 1.0
        assert "max_price_cents" in results
        assert results["max_price_cents"][0] == 1.0

    def test_single_constraint_failing(self):
        constraints = [HardConstraint("max_price_cents", 50_000)]
        resp = {"recommendations": [{"price_value": 700}]}
        avg, results = evaluate_all_constraints(constraints, resp)
        assert avg == 0.0

    def test_mixed_constraints_average(self):
        constraints = [
            HardConstraint("max_price_cents", 80_000),   # PASS (700 < 800)
            HardConstraint("excluded_brand", "HP"),       # FAIL (HP present)
        ]
        resp = {
            "recommendations": [{"brand": "HP", "name": "HP X", "price_value": 700}]
        }
        avg, results = evaluate_all_constraints(constraints, resp)
        assert avg == 0.5  # one pass, one fail

    def test_duplicate_check_types_get_indexed_keys(self):
        constraints = [
            HardConstraint("excluded_brand", "HP"),
            HardConstraint("excluded_brand", "Dell"),
        ]
        resp = {"recommendations": []}
        _, results = evaluate_all_constraints(constraints, resp)
        assert "excluded_brand" in results
        assert "excluded_brand_1" in results

    def test_empty_constraints_returns_1_0(self):
        avg, results = evaluate_all_constraints([], {})
        assert avg == 1.0
        assert results == {}
