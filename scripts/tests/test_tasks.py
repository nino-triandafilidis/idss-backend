"""
tests/test_tasks.py — Schema validation for IDSS-Shopping-Bench task definitions.
No DB, no LLM, no network. Pure data-integrity checks.
"""

import sys
import os

# Ensure scripts/ is on the path so we can import shopping_bench
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from shopping_bench.tasks import (
    TASKS,
    TASKS_BY_ID,
    TASKS_BY_CATEGORY,
    VALID_CHECK_TYPES,
    HardConstraint,
    SoftGoal,
    ShoppingTask,
    get_task,
    get_tasks_for_category,
)


# ---------------------------------------------------------------------------
# HardConstraint validation
# ---------------------------------------------------------------------------

class TestHardConstraint:
    def test_valid_check_type_accepted(self):
        for ct in VALID_CHECK_TYPES:
            hc = HardConstraint(check_type=ct, value=1)
            assert hc.check_type == ct

    def test_invalid_check_type_raises(self):
        with pytest.raises(ValueError, match="Unknown check_type"):
            HardConstraint(check_type="invalid_type", value=1)

    def test_max_price_cents_stores_value(self):
        hc = HardConstraint("max_price_cents", 80_000)
        assert hc.value == 80_000

    def test_excluded_brand_stores_string(self):
        hc = HardConstraint("excluded_brand", "HP")
        assert hc.value == "HP"


# ---------------------------------------------------------------------------
# Task ID uniqueness
# ---------------------------------------------------------------------------

class TestTaskIds:
    def test_task_ids_unique(self):
        ids = [t.id for t in TASKS]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_total_task_count_is_20(self):
        assert len(TASKS) == 20

    def test_tasks_by_id_has_all_tasks(self):
        assert len(TASKS_BY_ID) == len(TASKS)


# ---------------------------------------------------------------------------
# Category structure
# ---------------------------------------------------------------------------

class TestCategories:
    EXPECTED_CATEGORIES = {
        "budget",
        "brand_exclusion",
        "multi_constraint",
        "interview_elicitation",
        "cart_action",
    }

    def test_all_expected_categories_present(self):
        assert self.EXPECTED_CATEGORIES == set(TASKS_BY_CATEGORY.keys())

    def test_each_category_has_4_tasks(self):
        for cat, tasks in TASKS_BY_CATEGORY.items():
            assert len(tasks) == 4, f"Category {cat!r} has {len(tasks)} tasks (expected 4)"

    def test_task_category_field_matches_registry(self):
        for cat, tasks in TASKS_BY_CATEGORY.items():
            for t in tasks:
                assert t.category == cat


# ---------------------------------------------------------------------------
# Hard constraints — presence checks per category
# ---------------------------------------------------------------------------

class TestHardConstraintPresence:
    def test_all_tasks_have_at_least_one_hard_constraint(self):
        for t in TASKS:
            assert len(t.success_criteria) >= 1, \
                f"Task {t.id!r} has no success_criteria"

    def test_budget_tasks_have_max_price_constraint(self):
        for t in TASKS_BY_CATEGORY["budget"]:
            types = {c.check_type for c in t.success_criteria}
            # At least one price or response_type constraint (budget_04 checks response_type)
            assert types & {"max_price_cents", "response_type"}, \
                f"Budget task {t.id!r} has no price or response_type constraint"

    def test_brand_exclusion_tasks_have_excluded_brand_constraint(self):
        # All except brand_excl_03 (which tests re-inclusion) must have excluded_brand
        for t in TASKS_BY_CATEGORY["brand_exclusion"]:
            if t.id == "brand_excl_03":
                continue  # re-inclusion task checks response_type instead
            types = {c.check_type for c in t.success_criteria}
            assert "excluded_brand" in types, \
                f"Brand-exclusion task {t.id!r} has no excluded_brand constraint"

    def test_interview_tasks_have_response_type_constraint(self):
        for t in TASKS_BY_CATEGORY["interview_elicitation"]:
            types = {c.check_type for c in t.success_criteria}
            assert "response_type" in types, \
                f"Interview task {t.id!r} has no response_type constraint"

    def test_cart_tasks_have_cart_action_constraint(self):
        for t in TASKS_BY_CATEGORY["cart_action"]:
            types = {c.check_type for c in t.success_criteria}
            assert "cart_action" in types, \
                f"Cart task {t.id!r} has no cart_action constraint"

    def test_hard_constraint_check_types_are_all_valid(self):
        for t in TASKS:
            for c in t.success_criteria:
                assert c.check_type in VALID_CHECK_TYPES, \
                    f"Task {t.id!r}: invalid check_type {c.check_type!r}"

    def test_intermediate_check_types_are_all_valid(self):
        for t in TASKS:
            for turn_num, c in t.intermediate_checks:
                assert c.check_type in VALID_CHECK_TYPES, \
                    f"Task {t.id!r} intermediate check: invalid check_type {c.check_type!r}"


# ---------------------------------------------------------------------------
# Field non-empty checks
# ---------------------------------------------------------------------------

class TestFieldNonEmpty:
    def test_initial_message_nonempty(self):
        for t in TASKS:
            assert t.initial_message.strip(), f"Task {t.id!r} has empty initial_message"

    def test_persona_nonempty(self):
        for t in TASKS:
            assert t.persona.strip(), f"Task {t.id!r} has empty persona"

    def test_description_nonempty(self):
        for t in TASKS:
            assert t.description.strip(), f"Task {t.id!r} has empty description"

    def test_max_turns_positive(self):
        for t in TASKS:
            assert t.max_turns > 0, f"Task {t.id!r} has max_turns={t.max_turns}"

    def test_max_turns_reasonable_upper_bound(self):
        # Sanity: none of the 20 tasks should need more than 10 turns
        for t in TASKS:
            assert t.max_turns <= 10, \
                f"Task {t.id!r} has max_turns={t.max_turns} (seems too high)"


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

class TestLookupHelpers:
    def test_get_task_returns_correct_task(self):
        t = get_task("budget_01")
        assert t.id == "budget_01"
        assert t.category == "budget"

    def test_get_task_raises_for_unknown_id(self):
        with pytest.raises(KeyError, match="Unknown task ID"):
            get_task("nonexistent_task")

    def test_get_tasks_for_category_returns_4(self):
        tasks = get_tasks_for_category("budget")
        assert len(tasks) == 4

    def test_get_tasks_for_category_raises_for_unknown(self):
        with pytest.raises(KeyError, match="Unknown category"):
            get_tasks_for_category("nonexistent_category")


# ---------------------------------------------------------------------------
# Interview tasks — intermediate checks
# ---------------------------------------------------------------------------

class TestIntermediateChecks:
    def test_interview_tasks_have_intermediate_check_at_turn_0(self):
        # interview_01, _02, _03 must have a turn-0 response_type check
        for tid in ("interview_01", "interview_02", "interview_03"):
            t = get_task(tid)
            turn_nums = [turn for turn, _ in t.intermediate_checks]
            assert 0 in turn_nums, \
                f"Task {tid!r} missing intermediate check at turn 0"

    def test_interview_04_no_question_intermediate_check(self):
        # interview_04 is a fully-spec'd query — IDSS should NOT ask
        t = get_task("interview_04")
        rt_constraints = [
            c for _, c in t.intermediate_checks
            if c.check_type == "response_type" and c.value == "question"
        ]
        assert len(rt_constraints) == 0, \
            "interview_04 should not have an intermediate check expecting a question"
