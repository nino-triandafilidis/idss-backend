"""
Unit tests for agent/query_rewriter.py (Item 5).

Covers:
- Accessory disambiguation → clarifying question
- Context-aware expansion (budget, brand)
- Common-sense enrichment (family, work)
- Normal queries pass through unchanged
"""

import pytest
from agent.query_rewriter import rewrite, RewriteResult


# ---------------------------------------------------------------------------
# Accessory disambiguation
# ---------------------------------------------------------------------------

def test_accessory_laptop_bag_triggers_clarification():
    # "laptop" is a spec signal, so use bare "bag" with domain="laptops" to test accessory check
    result = rewrite("bag", session_history=[], domain="laptops", current_filters={}, question_count=0)
    assert result.is_clarification is True
    assert "laptop" in result.clarifying_question
    assert result.quick_replies and len(result.quick_replies) == 2


def test_accessory_not_triggered_when_spec_signals_present():
    # "16gb ram" is a spec signal — accessory check is suppressed
    result = rewrite("bag 16gb ram", session_history=[], domain="laptops", current_filters={}, question_count=0)
    assert result.is_clarification is False


def test_accessory_not_triggered_after_first_question():
    # question_count=1 → never ask accessory clarification again
    result = rewrite("laptop bag", session_history=[], domain="laptops", current_filters={}, question_count=1)
    assert result.is_clarification is False


def test_accessory_not_triggered_for_long_message():
    msg = "laptop bag sleeve dock charger adapter cable mouse webcam hub port hdmi usb upgrade parts peripheral"
    result = rewrite(msg, session_history=[], domain="laptops", current_filters={}, question_count=0)
    # >20 words — skip accessory check
    assert result.is_clarification is False


def test_accessory_not_triggered_when_no_domain():
    result = rewrite("bag", session_history=[], domain="", current_filters={}, question_count=0)
    assert result.is_clarification is False


# ---------------------------------------------------------------------------
# Context-aware expansion
# ---------------------------------------------------------------------------

def test_cheaper_expands_with_known_budget_cents():
    result = rewrite(
        "show me cheaper ones",
        session_history=[],
        domain="laptops",
        current_filters={"price_max_cents": 120000},  # $1200 in cents
        question_count=1,
    )
    assert result.is_clarification is False
    assert "1200" in result.rewritten or "under" in result.rewritten.lower()


def test_cheaper_expands_with_known_budget_dollars():
    result = rewrite(
        "cheaper",
        session_history=[],
        domain="laptops",
        current_filters={"budget": 800},  # already in dollars
        question_count=1,
    )
    assert "800" in result.rewritten


def test_cheaper_no_budget_passthrough():
    result = rewrite("show me cheaper ones", session_history=[], domain="laptops", current_filters={}, question_count=1)
    assert result.is_clarification is False
    # No expansion possible — should pass through unchanged
    assert "cheaper" in result.rewritten.lower()


def test_different_brand_expands_with_known_brand():
    result = rewrite(
        "show me a different brand",
        session_history=[],
        domain="laptops",
        current_filters={"brand": "Dell"},
        question_count=1,
    )
    assert "Dell" in result.rewritten


def test_different_brand_no_brand_passthrough():
    result = rewrite("different brand", session_history=[], domain="laptops", current_filters={}, question_count=1)
    assert "Dell" not in result.rewritten


# ---------------------------------------------------------------------------
# Common-sense enrichment
# ---------------------------------------------------------------------------

def test_for_son_adds_school_use_case():
    result = rewrite("I need a laptop for my son", session_history=[], domain="laptops", current_filters={}, question_count=0)
    assert "[use_case: school]" in result.rewritten


def test_for_daughter_adds_school_use_case():
    result = rewrite("this is for my daughter", session_history=[], domain="laptops", current_filters={}, question_count=0)
    assert "[use_case: school]" in result.rewritten


def test_for_kid_adds_school_use_case():
    result = rewrite("a laptop for my kid", session_history=[], domain="laptops", current_filters={}, question_count=1)
    assert "[use_case: school]" in result.rewritten


def test_for_work_adds_business_use_case():
    result = rewrite("I need a laptop for work", session_history=[], domain="laptops", current_filters={}, question_count=0)
    assert "[use_case: business]" in result.rewritten


def test_for_work_not_added_if_use_case_explicit():
    # "use_case" already in message — don't double-annotate
    result = rewrite("for work use_case gaming", session_history=[], domain="laptops", current_filters={}, question_count=0)
    assert "[use_case: business]" not in result.rewritten


# ---------------------------------------------------------------------------
# Normal queries pass through unchanged
# ---------------------------------------------------------------------------

def test_normal_query_passthrough():
    msg = "find me a gaming laptop under $900"
    result = rewrite(msg, session_history=[], domain="laptops", current_filters={}, question_count=0)
    assert result.is_clarification is False
    assert msg in result.rewritten  # original message preserved (possibly with enrichment)


def test_result_is_rewrite_result_type():
    result = rewrite("hello", session_history=[], domain="", current_filters={}, question_count=0)
    assert isinstance(result, RewriteResult)
    assert isinstance(result.rewritten, str)
