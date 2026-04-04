"""
tests/test_simulator.py — Unit tests for shopping_bench/simulator.py.
Tests prompt construction only — no LLM calls made.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from shopping_bench.tasks import HardConstraint, ShoppingTask, SoftGoal, get_task
from shopping_bench.simulator import (
    get_first_turn,
    build_simulator_prompt,
    _build_constraints_summary,
    _build_clarification_hints,
    _format_history,
    _describe_response_type,
    SIMULATOR_MODEL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_task():
    return ShoppingTask(
        id="test_01",
        category="budget",
        description="Test task for simulator tests.",
        persona="A college student looking for a laptop.",
        initial_message="I need a laptop for college, budget $800.",
        success_criteria=[HardConstraint("max_price_cents", 80_000)],
        clarification_answers={"brand": "I'm flexible on brand."},
    )


@pytest.fixture
def empty_response():
    return {}


@pytest.fixture
def recommendation_response():
    return {
        "response_type": "recommendation",
        "message": "Here are three laptops I recommend for you...",
        "recommendations": [
            {"brand": "Dell", "name": "Dell Inspiron 15", "price_value": 699.99},
        ],
    }


@pytest.fixture
def question_response():
    return {
        "response_type": "question",
        "message": "What is your budget and main use case?",
    }


# ---------------------------------------------------------------------------
# get_first_turn
# ---------------------------------------------------------------------------

class TestGetFirstTurn:
    def test_returns_initial_message(self, simple_task):
        result = get_first_turn(simple_task)
        assert result == simple_task.initial_message

    def test_returns_exact_string(self, simple_task):
        # No LLM call — should be a pure string return
        result = get_first_turn(simple_task)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_tasks_have_nonempty_initial_message(self):
        from shopping_bench.tasks import TASKS
        for task in TASKS:
            assert get_first_turn(task).strip(), f"Empty initial message for {task.id}"


# ---------------------------------------------------------------------------
# _build_constraints_summary
# ---------------------------------------------------------------------------

class TestBuildConstraintsSummary:
    def test_budget_constraint_shows_dollar_amount(self, simple_task):
        summary = _build_constraints_summary(simple_task)
        assert "$800" in summary or "800" in summary

    def test_excluded_brand_shows_brand_name(self):
        task = ShoppingTask(
            id="test_brand",
            category="brand_exclusion",
            description="Brand exclusion test",
            persona="test",
            initial_message="test",
            success_criteria=[HardConstraint("excluded_brand", "HP")],
        )
        summary = _build_constraints_summary(task)
        assert "HP" in summary

    def test_cart_action_mentions_cart(self):
        task = ShoppingTask(
            id="test_cart",
            category="cart_action",
            description="Cart test",
            persona="test",
            initial_message="test",
            success_criteria=[HardConstraint("cart_action", True)],
        )
        summary = _build_constraints_summary(task)
        assert "cart" in summary.lower()

    def test_no_constraints_returns_fallback(self):
        task = ShoppingTask(
            id="test_empty",
            category="budget",
            description="test",
            persona="test",
            initial_message="test",
            success_criteria=[HardConstraint("response_type", "recommendation")],
        )
        # response_type constraint produces no specific constraint summary
        summary = _build_constraints_summary(task)
        assert isinstance(summary, str)

    def test_multiple_constraints_all_present(self):
        task = ShoppingTask(
            id="test_multi",
            category="multi_constraint",
            description="test",
            persona="test",
            initial_message="test",
            success_criteria=[
                HardConstraint("max_price_cents", 100_000),
                HardConstraint("excluded_brand", "Lenovo"),
                HardConstraint("min_ram_gb", 16),
            ],
        )
        summary = _build_constraints_summary(task)
        assert "1000" in summary or "$1000" in summary
        assert "Lenovo" in summary
        assert "16" in summary


# ---------------------------------------------------------------------------
# _build_clarification_hints
# ---------------------------------------------------------------------------

class TestBuildClarificationHints:
    def test_hints_present_in_output(self, simple_task):
        hints = _build_clarification_hints(simple_task)
        assert "brand" in hints.lower()
        assert "flexible" in hints.lower()

    def test_no_hints_returns_fallback(self):
        task = ShoppingTask(
            id="test_no_hints",
            category="budget",
            description="test",
            persona="test",
            initial_message="test",
            success_criteria=[HardConstraint("max_price_cents", 80_000)],
        )
        hints = _build_clarification_hints(task)
        assert "flexible" in hints.lower() or "no specific hints" in hints.lower()

    def test_multiple_hints_all_included(self):
        task = ShoppingTask(
            id="test_multi_hints",
            category="interview_elicitation",
            description="test",
            persona="test",
            initial_message="test",
            success_criteria=[HardConstraint("response_type", "question")],
            clarification_answers={
                "budget": "Around $600.",
                "use": "Mostly browsing.",
                "brand": "No preference.",
            },
        )
        hints = _build_clarification_hints(task)
        assert "budget" in hints.lower()
        assert "use" in hints.lower()
        assert "brand" in hints.lower()


# ---------------------------------------------------------------------------
# _format_history
# ---------------------------------------------------------------------------

class TestFormatHistory:
    def test_empty_history_returns_no_turns_text(self):
        result = _format_history([])
        assert "no previous" in result.lower()

    def test_single_user_turn(self):
        history = [{"role": "user", "content": "I need a laptop."}]
        result = _format_history(history)
        assert "USER" in result
        assert "I need a laptop." in result

    def test_user_and_assistant_turns(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]
        result = _format_history(history)
        assert "USER" in result
        assert "ASSISTANT" in result
        assert "Hello" in result
        assert "Hi, how can I help?" in result

    def test_ordering_preserved(self):
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ]
        result = _format_history(history)
        assert result.index("First") < result.index("Second")


# ---------------------------------------------------------------------------
# _describe_response_type
# ---------------------------------------------------------------------------

class TestDescribeResponseType:
    def test_question_described_as_asking(self):
        desc = _describe_response_type({"response_type": "question"})
        assert "question" in desc.lower() or "asking" in desc.lower()

    def test_recommendation_described_as_recommending(self):
        desc = _describe_response_type({"response_type": "recommendation"})
        assert "recommend" in desc.lower()

    def test_unknown_type_returns_something(self):
        desc = _describe_response_type({"response_type": "foobar"})
        assert len(desc) > 0

    def test_empty_response_returns_something(self):
        desc = _describe_response_type({})
        assert len(desc) > 0


# ---------------------------------------------------------------------------
# build_simulator_prompt — structure tests
# ---------------------------------------------------------------------------

class TestBuildSimulatorPrompt:
    def test_returns_two_messages(self, simple_task, question_response):
        messages = build_simulator_prompt(simple_task, 1, [], question_response)
        assert len(messages) == 2

    def test_first_message_is_system(self, simple_task, question_response):
        messages = build_simulator_prompt(simple_task, 1, [], question_response)
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self, simple_task, question_response):
        messages = build_simulator_prompt(simple_task, 1, [], question_response)
        assert messages[1]["role"] == "user"

    def test_system_contains_persona(self, simple_task, question_response):
        messages = build_simulator_prompt(simple_task, 1, [], question_response)
        assert simple_task.persona in messages[0]["content"]

    def test_system_contains_constraints(self, simple_task, question_response):
        messages = build_simulator_prompt(simple_task, 1, [], question_response)
        # Budget constraint should appear in system prompt
        assert "800" in messages[0]["content"]

    def test_system_contains_clarification_hints(self, simple_task, question_response):
        messages = build_simulator_prompt(simple_task, 1, [], question_response)
        # Clarification answer should appear
        assert "flexible" in messages[0]["content"].lower()

    def test_user_prompt_contains_last_message(self, simple_task, question_response):
        messages = build_simulator_prompt(simple_task, 1, [], question_response)
        assert question_response["message"] in messages[1]["content"]

    def test_user_prompt_contains_turn_number(self, simple_task, question_response):
        messages = build_simulator_prompt(simple_task, 3, [], question_response)
        assert "3" in messages[1]["content"]

    def test_user_prompt_contains_history(self, simple_task, question_response):
        history = [
            {"role": "user", "content": "Previous user message"},
            {"role": "assistant", "content": "Previous assistant message"},
        ]
        messages = build_simulator_prompt(simple_task, 2, history, question_response)
        assert "Previous user message" in messages[1]["content"]

    def test_long_last_message_truncated(self, simple_task):
        long_response = {
            "response_type": "recommendation",
            "message": "x" * 1200,  # deliberately long
        }
        messages = build_simulator_prompt(simple_task, 1, [], long_response)
        # The user prompt should NOT contain the full 1200-char string
        assert "x" * 1200 not in messages[1]["content"]
        # But should contain the truncated version
        assert "x" * 50 in messages[1]["content"]  # first 50 chars still present

    def test_simulator_model_is_gpt4o_mini(self):
        assert SIMULATOR_MODEL == "gpt-4o-mini"
