"""
tests/test_runner.py — Unit tests for shopping_bench/runner.py.
All IDSS calls and simulator/judge LLM calls are mocked.
No network, no real LLM.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List

import httpx

from shopping_bench.tasks import HardConstraint, ShoppingTask, SoftGoal
from shopping_bench.runner import (
    run_task,
    _call_idss,
    _extract_products,
    _all_constraints_met,
    CONSTRAINT_WEIGHT,
    JUDGE_WEIGHT,
    PASS_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def budget_task():
    """Simple $800 budget task — 2 turns max for speed."""
    return ShoppingTask(
        id="test_budget",
        category="budget",
        description="Test budget task.",
        persona="A student who needs a laptop under $800.",
        initial_message="I need a laptop for college, budget $800.",
        success_criteria=[HardConstraint("max_price_cents", 80_000)],
        max_turns=2,
    )


@pytest.fixture
def brand_task():
    """HP exclusion task."""
    return ShoppingTask(
        id="test_brand",
        category="brand_exclusion",
        description="No HP task.",
        persona="Someone who dislikes HP.",
        initial_message="I need a laptop, no HP please.",
        success_criteria=[HardConstraint("excluded_brand", "HP")],
        max_turns=2,
    )


@pytest.fixture
def multi_constraint_task():
    """Budget + brand exclusion — two constraints."""
    return ShoppingTask(
        id="test_multi",
        category="multi_constraint",
        description="Budget and brand exclusion.",
        persona="Picky shopper.",
        initial_message="Laptop under $900, no Dell.",
        success_criteria=[
            HardConstraint("max_price_cents", 90_000),
            HardConstraint("excluded_brand", "Dell"),
        ],
        max_turns=2,
    )


@pytest.fixture
def interview_task():
    """Interview task — first response should be a question."""
    return ShoppingTask(
        id="test_interview",
        category="interview_elicitation",
        description="Vague query — should ask.",
        persona="New buyer.",
        initial_message="I want a good laptop.",
        success_criteria=[HardConstraint("response_type", "question")],
        max_turns=3,
        intermediate_checks=[(0, HardConstraint("response_type", "question"))],
    )


def make_idss_response(
    response_type: str = "recommendation",
    products: List[Dict] = None,
    message: str = "Here are some recommendations.",
    cart_items: List[Dict] = None,
) -> Dict:
    """Build a minimal IDSS /chat response dict."""
    return {
        "response_type": response_type,
        "message": message,
        "recommendations": products or [
            {"brand": "Dell", "name": "Dell Inspiron 15", "price_value": 699.99},
            {"brand": "Lenovo", "name": "Lenovo IdeaPad 5", "price_value": 749.99},
        ],
        "cart_items": cart_items or [],
    }


def make_mock_http_client(response: Dict) -> MagicMock:
    """Build a mock httpx.AsyncClient that always returns the given response."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = response

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


def make_mock_oai(judge_score: int = 8, simulator_reply: str = "That sounds great!") -> MagicMock:
    """Build a mock AsyncOpenAI that returns fixed simulator and judge responses."""
    mock_choice = MagicMock()
    mock_choice.message.content = f'{{"score": {judge_score}, "reason": "Good response"}}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    # For simulator turns, return the simulator_reply text
    sim_choice = MagicMock()
    sim_choice.message.content = simulator_reply
    sim_completion = MagicMock()
    sim_completion.choices = [sim_choice]

    # create is called for both simulator and judge — use side_effect to differentiate
    call_count = [0]

    async def fake_create(**kwargs):
        call_count[0] += 1
        # Judge call uses max_tokens=60; simulator uses max_tokens=80
        if kwargs.get("max_tokens") == 60:
            return mock_completion
        return sim_completion

    mock_oai = MagicMock()
    mock_oai.chat = MagicMock()
    mock_oai.chat.completions = MagicMock()
    mock_oai.chat.completions.create = fake_create

    return mock_oai


# ---------------------------------------------------------------------------
# _call_idss
# ---------------------------------------------------------------------------

class TestCallIdss:
    @pytest.mark.asyncio
    async def test_returns_parsed_json(self):
        response_data = make_idss_response()
        client = make_mock_http_client(response_data)
        result = await _call_idss("test message", "session-1", "http://localhost:8001", client)
        assert result == response_data

    @pytest.mark.asyncio
    async def test_http_error_returns_error_dict(self):
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.RequestError("connection refused"))
        result = await _call_idss("test", "session-1", "http://localhost:8001", mock_client)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_sends_session_id_in_payload(self):
        response_data = make_idss_response()
        client = make_mock_http_client(response_data)
        await _call_idss("hello", "my-session-123", "http://localhost:8001", client)
        # Verify post was called with session_id in the JSON body
        call_kwargs = client.post.call_args
        json_body = call_kwargs[1].get("json") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}
        # httpx.post(url, json=...) — extract from kwargs
        json_body = client.post.call_args.kwargs.get("json", {})
        assert json_body.get("session_id") == "my-session-123"

    @pytest.mark.asyncio
    async def test_sends_message_in_payload(self):
        response_data = make_idss_response()
        client = make_mock_http_client(response_data)
        await _call_idss("what laptop should I buy?", "s1", "http://localhost:8001", client)
        json_body = client.post.call_args.kwargs.get("json", {})
        assert json_body.get("message") == "what laptop should I buy?"


# ---------------------------------------------------------------------------
# _extract_products
# ---------------------------------------------------------------------------

class TestExtractProducts:
    def test_extracts_up_to_six_products(self):
        response = {
            "recommendations": [
                {"brand": f"Brand{i}", "name": f"Laptop {i}", "price_value": 100 * i}
                for i in range(10)
            ]
        }
        products = _extract_products(response)
        assert len(products) == 6

    def test_fewer_than_six_returns_all(self):
        response = {
            "recommendations": [
                {"brand": "Dell", "name": "Dell XPS", "price_value": 799}
            ]
        }
        products = _extract_products(response)
        assert len(products) == 1

    def test_extracted_fields_present(self):
        response = {
            "recommendations": [
                {"brand": "HP", "name": "HP Envy", "price_value": 699.99, "ram_gb": 16}
            ]
        }
        products = _extract_products(response)
        assert products[0]["brand"] == "HP"
        assert products[0]["name"] == "HP Envy"
        assert products[0]["price_value"] == 699.99

    def test_empty_response_returns_empty(self):
        assert _extract_products({}) == []


# ---------------------------------------------------------------------------
# _all_constraints_met
# ---------------------------------------------------------------------------

class TestAllConstraintsMet:
    def test_passing_constraints_returns_true(self):
        constraints = [HardConstraint("max_price_cents", 80_000)]
        response = {"recommendations": [{"price_value": 700}]}
        assert _all_constraints_met(constraints, response) is True

    def test_failing_constraint_returns_false(self):
        constraints = [HardConstraint("max_price_cents", 50_000)]
        response = {"recommendations": [{"price_value": 700}]}
        assert _all_constraints_met(constraints, response) is False

    def test_mixed_constraints_returns_false_if_any_fail(self):
        constraints = [
            HardConstraint("max_price_cents", 80_000),   # PASS
            HardConstraint("excluded_brand", "HP"),       # FAIL
        ]
        response = {
            "recommendations": [{"brand": "HP", "price_value": 700}]
        }
        assert _all_constraints_met(constraints, response) is False

    def test_empty_constraints_returns_true(self):
        assert _all_constraints_met([], {}) is True


# ---------------------------------------------------------------------------
# run_task — state machine tests
# ---------------------------------------------------------------------------

class TestRunTask:
    @pytest.mark.asyncio
    async def test_session_id_is_same_for_all_calls(self, budget_task):
        """All /chat calls must use the same session_id."""
        response = make_idss_response(products=[
            {"brand": "Dell", "name": "Dell X", "price_value": 700}
        ])
        client = make_mock_http_client(response)
        oai = make_mock_oai()
        sem = asyncio.Semaphore(1)

        await run_task(budget_task, "http://localhost:8001", client, oai, sem)

        call_session_ids = [
            call.kwargs.get("json", {}).get("session_id")
            for call in client.post.call_args_list
        ]
        # All session IDs should be the same (non-None) value
        assert len(set(call_session_ids)) == 1
        assert call_session_ids[0] is not None

    @pytest.mark.asyncio
    async def test_stops_at_max_turns(self, budget_task):
        """Runner should stop at max_turns even if constraints not satisfied."""
        # Response never satisfies budget (price > $800)
        bad_response = make_idss_response(products=[
            {"brand": "Apple", "name": "MacBook Pro", "price_value": 1299}
        ])
        client = make_mock_http_client(bad_response)
        oai = make_mock_oai()
        sem = asyncio.Semaphore(1)

        result = await run_task(budget_task, "http://localhost:8001", client, oai, sem)

        assert len(result.turn_results) <= budget_task.max_turns

    @pytest.mark.asyncio
    async def test_constraint_score_zero_on_budget_violation(self, budget_task):
        """Final constraint score should be 0.0 if all products exceed budget."""
        over_budget_response = make_idss_response(
            response_type="recommendation",
            products=[
                {"brand": "Apple", "name": "MacBook", "price_value": 1299},
                {"brand": "Dell", "name": "Dell XPS 15", "price_value": 950},
            ],
        )
        client = make_mock_http_client(over_budget_response)
        oai = make_mock_oai()
        sem = asyncio.Semaphore(1)

        result = await run_task(budget_task, "http://localhost:8001", client, oai, sem)
        assert result.final_constraint_score == 0.0

    @pytest.mark.asyncio
    async def test_constraint_score_one_on_clean_response(self, budget_task):
        """Final constraint score should be 1.0 when all products are within budget."""
        clean_response = make_idss_response(
            response_type="recommendation",
            products=[
                {"brand": "Dell", "name": "Dell Inspiron", "price_value": 699},
                {"brand": "Lenovo", "name": "Lenovo", "price_value": 750},
            ],
        )
        client = make_mock_http_client(clean_response)
        oai = make_mock_oai()
        sem = asyncio.Semaphore(1)

        result = await run_task(budget_task, "http://localhost:8001", client, oai, sem)
        assert result.final_constraint_score == 1.0

    @pytest.mark.asyncio
    async def test_passed_flag_true_when_score_above_threshold(self, budget_task):
        """passed=True when final_score >= PASS_THRESHOLD."""
        good_response = make_idss_response(products=[
            {"brand": "Dell", "name": "Dell X", "price_value": 700}
        ])
        client = make_mock_http_client(good_response)
        oai = make_mock_oai(judge_score=9)  # high judge score
        sem = asyncio.Semaphore(1)

        result = await run_task(budget_task, "http://localhost:8001", client, oai, sem)
        assert result.final_score >= PASS_THRESHOLD
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_passed_flag_false_when_score_below_threshold(self, budget_task):
        """passed=False when constraint fails and judge score low."""
        bad_response = make_idss_response(products=[
            {"brand": "Apple", "name": "MacBook", "price_value": 1299}
        ])
        client = make_mock_http_client(bad_response)
        oai = make_mock_oai(judge_score=0)  # low judge score
        sem = asyncio.Semaphore(1)

        result = await run_task(budget_task, "http://localhost:8001", client, oai, sem)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_excluded_brand_violation_scores_zero(self, brand_task):
        """HP in recommendations should give constraint_score=0.0 for excluded_brand=HP."""
        hp_response = make_idss_response(products=[
            {"brand": "HP", "name": "HP Envy 15", "price_value": 799},
        ])
        client = make_mock_http_client(hp_response)
        oai = make_mock_oai()
        sem = asyncio.Semaphore(1)

        result = await run_task(brand_task, "http://localhost:8001", client, oai, sem)
        assert result.final_constraint_score == 0.0

    @pytest.mark.asyncio
    async def test_intermediate_check_note_added_on_failure(self, interview_task):
        """Intermediate check failure at turn 0 should add a note."""
        # Return recommendation instead of question — violates intermediate check
        wrong_type_response = make_idss_response(response_type="recommendation")
        client = make_mock_http_client(wrong_type_response)
        oai = make_mock_oai()
        sem = asyncio.Semaphore(1)

        result = await run_task(interview_task, "http://localhost:8001", client, oai, sem)

        # Should have a note about the intermediate check failure
        intermediate_notes = [n for n in result.notes if "intermediate" in n.lower()]
        assert len(intermediate_notes) > 0

    @pytest.mark.asyncio
    async def test_task_result_has_correct_task_id(self, budget_task):
        response = make_idss_response()
        client = make_mock_http_client(response)
        oai = make_mock_oai()
        sem = asyncio.Semaphore(1)

        result = await run_task(budget_task, "http://localhost:8001", client, oai, sem)
        assert result.task_id == budget_task.id

    @pytest.mark.asyncio
    async def test_scoring_formula_applied_correctly(self, budget_task):
        """final_score must equal CONSTRAINT_WEIGHT * c_score + JUDGE_WEIGHT * j_score."""
        # Clean budget response: constraint_score = 1.0
        good_response = make_idss_response(products=[
            {"brand": "Dell", "name": "Dell X", "price_value": 700}
        ])
        client = make_mock_http_client(good_response)
        oai = make_mock_oai(judge_score=6)  # judge_score = 0.6
        sem = asyncio.Semaphore(1)

        result = await run_task(budget_task, "http://localhost:8001", client, oai, sem)

        expected = CONSTRAINT_WEIGHT * result.final_constraint_score + JUDGE_WEIGHT * result.judge_score
        assert abs(result.final_score - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_idss_error_response_stops_gracefully(self, budget_task):
        """If IDSS returns an error, runner should handle it gracefully."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.RequestError("connection refused")
        )
        oai = make_mock_oai()
        sem = asyncio.Semaphore(1)

        result = await run_task(budget_task, "http://localhost:8001", mock_client, oai, sem)
        # Should not raise — should return a result with notes about the error
        assert isinstance(result.notes, list)
        assert len(result.notes) > 0
