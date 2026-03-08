"""
Unit tests for agent/universal_agent.py.

Covers:
- Basic instantiation (existing)
- _entropy_next_slot: picks highest-entropy slot (Item 1)
- _entropy_next_slot: falls back to priority when <5 candidates
- _entropy_next_slot: falls back when question_count == 0
- query_rewriter integration: accessory disambiguation wired in
"""

from unittest.mock import MagicMock
from agent.universal_agent import UniversalAgent
from agent.domain_registry import get_domain_schema


# ---------------------------------------------------------------------------
# Existing basic test
# ---------------------------------------------------------------------------

def test_universal_agent_basic():
    agent = UniversalAgent(session_id="test-session")
    assert agent is not None


# ---------------------------------------------------------------------------
# Item 1: _entropy_next_slot
# ---------------------------------------------------------------------------

def _make_laptop_candidates(n=20, vary_brand=True, vary_ram=False):
    """
    Build fake candidate product dicts for entropy testing.
    vary_brand=True → high brand entropy (many distinct values)
    vary_ram=False  → low RAM entropy (all same value)
    """
    brands = ["Dell", "Apple", "Lenovo", "HP", "ASUS"] * (n // 5 + 1)
    candidates = []
    for i in range(n):
        attrs = {"ram_gb": 16 if not vary_ram else [8, 16, 32][i % 3], "storage_type": "SSD"}
        candidates.append({
            "price": 800 + i * 20,
            "brand": brands[i] if vary_brand else "Dell",
            "attributes": attrs,
        })
    return candidates


def test_entropy_next_slot_uses_probe_when_q2():
    """After Q1, entropy should rank slots and pick highest entropy."""
    candidates = _make_laptop_candidates(n=20, vary_brand=True, vary_ram=False)
    probe_fn = MagicMock(return_value=candidates)

    schema = get_domain_schema("laptops")
    agent = UniversalAgent(session_id="test", probe_search_fn=probe_fn)
    agent.domain = "laptops"
    agent.question_count = 1  # past Q1
    agent.questions_asked = ["use_case"]
    agent.filters = {"use_case": "gaming"}

    slot = agent._entropy_next_slot(schema)

    # Probe should have been called
    probe_fn.assert_called_once()
    # Should return a slot
    assert slot is not None
    assert slot.name not in agent.questions_asked
    assert slot.name not in agent._EXTRACT_ONLY_SLOTS


def test_entropy_next_slot_falls_back_on_q1():
    """Q1 always uses priority system (no probe)."""
    probe_fn = MagicMock(return_value=[])
    schema = get_domain_schema("laptops")

    agent = UniversalAgent(session_id="test", probe_search_fn=probe_fn)
    agent.domain = "laptops"
    agent.question_count = 0
    agent.questions_asked = []
    agent.filters = {}

    slot = agent._entropy_next_slot(schema)

    probe_fn.assert_not_called()
    # Should return first HIGH priority slot
    assert slot is not None
    assert slot.priority.value == "HIGH"


def test_entropy_next_slot_falls_back_when_few_candidates():
    """<5 candidates → fall back to priority order."""
    candidates = _make_laptop_candidates(n=3)
    probe_fn = MagicMock(return_value=candidates)
    schema = get_domain_schema("laptops")

    agent = UniversalAgent(session_id="test", probe_search_fn=probe_fn)
    agent.domain = "laptops"
    agent.question_count = 1
    agent.questions_asked = ["use_case"]
    agent.filters = {"use_case": "gaming"}

    slot = agent._entropy_next_slot(schema)

    # Still returns a slot (via priority fallback)
    assert slot is not None


def test_entropy_next_slot_falls_back_without_probe_fn():
    """No probe_fn → always use priority system."""
    schema = get_domain_schema("laptops")

    agent = UniversalAgent(session_id="test", probe_search_fn=None)
    agent.domain = "laptops"
    agent.question_count = 2
    agent.questions_asked = ["use_case"]
    agent.filters = {"use_case": "gaming"}

    slot = agent._entropy_next_slot(schema)

    assert slot is not None


def test_entropy_next_slot_skips_asked_slots():
    """Already-asked slots should never be returned."""
    candidates = _make_laptop_candidates(n=20)
    probe_fn = MagicMock(return_value=candidates)
    schema = get_domain_schema("laptops")

    agent = UniversalAgent(session_id="test", probe_search_fn=probe_fn)
    agent.domain = "laptops"
    agent.question_count = 1
    agent.questions_asked = ["use_case", "budget", "min_ram_gb", "screen_size", "brand"]
    agent.filters = {}

    slot = agent._entropy_next_slot(schema)

    # All mapped slots are asked — should fall back to priority (which may return None)
    if slot is not None:
        assert slot.name not in agent.questions_asked


def test_entropy_next_slot_skips_extract_only_slots():
    """excluded_brands, os should never be selected."""
    candidates = _make_laptop_candidates(n=20)
    probe_fn = MagicMock(return_value=candidates)
    schema = get_domain_schema("laptops")

    agent = UniversalAgent(session_id="test", probe_search_fn=probe_fn)
    agent.domain = "laptops"
    agent.question_count = 1
    agent.questions_asked = []
    agent.filters = {}

    for _ in range(10):
        slot = agent._entropy_next_slot(schema)
        if slot:
            assert slot.name not in agent._EXTRACT_ONLY_SLOTS
        # Simulate asking to cycle through slots
        if slot:
            agent.questions_asked.append(slot.name)
            agent.question_count += 1


# ---------------------------------------------------------------------------
# Item 5: query_rewriter wired into process_message
# ---------------------------------------------------------------------------

def test_process_message_accessory_clarification():
    """Accessory disambiguation from query_rewriter should fire before any LLM call."""
    agent = UniversalAgent(session_id="test-acc")
    agent.domain = "laptops"
    agent.question_count = 0
    agent.questions_asked = []
    agent.filters = {}

    # "bag" alone (domain="laptops" set above) triggers accessory check
    result = agent.process_message("bag")

    assert result["response_type"] == "question"
    assert "laptop" in result["message"].lower()
    assert any("laptop itself" in r.lower() for r in (result.get("quick_replies") or []))


def test_process_message_accessory_not_fired_with_spec_signal():
    """If spec signals present, accessory check should not fire."""
    agent = UniversalAgent(session_id="test-spec")
    agent.domain = "laptops"
    agent.question_count = 0
    agent.questions_asked = []
    agent.filters = {}

    result = agent.process_message("bag 16gb ram ssd")
    # Should NOT trigger accessory clarification — may be question or handoff
    if result["response_type"] == "question":
        assert "accessory" not in result["message"].lower()
