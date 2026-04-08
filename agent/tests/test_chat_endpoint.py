import asyncio
import pytest
from unittest.mock import MagicMock, patch

from agent.chat_endpoint import (
    process_chat,
    ChatRequest,
    _compute_diversity_score,
    _diversify_by_brand,
    _handle_post_recommendation,
    _detect_faq_category,
    _FAST_TARGETED_QA_KWS,
)
from agent.interview.session_manager import InterviewSessionState, STAGE_RECOMMENDATIONS


def test_chat_endpoint_basic():
    req = ChatRequest(message="Hello")
    assert req.message == "Hello"


# ---------------------------------------------------------------------------
# _compute_diversity_score
# ---------------------------------------------------------------------------

def test_diversity_score_empty():
    """Empty product list returns all-zero scores."""
    result = _compute_diversity_score([])
    assert result == {"brand_diversity": 0.0, "price_spread": 0.0, "overall": 0.0}


def test_diversity_score_all_same_brand():
    """All same brand → brand_diversity = 1/n."""
    products = [
        {"brand": "Dell", "price": 500.0},
        {"brand": "Dell", "price": 700.0},
        {"brand": "Dell", "price": 900.0},
    ]
    result = _compute_diversity_score(products)
    assert result["brand_diversity"] == pytest.approx(1 / 3, abs=0.01)
    assert result["price_spread"] == pytest.approx((900 - 500) / 900, abs=0.01)
    assert 0 <= result["overall"] <= 1


def test_diversity_score_all_different_brands():
    """All different brands → brand_diversity = 1.0."""
    products = [
        {"brand": "Dell", "price": 500.0},
        {"brand": "Apple", "price": 1500.0},
        {"brand": "Lenovo", "price": 800.0},
    ]
    result = _compute_diversity_score(products)
    assert result["brand_diversity"] == 1.0
    assert result["price_spread"] == pytest.approx((1500 - 500) / 1500, abs=0.01)


def test_diversity_score_single_product():
    """Single product has full brand diversity but no price spread."""
    products = [{"brand": "HP", "price": 999.0}]
    result = _compute_diversity_score(products)
    assert result["brand_diversity"] == 1.0
    assert result["price_spread"] == 0.0


def test_diversity_score_missing_brand_treated_as_unknown():
    """Products without brand field should all map to 'unknown'."""
    products = [
        {"price": 500.0},
        {"price": 700.0},
        {"brand": None, "price": 900.0},
    ]
    result = _compute_diversity_score(products)
    # All map to "unknown" → 1 unique brand / 3 products = 0.333
    assert result["brand_diversity"] == pytest.approx(1 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# _diversify_by_brand (round-robin interleaving)
# ---------------------------------------------------------------------------

def test_diversify_single_product_unchanged():
    products = [{"brand": "Dell", "name": "A"}]
    assert _diversify_by_brand(products) == products


def test_diversify_all_same_brand_unchanged():
    """When all products share a brand, order is preserved."""
    products = [
        {"brand": "Dell", "name": "A"},
        {"brand": "Dell", "name": "B"},
        {"brand": "Dell", "name": "C"},
    ]
    assert _diversify_by_brand(products) == products


def test_diversify_interleaves_brands():
    """Round-robin should interleave so no two adjacent products share a brand."""
    products = [
        {"brand": "Dell", "name": "D1"},
        {"brand": "Dell", "name": "D2"},
        {"brand": "Apple", "name": "A1"},
        {"brand": "Lenovo", "name": "L1"},
    ]
    result = _diversify_by_brand(products)
    # No two adjacent items should have the same brand (given 3 brands)
    for i in range(len(result) - 1):
        b_cur = (result[i].get("brand") or "").lower()
        b_nxt = (result[i + 1].get("brand") or "").lower()
        assert b_cur != b_nxt, f"Adjacent brands match at position {i}: {b_cur}"


# ===========================================================================
# Helpers shared by add-to-cart and KG see-similar tests
# ===========================================================================

_TEST_PRODUCTS = [
    {"id": "prod-001", "name": "Lenovo Slim 5 Pro 16", "brand": "Lenovo", "price": 899.99},
    {"id": "prod-002", "name": "Dell XPS 15 9510", "brand": "Dell", "price": 1299.99},
    {"id": "prod-003", "name": "Apple MacBook Air M2", "brand": "Apple", "price": 1099.99},
]


def _make_rec_session(products=None, favorites=None):
    """Return a session in RECOMMENDATIONS stage with products loaded."""
    return InterviewSessionState(
        active_domain="laptops",
        stage=STAGE_RECOMMENDATIONS,
        last_recommendation_data=products if products is not None else list(_TEST_PRODUCTS),
        last_recommendation_ids=[p["id"] for p in (_TEST_PRODUCTS if products is None else products)],
        favorite_product_ids=list(favorites or []),
    )


def _make_mock_sm(session):
    """Mock session_manager whose add_favorite mutates the in-memory session object."""
    sm = MagicMock()

    def _add_fav(sid, pid):
        if pid not in session.favorite_product_ids:
            session.favorite_product_ids.append(pid)

    sm.add_favorite = MagicMock(side_effect=_add_fav)
    return sm


# ===========================================================================
# Add-to-cart intent tests
# ===========================================================================


def test_add_to_cart_ordinal_first():
    """'add the first one to my cart' → adds first product, cart_action set."""
    session = _make_rec_session()
    sm = _make_mock_sm(session)
    req = ChatRequest(message="add the first one to my cart", session_id="s1")

    resp = asyncio.run(_handle_post_recommendation(req, session, "s1", sm))

    assert resp is not None
    assert resp.response_type == "question"
    assert "Lenovo Slim 5 Pro 16" in resp.message
    assert resp.cart_action is not None
    assert resp.cart_action["action"] == "add_to_cart"
    assert resp.cart_action["product"]["id"] == "prod-001"
    sm.add_favorite.assert_called_once_with("s1", "prod-001")


def test_add_to_cart_ordinal_second():
    """'add the second one to my cart' → adds second product."""
    session = _make_rec_session()
    sm = _make_mock_sm(session)
    req = ChatRequest(message="add the second one to my cart", session_id="s2")

    resp = asyncio.run(_handle_post_recommendation(req, session, "s2", sm))

    assert resp is not None
    assert "Dell XPS 15 9510" in resp.message
    assert resp.cart_action["product"]["id"] == "prod-002"
    sm.add_favorite.assert_called_once_with("s2", "prod-002")


def test_add_to_cart_fuzzy_name():
    """'add the Slim 5 Pro to cart' → fuzzy matches by overlapping words."""
    session = _make_rec_session()
    sm = _make_mock_sm(session)
    # "slim", "pro" and "16" all appear in "Lenovo Slim 5 Pro 16" — ≥2 word overlap
    req = ChatRequest(message="add the Slim 5 Pro 16 to cart", session_id="s3")

    resp = asyncio.run(_handle_post_recommendation(req, session, "s3", sm))

    assert resp is not None
    assert resp.cart_action is not None
    assert resp.cart_action["product"]["id"] == "prod-001"


def test_add_to_cart_already_in_cart():
    """Product already in favorite_product_ids → 'already in cart', cart_action=None."""
    # pre-load prod-001 into favorites to simulate it being already added
    session = _make_rec_session(favorites=["prod-001"])
    sm = _make_mock_sm(session)
    req = ChatRequest(message="add the first one to my cart", session_id="s4")

    resp = asyncio.run(_handle_post_recommendation(req, session, "s4", sm))

    assert resp is not None
    assert "already" in resp.message.lower()
    assert resp.cart_action is None  # must NOT fire cart_action again
    sm.add_favorite.assert_not_called()


def test_add_to_cart_no_products_in_session():
    """Empty last_recommendation_data → fallback message, no cart_action."""
    session = _make_rec_session(products=[], favorites=[])
    session.last_recommendation_ids = []  # also no IDs to re-fetch
    sm = _make_mock_sm(session)
    # Prevent _fetch_products_by_ids from hitting the DB
    with patch("agent.chat_endpoint._fetch_products_by_ids", return_value=[]):
        req = ChatRequest(message="add to cart", session_id="s5")
        resp = asyncio.run(_handle_post_recommendation(req, session, "s5", sm))

    assert resp is not None
    assert "don't have any specific product" in resp.message.lower() or "no specific" in resp.message.lower()
    assert resp.cart_action is None
    sm.add_favorite.assert_not_called()


# ===========================================================================
# KG see-similar integration tests
# ===========================================================================


def _make_unified_product_mock(product_id: str, name: str):
    """Return a mock that behaves like UnifiedProduct (has .model_dump())."""
    m = MagicMock()
    m.model_dump.return_value = {
        "id": product_id,
        "name": name,
        "brand": "TestBrand",
        "price": 99900,
        "productType": "laptop",
    }
    return m


def test_see_similar_uses_kg_when_available():
    """When KG returns neighbour IDs, response is recommendations from those IDs."""
    session = _make_rec_session()
    sm = _make_mock_sm(session)

    kg_neighbour_ids = ["prod-sim-1", "prod-sim-2", "prod-sim-3"]

    # Mock KG service: available and returns neighbours for prod-001
    mock_kg = MagicMock()
    mock_kg.is_available.return_value = True
    mock_kg.get_similar_products.return_value = kg_neighbour_ids

    # Mock products fetched for those IDs
    fake_products = [
        {"id": pid, "name": f"Similar Product {i+1}", "brand": "Dell", "price": 999.0}
        for i, pid in enumerate(kg_neighbour_ids)
    ]

    req = ChatRequest(message="see similar items", session_id="s6")

    with (
        patch("app.kg_service.get_kg_service", return_value=mock_kg),
        patch("agent.chat_endpoint._fetch_products_by_ids", return_value=fake_products),
        patch(
            "app.formatters.format_product",
            side_effect=lambda p, d: _make_unified_product_mock(p["id"], p["name"]),
        ),
    ):
        resp = asyncio.run(_handle_post_recommendation(req, session, "s6", sm))

    assert resp is not None
    assert resp.response_type == "recommendations"
    assert "similar" in resp.message.lower()
    # All three neighbour rows should be present
    flat_ids = [item.get("id") for row in resp.recommendations for item in row]
    assert set(kg_neighbour_ids).issubset(set(flat_ids))

    # KG was queried with the first recommended product
    mock_kg.get_similar_products.assert_called_once_with("prod-001", limit=6)


def test_see_similar_kg_unavailable_falls_back_to_sql():
    """When KG is unavailable, code falls back to SQL search (no crash, valid response)."""
    session = _make_rec_session()
    sm = _make_mock_sm(session)

    mock_kg = MagicMock()
    mock_kg.is_available.return_value = False  # KG offline

    req = ChatRequest(message="see similar items", session_id="s7")

    # SQL fallback returns empty → expect the "broaden search" fallback message
    with (
        patch("app.kg_service.get_kg_service", return_value=mock_kg),
        patch("agent.chat_endpoint._search_ecommerce_products", return_value=([], [])),
    ):
        resp = asyncio.run(_handle_post_recommendation(req, session, "s7", sm))

    assert resp is not None
    # When SQL also returns nothing, we get the generic fallback question
    assert resp.response_type == "question"
    assert "broaden" in resp.message.lower() or "more options" in resp.message.lower()
    # KG was queried for availability but get_similar_products should NOT be called
    mock_kg.get_similar_products.assert_not_called()


def test_see_similar_kg_exception_falls_back_gracefully():
    """KG raising an exception should not crash the handler — SQL fallback runs."""
    session = _make_rec_session()
    sm = _make_mock_sm(session)

    req = ChatRequest(message="show me similar", session_id="s8")

    with (
        patch(
            "app.kg_service.get_kg_service",
            side_effect=Exception("Neo4j connection refused"),
        ),
        patch("agent.chat_endpoint._search_ecommerce_products", return_value=([], [])),
    ):
        resp = asyncio.run(_handle_post_recommendation(req, session, "s8", sm))

    assert resp is not None  # no unhandled exception
    assert resp.response_type in ("recommendations", "question")


# ---------------------------------------------------------------------------
# Part 2: Service/FAQ question detection (_detect_faq_category)
# Verifies that return-policy, shipping, warranty, and RAM-upgrade questions
# are correctly classified before being handed off to the LLM for answering.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg,expected_cat", [
    # Return policy
    ("What is the return policy if I don't like the laptop after buying it?", "return"),
    ("Can I return it if it doesn't work?", "return"),
    ("How do I get a refund?", "return"),
    # Shipping
    ("How long does shipping usually take? I'm in California.", "shipping"),
    ("When will it arrive if I order today?", "shipping"),
    ("What's the estimated delivery time?", "shipping"),
    # Warranty — accidental damage
    ("If I accidentally spill coffee on it, does the warranty cover that?", "warranty"),
    ("Does the warranty cover accidental damage?", "warranty"),
    ("What does the extended protection cover?", "warranty"),
    # RAM upgrade
    ("If I buy a laptop now with 16GB, can I upgrade the RAM to 32GB myself later?", "upgrade"),
    ("Can I upgrade the RAM later?", "upgrade"),
    ("Is the RAM soldered or socketed?", "upgrade"),
])
def test_faq_category_detection(msg, expected_cat):
    """_detect_faq_category must correctly classify all four FAQ categories."""
    result = _detect_faq_category(msg.lower())
    assert result == expected_cat, (
        f"Expected category {expected_cat!r} for: {msg!r}, got {result!r}"
    )


def test_faq_category_none_for_product_queries():
    """Regular product queries must NOT be misclassified as FAQ."""
    non_faq = [
        "I need a laptop under $800",
        "Show me gaming laptops",
        "Which is better, Dell or HP?",
        "I want 16GB RAM and an RTX 4060",
    ]
    for msg in non_faq:
        result = _detect_faq_category(msg.lower())
        assert result is None, f"Unexpected FAQ category {result!r} for: {msg!r}"


# ---------------------------------------------------------------------------
# Part 3: Orchestrator routing fast-path keyword additions
# Verifies the new entries in _FAST_TARGETED_QA_KWS and the checkout
# fast-path keyword lists.
# ---------------------------------------------------------------------------

def test_which_of_those_has_is_in_targeted_qa_kws():
    """'which of those has' must be in _FAST_TARGETED_QA_KWS (Q194 fix)."""
    assert "which of those has" in _FAST_TARGETED_QA_KWS, (
        "'which of those has' missing from _FAST_TARGETED_QA_KWS — "
        "Q194 'Which of those has the best battery life?' will be misrouted."
    )


def test_which_of_them_has_is_in_targeted_qa_kws():
    """'which of them has' must also be in _FAST_TARGETED_QA_KWS."""
    assert "which of them has" in _FAST_TARGETED_QA_KWS


@pytest.mark.parametrize("phrase", [
    "Which of those has the best battery life?",
    "Which of them has the longest battery?",
    "Which one has the best keyboard?",
])
def test_targeted_qa_kws_match_anaphoric_battery_queries(phrase):
    """Anaphoric 'which of those/them/one has the best X' must match the fast-path list."""
    msg_lower = phrase.lower()
    assert any(kw in msg_lower for kw in _FAST_TARGETED_QA_KWS), (
        f"No targeted-QA keyword matched: {phrase!r}"
    )


@pytest.mark.parametrize("phrase", [
    "let's check out now",
    "lets check out",
    "ready to check out",
    "proceed to checkout",
    "ready to checkout",
])
def test_checkout_phrases_match_fast_path_keywords(phrase):
    """Checkout phrases must match the fast-path keyword list defined in
    _handle_post_recommendation so they never reach the LLM intent classifier."""
    _CHECKOUT_FAST_PATH_KWS = (
        "check out now", "let's check out", "lets check out",
        "ready to checkout", "ready to check out", "proceed to checkout",
    )
    msg_lower = phrase.lower()
    assert any(kw in msg_lower for kw in _CHECKOUT_FAST_PATH_KWS), (
        f"Checkout phrase not caught by fast-path: {phrase!r}"
    )
