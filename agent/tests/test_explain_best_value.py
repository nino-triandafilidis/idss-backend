"""
Unit tests for _explain_best_value() in agent/chat_endpoint.py (Item 2).

Covers:
- Positive bullets still generated
- ⚠️ Most expensive con when product is priciest in set
- ⚠️ Low RAM con when ram_gb < 16
- ⚠️ Short battery con when battery_life_hours < 6
- ⚠️ Mixed reviews con when rating < 4.0
- ⚠️ Low storage con when storage_gb < 256
- At most 2 cons shown even if multiple apply
- No cons when product is strong across the board
- No cons when only 1 product in set
"""

from agent.chat_endpoint import _explain_best_value


def _product(
    name="Test Laptop",
    price=999,
    rating=4.5,
    reviews=500,
    ram_gb=16,
    storage_gb=512,
    battery=8,
    cpu="Intel i7",
    brand="Dell",
):
    return {
        "name": name,
        "price": price,
        "rating": rating,
        "reviews_count": reviews,
        "brand": brand,
        "attributes": {
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "storage_type": "SSD",
            "battery_life_hours": battery,
            "cpu": cpu,
        },
    }


# ---------------------------------------------------------------------------
# Positive bullets
# ---------------------------------------------------------------------------

def test_explanation_includes_product_name():
    p = _product(name="MacBook Air")
    result = _explain_best_value(p, "laptops")
    assert "MacBook Air" in result


def test_explanation_has_price_bullet():
    p = _product(price=799)
    result = _explain_best_value(p, "laptops")
    assert "799" in result


def test_explanation_has_at_least_3_bullets():
    p = _product()
    result = _explain_best_value(p, "laptops")
    bullets = [l for l in result.split("\n") if l.strip().startswith("- ")]
    assert len(bullets) >= 3


# ---------------------------------------------------------------------------
# Cons: most expensive
# ---------------------------------------------------------------------------

def test_con_most_expensive_when_priciest():
    products = [_product(price=600), _product(price=800), _product(price=1200, name="Best")]
    best = products[-1]
    result = _explain_best_value(best, "laptops", all_products=products)
    assert "⚠️" in result
    assert "expensive" in result.lower()


def test_no_con_most_expensive_when_not_priciest():
    products = [_product(price=600, name="Best"), _product(price=800), _product(price=1200)]
    best = products[0]
    result = _explain_best_value(best, "laptops", all_products=products)
    assert "expensive" not in result.lower()


# ---------------------------------------------------------------------------
# Cons: low RAM
# ---------------------------------------------------------------------------

def test_con_low_ram_when_8gb():
    p = _product(ram_gb=8)
    others = [_product(price=500), _product(price=700)]
    result = _explain_best_value(p, "laptops", all_products=[p] + others)
    assert "8GB RAM" in result or "8gb" in result.lower()
    assert "⚠️" in result


def test_no_con_ram_when_16gb():
    p = _product(ram_gb=16)
    others = [_product(price=500)]
    result = _explain_best_value(p, "laptops", all_products=[p] + others)
    # 16GB should not trigger low-RAM con
    lines_with_warning = [l for l in result.split("\n") if "⚠️" in l and "RAM" in l]
    assert len(lines_with_warning) == 0


# ---------------------------------------------------------------------------
# Cons: short battery
# ---------------------------------------------------------------------------

def test_con_short_battery_when_4h():
    p = _product(battery=4)
    others = [_product(price=500)]
    result = _explain_best_value(p, "laptops", all_products=[p] + others)
    assert "⚠️" in result
    assert "battery" in result.lower() or "4h" in result.lower()


def test_no_con_battery_when_8h():
    p = _product(battery=8)
    others = [_product(price=500)]
    result = _explain_best_value(p, "laptops", all_products=[p] + others)
    lines_with_battery_warning = [l for l in result.split("\n") if "⚠️" in l and "attery" in l]
    assert len(lines_with_battery_warning) == 0


# ---------------------------------------------------------------------------
# Cons: mixed reviews
# ---------------------------------------------------------------------------

def test_con_mixed_reviews_when_low_rating():
    p = _product(rating=3.2, reviews=100)
    others = [_product(price=500)]
    result = _explain_best_value(p, "laptops", all_products=[p] + others)
    assert "⚠️" in result
    assert "3.2" in result


def test_no_con_reviews_when_good_rating():
    p = _product(rating=4.6)
    others = [_product(price=500)]
    result = _explain_best_value(p, "laptops", all_products=[p] + others)
    lines_with_review_warning = [l for l in result.split("\n") if "⚠️" in l and "eview" in l]
    assert len(lines_with_review_warning) == 0


# ---------------------------------------------------------------------------
# Cons: low storage
# ---------------------------------------------------------------------------

def test_con_low_storage_when_128gb():
    p = _product(storage_gb=128)
    others = [_product(price=500)]
    result = _explain_best_value(p, "laptops", all_products=[p] + others)
    assert "⚠️" in result
    assert "128" in result


def test_no_con_storage_when_512gb():
    p = _product(storage_gb=512)
    others = [_product(price=500)]
    result = _explain_best_value(p, "laptops", all_products=[p] + others)
    lines_with_storage_warning = [l for l in result.split("\n") if "⚠️" in l and "storage" in l.lower()]
    assert len(lines_with_storage_warning) == 0


# ---------------------------------------------------------------------------
# Cap at 2 cons
# ---------------------------------------------------------------------------

def test_at_most_2_cons_even_if_many_apply():
    # Worst possible product: expensive, low RAM, short battery, bad reviews, tiny storage
    products = [_product(price=500), _product(price=700)]
    worst = _product(price=1200, ram_gb=4, battery=3, rating=2.5, storage_gb=64)
    products.append(worst)
    result = _explain_best_value(worst, "laptops", all_products=products)
    con_lines = [l for l in result.split("\n") if l.strip().startswith("- ⚠️")]
    assert len(con_lines) <= 2


# ---------------------------------------------------------------------------
# No cons for single-product set
# ---------------------------------------------------------------------------

def test_no_cons_when_single_product():
    p = _product(price=1500, ram_gb=4, battery=3, rating=2.0, storage_gb=64)
    result = _explain_best_value(p, "laptops", all_products=[p])
    assert "⚠️" not in result


def test_no_cons_when_all_products_is_none():
    p = _product(price=1500, ram_gb=4, battery=3, rating=2.0, storage_gb=64)
    result = _explain_best_value(p, "laptops", all_products=None)
    assert "⚠️" not in result
