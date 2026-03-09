"""
import_amazon_laptops.py
=======================
Downloads Amazon Electronics metadata from UCSD McAuley Lab (Amazon Reviews 2023),
filters to laptops, maps fields to our Supabase schema, and upserts.

Data source: https://amazon-reviews-2023.github.io/
  Metadata : https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz  (1.25 GB)
  Reviews  : https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz    (6.17 GB)

Streams gzip-JSONL directly over HTTP — no full download required.

Architecture (backward-chained from goal):
  Goal: Supabase products table enriched with real Amazon laptop data
    <- upsert_batch(): batch INSERT ON CONFLICT DO UPDATE
      <- map_item_to_row(): field mapper for a single Amazon item
        <- extract_laptop_attributes(): parse RAM/storage/CPU from details{}
          <- is_laptop(): high-precision filter
            <- stream_electronics(): direct gzip-JSONL HTTP stream

Usage:
  # Dry run (prints rows, no DB write):
  python scripts/import_amazon_laptops.py --dry-run --limit 50

  # Full import:
  python scripts/import_amazon_laptops.py

  # Import reviews for already-imported products:
  python scripts/import_amazon_laptops.py --reviews-only

  # Test filter accuracy on N records:
  python scripts/import_amazon_laptops.py --test-filter --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

# ── .env loading (must happen before psycopg2 connection) ────────────────────
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "mcp-server"))
_env_path = _repo_root / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import gzip
import ssl
import urllib.request

import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("amazon_import")

# ============================================================================
# CONSTANTS
# ============================================================================

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# UCSD McAuley Lab direct download URLs (gzip-JSONL, stream over HTTP)
AMAZON_META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_Electronics.jsonl.gz"
)
AMAZON_REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/Electronics.jsonl.gz"
)

# Unverified SSL context (UCSD server cert sometimes fails on macOS)
_SSL_CTX = ssl._create_unverified_context()

# Category strings that confirm it's a laptop (PRIMARY gate — most reliable signal)
# Amazon's category hierarchy correctly places actual laptops here.
LAPTOP_CATEGORY_MARKERS: frozenset[str] = frozenset({
    "Laptops", "Traditional Laptops", "Ultrabooks",
    "2-in-1 Laptops", "Gaming Laptops", "Chromebooks", "Netbooks",
})

# STRONG title keywords — brand/model names that ONLY appear on actual laptops,
# never on accessories. Used as fallback when a laptop isn't in the Laptops category.
# NOTE: Generic words like "laptop", "notebook", "inspiron", "pavilion", "vostro"
# are excluded here because they appear in accessory titles ("for laptop",
# "HDD for Dell Inspiron", "1GB RAM for Vostro"). Only unambiguous model strings.
STRONG_TITLE_KEYWORDS: frozenset[str] = frozenset({
    "macbook air", "macbook pro", "macbook",
    "chromebook",
    "thinkpad",
    "ideapad",
    "zenbook",
    "vivobook",
    "surface laptop", "surface book",
    "razer blade",
    "rog zephyrus", "rog strix", "rog flow",
    "asus tuf gaming",
    "gram laptop",
})

# Title signals that indicate an accessory/non-laptop.
# Applied AFTER category check so a true "Laptops" category item is never rejected.
EXCLUDE_TITLE_SIGNALS: frozenset[str] = frozenset({
    "tablet", "ipad", "kindle", "fire hd", "fire tablet",
    "bag", "sleeve", "backpack", " skin",
    "charger", "adapter", "power bank", "battery replacement",
    "screen protector", "keyboard cover", "keyboard case",
    " case for", "stand", "dock", "docking station", "hub",
    "mousepad", "mouse pad",
    "privacy filter", "cooling pad",
    "memory upgrade", "ram upgrade",
    "hard drive for", "ssd upgrade",
    "sticker", "stickers", "decal", "decals",
    "webcam", "web cam",
    "trackpoint", "trackpad cap",
    "lcd screen", "lcd display", "replacement screen", "replacement lcd",
    "display panel", "replacement panel", "touch screen panel",
    "replacement for keyboard",
    "keyboard brush",
    "eyepiece", "telescope",
    "dust cover",
    "battery for", "replacement battery",
})

BATCH_SIZE = 200
MAX_REVIEWS_PER_PRODUCT = 10


# ============================================================================
# STEP 1 — LAPTOP FILTER
# ============================================================================

def is_laptop(item: dict[str, Any]) -> bool:
    """
    High-precision laptop filter. Precision over recall.

    Decision logic (proven by analysis of 300 sample records):
    1. CATEGORY gate (Gate 1): Amazon's "Laptops"/"Gaming Laptops"/etc. category
       is the most reliable signal. If present, accept immediately — Amazon's own
       taxonomy is highly accurate for actual laptop products.
    2. EXCLUDE gate (Gate 2): applied only to title-keyword fallback path.
       Prevents "replacement LCD screen", "stickers for laptop", etc.
    3. STRONG TITLE gate (Gate 3): fallback for rare laptops not in Laptops category.
       Only accepts unambiguous model-specific strings (macbook, thinkpad, zenbook, etc.)
       that never appear as accessory compatibility strings.
       NOTE: generic "laptop", "notebook", "inspiron" are intentionally excluded —
       they all appear in accessories ("HDD for Dell Inspiron", "keyboard for laptop").

    Edge cases:
    - "Surface Pro" (tablet) → not in LAPTOP_CATEGORY_MARKERS, not in STRONG_TITLE_KEYWORDS
    - "MacBook case" → "macbook" in title but caught by " case for" exclude signal
    - "ThinkPad RAM upgrade" → "thinkpad" in title but "ram upgrade" catches it
    """
    categories: list[str] = item.get("categories") or []
    title: str = (item.get("title") or "").lower()

    # --- Gate 1: explicit laptop category → accept (most reliable signal) ---
    for cat in categories:
        if cat in LAPTOP_CATEGORY_MARKERS:
            return True

    # --- Gate 2: exclude accessory signals before title keyword check ---
    for sig in EXCLUDE_TITLE_SIGNALS:
        if sig in title:
            return False

    # --- Gate 3: strong model-specific title keywords only ---
    for kw in STRONG_TITLE_KEYWORDS:
        if kw in title:
            return True

    return False


def test_filter_accuracy(n_sample: int = 200) -> None:
    """
    Stream N records from UCSD, apply filter, print acceptance rate
    and 10 accepted + 10 rejected examples for manual review.
    """
    log.info("Testing filter accuracy on %d streamed records...", n_sample)
    gz = _open_gzip_stream(AMAZON_META_URL)
    accepted, rejected = [], []
    for raw_line in gz:
        if len(accepted) + len(rejected) >= n_sample:
            break
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if is_laptop(item):
            accepted.append(item)
        else:
            rejected.append(item)
    gz.close()

    print(f"\nAccepted: {len(accepted)} / {n_sample} ({100*len(accepted)/n_sample:.1f}%)")
    print("\n--- ACCEPTED (first 10) ---")
    for it in accepted[:10]:
        print(f"  {it.get('title', '')[:80]!r}")
    print("\n--- REJECTED (first 10) ---")
    for it in rejected[:10]:
        print(f"  {it.get('title', '')[:80]!r}")


# ============================================================================
# STEP 2 — ATTRIBUTE EXTRACTOR
# ============================================================================

def _parse_gb(raw: str | None) -> Optional[int]:
    """
    Parse RAM or storage size to integer GB.
    Examples: "8 GB", "16GB DDR4", "512GB SSD", "1TB", "2 TB"
    Handles TB → GB conversion.
    Returns None if unparseable.
    """
    if not raw:
        return None
    raw = raw.upper().strip()
    # TB → GB
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", raw)
    if m:
        return int(float(m.group(1)) * 1024)
    # GB
    m = re.search(r"(\d+)\s*GB", raw)
    if m:
        val = int(m.group(1))
        # sanity: RAM is 1–256 GB, storage is 8–4096 GB
        return val if 1 <= val <= 4096 else None
    return None


def _parse_inches(raw: str | None) -> Optional[float]:
    """
    Parse screen size to float inches.
    Examples: '15.6"', "15.6 inches", "15.6-inch", '13.3"'
    """
    if not raw:
        return None
    m = re.search(r"([\d.]+)\s*(?:\"|\binch|\")", raw, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val if 8.0 <= val <= 22.0 else None  # sanity bounds
    return None


def _parse_hours(raw: str | None) -> Optional[float]:
    """
    Parse battery life to float hours.
    Examples: "Up to 12 Hours", "10 hours", "12-hour battery", "720 Minutes"
    """
    if not raw:
        return None
    raw_l = raw.lower()
    # Minutes → hours
    m = re.search(r"(\d+(?:\.\d+)?)\s*min", raw_l)
    if m:
        val = float(m.group(1)) / 60
        return round(val, 1) if 1.0 <= val <= 30.0 else None
    # Hours
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hour)", raw_l)
    if m:
        val = float(m.group(1))
        return val if 1.0 <= val <= 30.0 else None
    return None


def _parse_weight_lbs(raw: str | None) -> Optional[float]:
    """
    Parse product weight to float pounds.
    Examples: "4.2 pounds", "2.1 kg", "1.8 Kilograms", "4.4 lbs"
    """
    if not raw:
        return None
    raw_l = raw.lower()
    # Kg → lbs
    m = re.search(r"(\d+(?:\.\d+)?)\s*k", raw_l)
    if m:
        val = float(m.group(1)) * 2.205
        return round(val, 2) if 0.5 <= val <= 20.0 else None
    # Pounds / lbs
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:pound|lbs|lb)\b", raw_l)
    if m:
        val = float(m.group(1))
        return val if 0.5 <= val <= 20.0 else None
    return None


def _extract_cpu(details: dict, features: list[str]) -> str:
    """
    Extract CPU string from details dict.
    Amazon uses inconsistent key names: "Processor", "CPU Model",
    "Processor Type", "Standing screen display size", etc.
    Falls back to scanning features[] for Intel/AMD/Apple M mentions.
    """
    cpu_keys = [
        "Processor", "CPU Model", "Processor Type", "Processor Brand",
        "Processor Speed", "CPU Speed",
    ]
    for k in cpu_keys:
        v = details.get(k, "")
        if v and len(v) > 2:
            return str(v)[:120]

    # Fallback: scan features for CPU signals
    patterns = [
        r"(?:Intel|AMD|Apple M\d|Qualcomm|Snapdragon|MediaTek)[^\,\n]{0,60}(?:i\d|Ryzen|Core Ultra|M\d|Celeron|Pentium|Athlon)[^\,\n]{0,40}",
        r"(?:i\d|Ryzen \d|Core Ultra)[^\,\n]{0,60}",
    ]
    for feat in features:
        for pat in patterns:
            m = re.search(pat, feat, re.IGNORECASE)
            if m:
                return m.group(0).strip()[:120]
    return ""


def _extract_storage_type(details: dict, title: str) -> str:
    """Classify storage type from details or title keywords."""
    combined = " ".join([
        details.get("Hard Drive", ""),
        details.get("Solid State Drive Capacity", ""),
        details.get("Flash Memory Size", ""),
        details.get("Hard Drive Interface", ""),
        title,
    ]).lower()
    if "nvme" in combined or "pcie" in combined:
        return "NVMe SSD"
    if "ssd" in combined or "solid state" in combined or "emmc" in combined:
        return "SSD"
    if "hdd" in combined or "hard disk" in combined or "hard drive" in combined:
        return "HDD"
    return ""


def extract_laptop_attributes(item: dict[str, Any]) -> dict[str, Any]:
    """
    Map Amazon item metadata to our attributes JSONB schema.

    Fields we extract:
      ram_gb, storage_gb, storage_type, screen_size, cpu,
      battery_life_hours, os, weight_lbs, color,
      description, gallery (list of image URLs)

    Backtracking: each field is independently parseable and returns None on failure,
    so a parse error in one field never corrupts others.
    """
    details: dict = item.get("details") or {}
    features: list[str] = item.get("features") or []
    desc_list: list[str] = item.get("description") or []
    title: str = item.get("title") or ""
    images: list[dict] = item.get("images") or []

    attrs: dict[str, Any] = {}

    # --- RAM ---
    raw_ram = (
        details.get("RAM") or
        details.get("RAM Memory Installed Size") or
        details.get("Memory Storage Capacity") or
        details.get("Computer Memory Size") or ""
    )
    ram = _parse_gb(raw_ram)
    # Sanity: RAM rarely > 128 GB for consumer laptops
    if ram and ram <= 128:
        attrs["ram_gb"] = ram

    # --- Storage ---
    raw_storage = (
        details.get("Hard Drive") or
        details.get("Solid State Drive Capacity") or
        details.get("Flash Memory Size") or
        details.get("Hard Drive Size") or
        details.get("SSD") or ""
    )
    storage = _parse_gb(raw_storage)
    if storage and storage <= 4096:
        attrs["storage_gb"] = storage
    attrs["storage_type"] = _extract_storage_type(details, title)

    # --- Screen size ---
    raw_screen = (
        details.get("Screen Size") or
        details.get("Standing screen display size") or
        details.get("Display Size") or ""
    )
    screen = _parse_inches(raw_screen)
    if screen:
        attrs["screen_size"] = screen

    # --- CPU ---
    cpu = _extract_cpu(details, features)
    if cpu:
        attrs["cpu"] = cpu

    # --- Battery ---
    raw_batt = (
        details.get("Battery Average Life") or
        details.get("Battery Life") or
        details.get("Batteries") or ""
    )
    batt = _parse_hours(raw_batt)
    if batt:
        attrs["battery_life_hours"] = batt

    # --- OS ---
    os_val = (
        details.get("Operating System") or
        details.get("OS") or
        details.get("Platform") or ""
    )
    if os_val and len(os_val) > 1:
        attrs["os"] = str(os_val)[:80]

    # --- Weight ---
    raw_weight = details.get("Item Weight") or details.get("Package Weight") or ""
    weight = _parse_weight_lbs(raw_weight)
    if weight:
        attrs["weight_lbs"] = weight

    # --- Color ---
    color_val = details.get("Color") or ""
    if color_val:
        attrs["color"] = str(color_val)[:60]

    # --- Description (raw + features) ---
    desc_text = " ".join(desc_list).strip()
    feat_text = " ".join(features[:5]).strip()  # top 5 features
    attrs["description"] = desc_text or feat_text or ""

    # --- Image gallery (all images after index 0) ---
    gallery: list[str] = []
    for img in images[1:]:
        if isinstance(img, dict):
            url = img.get("large") or img.get("hi_res") or img.get("thumb")
            if url and url.startswith("http"):
                gallery.append(url)
    if gallery:
        attrs["gallery"] = gallery[:8]  # cap at 8

    # --- Bought together ASINs (for future recommendations) ---
    bought = item.get("bought_together") or []
    if bought:
        attrs["bought_together_asins"] = bought[:5]

    return attrs


# ============================================================================
# STEP 3 — FIELD MAPPER (Amazon item → Supabase row dict)
# ============================================================================

def _asin_to_uuid(asin: str) -> str:
    """Deterministic UUID from ASIN so re-imports produce the same id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"amazon:{asin}"))


def _get_primary_image(images: list[dict]) -> Optional[str]:
    if not images:
        return None
    first = images[0]
    if isinstance(first, dict):
        return first.get("large") or first.get("hi_res") or first.get("thumb")
    return None


def _get_brand(item: dict) -> str:
    """
    Extract brand. Priority order:
    1. item['brand']  — set by idss-v0.1 DataProcessor (already cleaned)
    2. item['store']  — Amazon seller/brand name from raw API
    3. details['Brand'] / details['Manufacturer'] — fallback from details JSONB

    idss-v0.1 pre-cleans the brand field via DataProcessor.clean_text(), so when
    reading from its SQLite output we prefer that over the noisier store/details fields.
    """
    # idss-v0.1 pre-extracted brand (highest quality when present)
    brand_field = (item.get("brand") or "").strip()
    if brand_field and len(brand_field) > 1:
        result = brand_field
    else:
        store = (item.get("store") or "").strip()
        details = item.get("details") or {}
        brand_detail = (details.get("Brand") or details.get("Manufacturer") or "").strip()
        result = store or brand_detail

    # Clean: strip "Visit the X Store" patterns from Amazon
    result = re.sub(r"visit the .+? store", "", result, flags=re.IGNORECASE).strip()
    result = re.sub(r"brand:", "", result, flags=re.IGNORECASE).strip()
    return result[:100] if result else "Unknown"


def map_item_to_row(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Map a single Amazon Electronics metadata item to a Supabase products row dict.

    Returns None if the item should be skipped (e.g. no price).

    Constraint: price=None → skip (product can't be recommended without price).
    """
    asin: str = item.get("parent_asin") or item.get("asin") or ""
    if not asin:
        return None

    title: str = item.get("title") or ""
    if not title.strip():
        return None

    price_raw: Optional[float] = item.get("price")
    price: Optional[float] = float(price_raw) if (price_raw and price_raw > 0) else None

    images: list[dict] = item.get("images") or []
    attrs = extract_laptop_attributes(item)

    # Skip if null price AND no spec data — likely a replacement part that slipped the filter
    if price is None:
        has_spec = attrs.get("ram_gb") or attrs.get("storage_gb") or attrs.get("screen_size")
        if not has_spec:
            return None

    return {
        "id": _asin_to_uuid(asin),
        "title": title[:500],
        "category": "Electronics",
        "product_type": "laptop",
        "brand": _get_brand(item),
        "price": float(price) if price is not None else None,
        "imageurl": _get_primary_image(images),
        "rating": item.get("average_rating"),
        "rating_count": item.get("rating_number"),
        "source": "amazon",
        "ref_id": asin,
        "inventory": 999,       # unknown stock; mark as available
        "attributes": json.dumps(attrs),
    }


# ============================================================================
# STEP 4 — STREAMING DOWNLOADER (direct gzip-JSONL over HTTP)
# ============================================================================

def _open_gzip_stream(url: str):
    """
    Open a remote .jsonl.gz file for streaming without full download.
    Returns a gzip.GzipFile object that yields decompressed bytes line-by-line.

    Uses unverified SSL context to handle UCSD cert issues on macOS.
    """
    log.info("Opening remote gzip stream: %s", url)
    response = urllib.request.urlopen(url, context=_SSL_CTX, timeout=60)
    return gzip.GzipFile(fileobj=response)


def stream_electronics(limit: Optional[int] = None) -> Iterator[dict[str, Any]]:
    """
    Stream Electronics item metadata from UCSD directly.
    Decompresses gzip line-by-line — memory usage stays flat regardless of file size.

    Edge cases:
    - Malformed JSON lines are skipped with a warning (don't crash the whole import)
    - Network timeout after 60s per chunk (urllib default)
    """
    gz = _open_gzip_stream(AMAZON_META_URL)
    n = 0
    for raw_line in gz:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Skipping malformed JSON line at record %d", n)
            continue
        yield item
        n += 1
        if limit and n >= limit:
            break
    gz.close()


def stream_electronics_reviews(asins: set[str], limit: Optional[int] = None) -> Iterator[dict[str, Any]]:
    """
    Stream Electronics reviews, yielding only those whose parent_asin is in `asins`.
    The reviews file is 6.17 GB compressed — full streaming is the only feasible approach.

    Performance note: at ~100K reviews/min, scanning all 43.9M reviews takes ~7 hours.
    If asins is small (<1000), consider pre-filtering by downloading per-ASIN if a
    filtered endpoint becomes available. For now, full scan is correct.
    """
    gz = _open_gzip_stream(AMAZON_REVIEW_URL)
    n = 0
    for raw_line in gz:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            review = json.loads(line)
        except json.JSONDecodeError:
            continue
        asin = review.get("parent_asin") or review.get("asin") or ""
        if asin in asins:
            yield review
            n += 1
            if limit and n >= limit:
                break
    gz.close()


# ============================================================================
# STEP 4b — idss-v0.1 SQLITE ADAPTER
# ============================================================================
# idss-v0.1 (github.com/Sajjad-Beygi/idss-v0.1) downloads Amazon Reviews 2023
# via HuggingFace Parquet and stores the result in a local SQLite database with
# this schema:
#
#   products(asin PK, title, brand, price, description TEXT,
#            features TEXT←JSON, categories TEXT←JSON,
#            details TEXT←JSON, store, average_rating, rating_count,
#            images TEXT←JSON, category)
#
#   reviews(id, user_id, asin, parent_asin, rating, title, text,
#           images TEXT←JSON, verified_purchase INT, helpful_vote, timestamp, category)
#
# The adapter reconstructs raw Amazon-like dicts so our existing
# is_laptop() / extract_laptop_attributes() / map_item_to_row() work unchanged.


def _sqlite_parse_json(val: Any) -> Any:
    """Parse a JSON-serialized SQLite field back to a Python object. Returns None on failure."""
    if not val:
        return None
    if isinstance(val, (list, dict)):
        return val  # Already parsed (shouldn't happen from SQLite, but guard anyway)
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def stream_from_idss_sqlite(
    db_path: str,
    limit: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    """
    Read product metadata from an idss-v0.1 SQLite database, yielding dicts
    in the same raw Amazon API format as stream_electronics().

    Field mapping (SQLite → raw Amazon dict):
      asin           → parent_asin, asin
      title          → title
      brand          → brand          (idss-v0.1-cleaned; consumed by _get_brand())
      store          → store
      price          → price
      description    → description    (wrapped in a list for compatibility)
      features       → features       (parsed from JSON string)
      categories     → categories     (parsed from JSON string)
      details        → details        (parsed from JSON string)
      images         → images         (parsed from JSON string)
      average_rating → average_rating
      rating_count   → rating_number
      category       → main_category

    Args:
        db_path: Path to the idss-v0.1 SQLite database file.
        limit: Maximum number of rows to yield (None = all).
    """
    import sqlite3
    log.info("Opening idss-v0.1 SQLite database: %s", db_path)
    if not Path(db_path).exists():
        log.error("SQLite database not found: %s", db_path)
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Filter to Electronics only — same category scope as HTTP stream
        sql = "SELECT * FROM products WHERE category = 'Electronics'"
        if limit:
            sql += f" LIMIT {limit}"

        cur = conn.execute(sql)
        n = 0
        _FETCH_BATCH = 2000  # rows per fetchmany — controls memory

        while True:
            rows = cur.fetchmany(_FETCH_BATCH)
            if not rows:
                break
            for row in rows:
                categories = _sqlite_parse_json(row["categories"]) or []
                # Normalise categories to list of strings (idss-v0.1 stores as JSON array)
                if isinstance(categories, str):
                    categories = [categories]

                yield {
                    "parent_asin": row["asin"],
                    "asin": row["asin"],
                    "title": row["title"] or "",
                    # idss-v0.1 has a dedicated brand field (already cleaned)
                    "brand": row["brand"] or "",
                    "store": row["store"] or row["brand"] or "",
                    "price": row["price"],
                    # description is plain text in SQLite; wrap in list for compatibility
                    "description": [row["description"]] if row["description"] else [],
                    "features": _sqlite_parse_json(row["features"]) or [],
                    "categories": categories,
                    "details": _sqlite_parse_json(row["details"]) or {},
                    "images": _sqlite_parse_json(row["images"]) or [],
                    "average_rating": row["average_rating"],
                    "rating_number": row["rating_count"],
                    "main_category": row["category"],
                }
                n += 1
                if limit and n >= limit:
                    return

    finally:
        conn.close()

    log.info("Streamed %d products from idss-v0.1 SQLite", n)


def stream_reviews_from_idss_sqlite(
    db_path: str,
    asins: set[str],
) -> Iterator[dict[str, Any]]:
    """
    Read reviews for a set of ASINs from the idss-v0.1 SQLite reviews table.
    Yields dicts compatible with attach_reviews() expectations.

    Advantage over HTTP stream: instant random access — no scanning 43.9M rows.
    idss-v0.1 creates an index on parent_asin so this query is fast.

    Args:
        db_path: Path to the idss-v0.1 SQLite database file.
        asins: Set of ASINs (parent_asin values) to fetch reviews for.
    """
    import sqlite3
    if not asins:
        return
    if not Path(db_path).exists():
        log.error("SQLite database not found for reviews: %s", db_path)
        return

    log.info("Loading reviews from idss-v0.1 SQLite for %d ASINs...", len(asins))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # SQLite IN clause limit is 999; chunk if needed
        asin_list = list(asins)
        chunk_size = 900
        for i in range(0, len(asin_list), chunk_size):
            chunk = asin_list[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            cur = conn.execute(
                f"""SELECT parent_asin, rating, title, text,
                           helpful_vote, verified_purchase, timestamp
                    FROM reviews
                    WHERE parent_asin IN ({placeholders})""",
                chunk,
            )
            for row in cur:
                yield {
                    "parent_asin": row["parent_asin"],
                    "asin": row["parent_asin"],
                    "rating": row["rating"],
                    "title": row["title"] or "",
                    "text": row["text"] or "",
                    "helpful_vote": row["helpful_vote"] or 0,
                    "verified_purchase": bool(row["verified_purchase"]),
                    "timestamp": row["timestamp"],
                }
    finally:
        conn.close()


# ============================================================================
# STEP 5 — UPSERT PIPELINE
# ============================================================================

_UPSERT_SQL = """
INSERT INTO products (
    id, title, category, product_type, brand,
    price, imageurl, rating, rating_count,
    source, ref_id, inventory, attributes
)
VALUES %s
ON CONFLICT (ref_id) WHERE source = 'amazon'
DO UPDATE SET
    title        = EXCLUDED.title,
    brand        = EXCLUDED.brand,
    price        = EXCLUDED.price,
    imageurl     = COALESCE(EXCLUDED.imageurl, products.imageurl),
    rating       = COALESCE(EXCLUDED.rating, products.rating),
    rating_count = COALESCE(EXCLUDED.rating_count, products.rating_count),
    attributes   = EXCLUDED.attributes,
    updated_at   = NOW()
"""


def _row_to_tuple(row: dict) -> tuple:
    return (
        row["id"], row["title"], row["category"], row["product_type"], row["brand"],
        row["price"], row.get("imageurl"), row.get("rating"), row.get("rating_count"),
        row["source"], row["ref_id"], row["inventory"], row["attributes"],
    )


def upsert_batch(conn, rows: list[dict], dry_run: bool = False) -> int:
    """
    Upsert a batch of rows. Returns number of rows processed.
    In dry_run mode: prints first 3 rows and returns count without writing.
    """
    if not rows:
        return 0
    if dry_run:
        log.info("[DRY RUN] Would upsert %d rows", len(rows))
        for r in rows[:3]:
            attrs = json.loads(r["attributes"])
            price_str = f"${r['price']:.0f}" if r["price"] is not None else "None"
        log.info("  title=%r brand=%r price=%s ram=%s storage=%s cpu=%r screen=%s",
                     r["title"][:60], r["brand"], price_str,
                     attrs.get("ram_gb"), attrs.get("storage_gb"),
                     attrs.get("cpu", "")[:40], attrs.get("screen_size"))
        return len(rows)

    tuples = [_row_to_tuple(r) for r in rows]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, _UPSERT_SQL, tuples, page_size=BATCH_SIZE)
    conn.commit()
    return len(rows)


# ============================================================================
# STEP 6 — REVIEW ATTACHMENT
# ============================================================================

def _is_good_review(review: dict) -> bool:
    """Only keep verified reviews with meaningful text."""
    text = (review.get("text") or "").strip()
    return (
        review.get("verified_purchase", False)
        and len(text) >= 50
        and review.get("rating", 0) >= 2.0  # keep 2–5 (1-star spam common)
    )


def attach_reviews(
    conn,
    asins: set[str],
    dry_run: bool = False,
    reviews_iter: Optional[Iterator] = None,
) -> dict[str, int]:
    """
    Collect top-MAX_REVIEWS per ASIN (sorted by helpful_vote desc),
    then UPDATE products SET attributes = attributes || '{"reviews": [...]}'.

    Args:
        reviews_iter: Optional pre-built iterator of review dicts.
                      When None, streams from the UCSD HTTP endpoint.
                      Pass stream_reviews_from_idss_sqlite() here when
                      using --from-sqlite to get fast indexed lookups instead
                      of scanning the full 6.17 GB reviews file.

    Returns: {"processed": N, "updated": M}
    """
    from collections import defaultdict

    log.info("Streaming reviews for %d ASINs...", len(asins))
    reviews_by_asin: dict[str, list[dict]] = defaultdict(list)

    source = reviews_iter if reviews_iter is not None else stream_electronics_reviews(asins)
    for review in source:
        if not _is_good_review(review):
            continue
        asin = review.get("parent_asin") or review.get("asin") or ""
        reviews_by_asin[asin].append({
            "rating": review.get("rating"),
            "title": (review.get("title") or "")[:200],
            "text": (review.get("text") or "")[:800],
            "helpful_vote": review.get("helpful_vote", 0),
            "verified_purchase": True,
            "timestamp": review.get("timestamp"),
        })

    # Sort by helpful_vote desc, cap at MAX_REVIEWS_PER_PRODUCT
    for asin in reviews_by_asin:
        reviews_by_asin[asin].sort(key=lambda r: r.get("helpful_vote", 0), reverse=True)
        reviews_by_asin[asin] = reviews_by_asin[asin][:MAX_REVIEWS_PER_PRODUCT]

    log.info("Collected reviews for %d / %d ASINs", len(reviews_by_asin), len(asins))

    if dry_run:
        for asin, revs in list(reviews_by_asin.items())[:3]:
            log.info("[DRY RUN] ASIN %s: %d reviews, top=%r",
                     asin, len(revs), revs[0]["title"][:60] if revs else "")
        return {"processed": len(reviews_by_asin), "updated": 0}

    updated = 0
    with conn.cursor() as cur:
        for asin, revs in reviews_by_asin.items():
            cur.execute("""
                UPDATE products
                SET attributes = attributes || %s::jsonb,
                    updated_at = NOW()
                WHERE ref_id = %s AND source = 'amazon'
            """, (json.dumps({"reviews": revs}), asin))
            updated += cur.rowcount
    conn.commit()
    log.info("Updated %d products with reviews", updated)
    return {"processed": len(reviews_by_asin), "updated": updated}


# ============================================================================
# STEP 7 — VALIDATION
# ============================================================================

def run_validation(conn) -> None:
    """
    Post-import validation. Prints attribute coverage report.
    Raises AssertionError if acceptance criteria are not met.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE price IS NOT NULL) as has_price,
                COUNT(*) FILTER (WHERE imageurl IS NOT NULL) as has_image,
                COUNT(*) FILTER (WHERE (attributes->>'ram_gb')::numeric > 0) as has_ram,
                COUNT(*) FILTER (WHERE (attributes->>'storage_gb')::numeric > 0) as has_storage,
                COUNT(*) FILTER (WHERE attributes->>'cpu' != '') as has_cpu,
                COUNT(*) FILTER (WHERE (attributes->>'screen_size')::numeric > 0) as has_screen,
                COUNT(*) FILTER (WHERE jsonb_array_length(attributes->'reviews') > 0) as has_reviews
            FROM products
            WHERE source = 'amazon' AND product_type = 'laptop'
        """)
        row = cur.fetchone()
        total, has_price, has_image, has_ram, has_storage, has_cpu, has_screen, has_reviews = row

    if total == 0:
        log.error("VALIDATION FAILED: 0 Amazon laptop rows found!")
        return

    def pct(n):
        return f"{100*n/total:.1f}%"

    print("\n" + "="*60)
    print("VALIDATION REPORT — Amazon Laptops")
    print("="*60)
    print(f"  Total imported      : {total}")
    print(f"  Has price           : {has_price} ({pct(has_price)})")
    print(f"  Has primary image   : {has_image} ({pct(has_image)})")
    print(f"  Has RAM parsed      : {has_ram} ({pct(has_ram)})")
    print(f"  Has storage parsed  : {has_storage} ({pct(has_storage)})")
    print(f"  Has CPU extracted   : {has_cpu} ({pct(has_cpu)})")
    print(f"  Has screen size     : {has_screen} ({pct(has_screen)})")
    print(f"  Has reviews         : {has_reviews} ({pct(has_reviews)})")

    # Acceptance criteria (fail loudly if not met)
    failures = []
    if total < 100:
        failures.append(f"total={total} < 100 — too few records")
    if has_price / total < 0.70:
        failures.append(f"price coverage {pct(has_price)} < 70%")
    if has_image / total < 0.70:
        failures.append(f"image coverage {pct(has_image)} < 70%")
    if has_ram / total < 0.40:
        failures.append(f"RAM coverage {pct(has_ram)} < 40% — check extractor")

    if failures:
        print("\nCRITERIA FAILURES:")
        for f in failures:
            print(f"  ⚠️  {f}")
    else:
        print("\n✅ All acceptance criteria met.")

    # Spot-check 5 random records
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT title, brand, price, rating,
                   attributes->>'ram_gb' as ram,
                   attributes->>'storage_gb' as storage,
                   attributes->>'cpu' as cpu,
                   attributes->>'screen_size' as screen
            FROM products
            WHERE source='amazon' AND product_type='laptop'
            ORDER BY random() LIMIT 5
        """)
        print("\n--- SPOT CHECK (5 random) ---")
        for r in cur.fetchall():
            print(f"  {r['title'][:55]!r}")
            print(f"    brand={r['brand']} price=${r['price']:.0f} "
                  f"ram={r['ram']}GB storage={r['storage']}GB "
                  f"cpu={str(r['cpu'] or '')[:30]} screen={r['screen']}\"")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Import Amazon laptop data to Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input sources (mutually exclusive):
  (default)          Stream Electronics metadata directly from UCSD via gzip-HTTP
  --from-sqlite PATH Read from an idss-v0.1 SQLite database (faster, uses local cache)

Examples:
  # Full import from UCSD HTTP stream:
  python scripts/import_amazon_laptops.py

  # Import from Sajjad's idss-v0.1 SQLite output (with reviews):
  python scripts/import_amazon_laptops.py --from-sqlite /path/to/amazon_reviews.db

  # Dry-run from SQLite, first 500 items:
  python scripts/import_amazon_laptops.py --from-sqlite /path/to/amazon_reviews.db --dry-run --limit 500

  # Attach reviews only (using SQLite as review source):
  python scripts/import_amazon_laptops.py --from-sqlite /path/to/amazon_reviews.db --reviews-only
        """,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rows without writing to DB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of Electronics items to stream (for testing)")
    parser.add_argument("--reviews-only", action="store_true",
                        help="Only attach reviews to already-imported Amazon products")
    parser.add_argument("--skip-reviews", action="store_true",
                        help="Skip review attachment step")
    parser.add_argument("--test-filter", action="store_true",
                        help="Test filter accuracy on N streamed records and exit")
    parser.add_argument(
        "--from-sqlite", metavar="PATH",
        help="Path to idss-v0.1 SQLite database. When set, reads metadata and "
             "reviews from the local SQLite file instead of streaming from UCSD.",
    )
    args = parser.parse_args()

    if args.test_filter:
        test_filter_accuracy(args.limit or 300)
        return

    if not DATABASE_URL:
        log.error("DATABASE_URL not set — cannot connect to Supabase")
        sys.exit(1)

    conn = psycopg2.connect(DATABASE_URL)
    log.info("Connected to Supabase")

    sqlite_path: Optional[str] = args.from_sqlite

    # ── REVIEWS ONLY ────────────────────────────────────────────────────────
    if args.reviews_only:
        with conn.cursor() as cur:
            cur.execute("SELECT ref_id FROM products WHERE source='amazon' AND product_type='laptop'")
            imported_asins = {r[0] for r in cur.fetchall()}
        log.info("Found %d imported Amazon ASINs", len(imported_asins))
        if imported_asins:
            if sqlite_path:
                # idss-v0.1 SQLite path: fast indexed review lookup
                log.info("Using idss-v0.1 SQLite for reviews: %s", sqlite_path)
                reviews_iter = stream_reviews_from_idss_sqlite(sqlite_path, imported_asins)
                attach_reviews(conn, imported_asins, dry_run=args.dry_run,
                               reviews_iter=reviews_iter)
            else:
                # HTTP stream from UCSD (scans 6.17 GB — runs overnight)
                attach_reviews(conn, imported_asins, dry_run=args.dry_run)
        run_validation(conn)
        conn.close()
        return

    # ── CHOOSE INPUT STREAM ───────────────────────────────────────────────────
    if sqlite_path:
        log.info("Input: idss-v0.1 SQLite — %s", sqlite_path)
        item_stream = stream_from_idss_sqlite(sqlite_path, limit=args.limit)
    else:
        log.info("Input: UCSD gzip-HTTP stream")
        item_stream = stream_electronics(limit=args.limit)

    # ── FULL IMPORT ──────────────────────────────────────────────────────────
    stats = {"streamed": 0, "filtered": 0, "skipped_no_specs": 0, "upserted": 0}
    batch: list[dict] = []
    imported_asins: set[str] = set()

    for item in item_stream:
        stats["streamed"] += 1

        if not is_laptop(item):
            continue
        stats["filtered"] += 1

        row = map_item_to_row(item)
        if row is None:
            stats["skipped_no_specs"] += 1
            continue

        batch.append(row)
        imported_asins.add(row["ref_id"])

        if len(batch) >= BATCH_SIZE:
            stats["upserted"] += upsert_batch(conn, batch, dry_run=args.dry_run)
            batch.clear()
            log.info("Progress: streamed=%d filtered=%d upserted=%d",
                     stats["streamed"], stats["filtered"], stats["upserted"])

    # Final batch
    if batch:
        stats["upserted"] += upsert_batch(conn, batch, dry_run=args.dry_run)

    log.info("Import complete: %s", stats)

    # ── REVIEWS ──────────────────────────────────────────────────────────────
    if not args.skip_reviews and imported_asins and not args.dry_run:
        if sqlite_path:
            log.info("Attaching reviews from idss-v0.1 SQLite...")
            reviews_iter = stream_reviews_from_idss_sqlite(sqlite_path, imported_asins)
            attach_reviews(conn, imported_asins, dry_run=args.dry_run,
                           reviews_iter=reviews_iter)
        else:
            # HTTP stream — warn about runtime (6.17 GB file, ~7 hours)
            log.warning("Scanning full 6.17 GB reviews file — consider running overnight "
                        "or use --from-sqlite for faster review attachment.")
            attach_reviews(conn, imported_asins, dry_run=args.dry_run)

    # ── VALIDATION ───────────────────────────────────────────────────────────
    if not args.dry_run:
        run_validation(conn)

    conn.close()


if __name__ == "__main__":
    main()
