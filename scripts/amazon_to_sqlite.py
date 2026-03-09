"""
amazon_to_sqlite.py
===================
Downloads Amazon product metadata from UCSD McAuley Lab and converts it to a
local SQLite database for development and testing.

Supports two dataset versions:
  --source v2    Amazon Reviews 2018 (v2) — lighter metadata, 786K products
                 URL: mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/
  --source 2023  Amazon Reviews 2023    — richer attributes, newer data (default)
                 URL: mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/

The SQLite schema mirrors the Supabase products table so data can be used as a
drop-in local replacement or to seed test fixtures.

Usage
-----
  # Recommended: 500 modern laptops with rich specs → dev.db
  # (scans up to 500K records from the 2023 Electronics file)
  python scripts/amazon_to_sqlite.py --limit 500 --scan-limit 500000

  # Faster but older data (v2 / 2018):
  python scripts/amazon_to_sqlite.py --source v2 --limit 500 --scan-limit 200000

  # Dry run — see sample output without writing (defaults to scanning 50K records)
  python scripts/amazon_to_sqlite.py --dry-run

  # Custom output path + reviews
  python scripts/amazon_to_sqlite.py --output /tmp/laptops.db --limit 500 --reviews

  # Non-laptop category (all items, no laptop filter)
  python scripts/amazon_to_sqlite.py --category "Books" --no-laptop-filter --limit 2000

Performance notes
-----------------
  The Electronics file is 1.25 GB compressed (~10 GB uncompressed). Laptops are
  ~5-10% of all Electronics items, spread throughout alphabetically by ASIN.
  Use --scan-limit to cap streaming time (every ~200K records ≈ 10 seconds).
  With --scan-limit 500000 you'll typically find 2000-5000 laptops.

  Dataset quality:
    2023 — richer structured specs (RAM, CPU, screen, battery); recommended
    v2   — sparser metadata for older items; early ASINs are 2001-2005 era products

Schema
------
  products  — one row per ASIN; UUID id matches Supabase convention
  reviews   — optional; populated with --reviews flag
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sqlite3
import ssl
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("amazon_sqlite")

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------

_V2_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
_2023_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

DATASET_URLS: dict[str, dict[str, str]] = {
    "v2": {
        "meta": f"{_V2_BASE}/metaFiles2/meta_{{category}}.json.gz",
        "reviews": f"{_V2_BASE}/categoryFiles/{{category}}.json.gz",
    },
    "2023": {
        "meta": f"{_2023_BASE}/meta_categories/meta_{{category}}.jsonl.gz",
        "reviews": f"{_2023_BASE}/review_categories/{{category}}.jsonl.gz",
    },
}

# Unverified SSL — UCSD cert sometimes fails on macOS
_SSL = ssl._create_unverified_context()

# ---------------------------------------------------------------------------
# Laptop filter constants (same logic as import_amazon_laptops.py)
# ---------------------------------------------------------------------------

_LAPTOP_CATS: frozenset[str] = frozenset({
    "Laptops", "Traditional Laptops", "Ultrabooks",
    "2-in-1 Laptops", "Gaming Laptops", "Chromebooks", "Netbooks",
    "Laptops & Netbooks",
})

_STRONG_KWS: frozenset[str] = frozenset({
    "macbook air", "macbook pro", "macbook",
    "chromebook", "thinkpad", "ideapad",
    "zenbook", "vivobook",
    "surface laptop", "surface book",
    "razer blade",
    "rog zephyrus", "rog strix", "rog flow",
    "asus tuf gaming",
    "gram laptop",
})

_EXCLUDE_KWS: frozenset[str] = frozenset({
    "tablet", "ipad", "kindle", "fire hd", "fire tablet",
    "bag", "sleeve", "backpack", " skin",
    "charger", "adapter", "power bank", "battery replacement",
    "screen protector", "keyboard cover", "keyboard case",
    " case for", "stand", "dock", "docking station", "hub",
    "mousepad", "mouse pad",
    "privacy filter", "cooling pad",
    "memory upgrade", "ram upgrade",
    "hard drive for", "ssd upgrade",
    "sticker", "stickers", "decal",
    "webcam", "web cam",
    "lcd screen", "lcd display", "replacement screen",
    "display panel", "replacement panel",
    "replacement for keyboard", "dust cover",
    "battery for", "replacement battery",
})


def is_laptop(item: dict, source: str) -> bool:
    """Return True if item is a laptop product (not an accessory)."""
    title: str = (item.get("title") or "").lower()
    categories = item.get("categories") or []

    # Flatten categories: v2 uses list-of-lists; 2023 uses flat list of strings
    flat_cats: list[str] = []
    for c in categories:
        if isinstance(c, list):
            flat_cats.extend(c)
        elif isinstance(c, str):
            flat_cats.append(c)

    # Gate 1: explicit laptop category
    for cat in flat_cats:
        if cat in _LAPTOP_CATS:
            return True

    # Gate 2: exclude accessory title signals
    for sig in _EXCLUDE_KWS:
        if sig in title:
            return False

    # Gate 3: strong model-specific title keywords
    for kw in _STRONG_KWS:
        if kw in title:
            return True

    return False


# ---------------------------------------------------------------------------
# Attribute parsers (unchanged from import_amazon_laptops.py)
# ---------------------------------------------------------------------------

def _parse_gb(raw: str | None) -> Optional[int]:
    if not raw:
        return None
    raw = raw.upper().strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", raw)
    if m:
        return int(float(m.group(1)) * 1024)
    m = re.search(r"(\d+)\s*GB", raw)
    if m:
        val = int(m.group(1))
        return val if 1 <= val <= 4096 else None
    return None


def _parse_inches(raw: str | None) -> Optional[float]:
    if not raw:
        return None
    m = re.search(r"([\d.]+)\s*(?:\"|\binch|\")", raw, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val if 8.0 <= val <= 22.0 else None
    return None


def _parse_hours(raw: str | None) -> Optional[float]:
    if not raw:
        return None
    raw_l = raw.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*min", raw_l)
    if m:
        val = float(m.group(1)) / 60
        return round(val, 1) if 1.0 <= val <= 30.0 else None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hour)", raw_l)
    if m:
        val = float(m.group(1))
        return val if 1.0 <= val <= 30.0 else None
    return None


def _parse_weight_lbs(raw: str | None) -> Optional[float]:
    if not raw:
        return None
    raw_l = raw.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*k", raw_l)
    if m:
        val = float(m.group(1)) * 2.205
        return round(val, 2) if 0.5 <= val <= 20.0 else None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:pound|lbs|lb)\b", raw_l)
    if m:
        val = float(m.group(1))
        return val if 0.5 <= val <= 20.0 else None
    return None


def _extract_cpu(details: dict, features: list[str]) -> str:
    cpu_keys = [
        "Processor", "CPU Model", "Processor Type",
        "Processor Brand", "Processor Speed", "CPU Speed",
    ]
    for k in cpu_keys:
        v = details.get(k, "")
        if v and len(v) > 2:
            return str(v)[:120]
    patterns = [
        r"(?:Intel|AMD|Apple M\d|Qualcomm)[^\,\n]{0,60}(?:i\d|Ryzen|Core Ultra|M\d|Celeron|Pentium|Athlon)[^\,\n]{0,40}",
        r"(?:i\d|Ryzen \d|Core Ultra)[^\,\n]{0,60}",
    ]
    for feat in features:
        for pat in patterns:
            m = re.search(pat, feat, re.IGNORECASE)
            if m:
                return m.group(0).strip()[:120]
    return ""


# ---------------------------------------------------------------------------
# Item → row mapper (handles both v2 and 2023 schemas)
# ---------------------------------------------------------------------------

def _asin_to_uuid(asin: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"amazon:{asin}"))


def item_to_row(item: dict, source: str) -> dict:
    """
    Map a raw Amazon metadata record to a products table row dict.

    Schema differences:
      v2   : feature (list), imUrl (str), tech1/tech2 (dict of tables)
      2023 : features (list), images (list[dict]), details (flat dict)
    """
    asin: str = item.get("asin") or ""
    title: str = (item.get("title") or "").strip()

    # ── Brand ────────────────────────────────────────────────────────────────
    brand = (item.get("brand") or item.get("store") or "").strip()
    if not brand:
        # v2: sometimes brand is in details
        details_brand = (item.get("details") or {}).get("Brand") or ""
        brand = details_brand.strip()

    # ── Price ────────────────────────────────────────────────────────────────
    price_raw = item.get("price")
    price: Optional[float] = None
    if price_raw not in (None, "", "None"):
        try:
            price = float(str(price_raw).replace("$", "").replace(",", "").strip())
        except ValueError:
            pass

    # ── Category / subcategory ───────────────────────────────────────────────
    categories = item.get("categories") or []
    flat_cats: list[str] = []
    for c in categories:
        if isinstance(c, list):
            flat_cats.extend(c)
        elif isinstance(c, str):
            flat_cats.append(c)
    category = flat_cats[-1] if flat_cats else ""
    subcategory = ""
    for cat in ("Gaming Laptops", "Ultrabooks", "2-in-1 Laptops", "Chromebooks"):
        if cat in flat_cats:
            subcategory = cat
            break

    # ── Image URL ────────────────────────────────────────────────────────────
    if source == "v2":
        image_url = item.get("imageURLHighRes") or item.get("imUrl") or ""
    else:
        images: list = item.get("images") or []
        image_url = ""
        if images:
            first = images[0]
            if isinstance(first, dict):
                image_url = first.get("large") or first.get("hi_res") or first.get("thumb") or ""

    # ── Attributes (RAM, storage, CPU, etc.) ────────────────────────────────
    if source == "v2":
        # v2: specs are in tech1/tech2 (list of [key, val] pairs) or flat string
        details: dict = {}
        for tech_key in ("tech1", "tech2"):
            tech_val = item.get(tech_key)
            if isinstance(tech_val, dict):
                details.update(tech_val)
            elif isinstance(tech_val, list):
                for entry in tech_val:
                    if isinstance(entry, list) and len(entry) == 2:
                        details[entry[0]] = entry[1]
        features = item.get("feature") or []
    else:
        details = item.get("details") or {}
        features = item.get("features") or []

    ram_gb = _parse_gb(
        details.get("RAM") or
        details.get("RAM Memory Installed Size") or
        details.get("Computer Memory Size") or ""
    )
    if ram_gb and ram_gb > 128:
        ram_gb = None

    storage_gb = _parse_gb(
        details.get("Hard Drive") or
        details.get("Solid State Drive Capacity") or
        details.get("Flash Memory Size") or
        details.get("Hard Drive Size") or ""
    )
    if storage_gb and storage_gb > 4096:
        storage_gb = None

    storage_type_combined = " ".join([
        details.get("Hard Drive", ""),
        details.get("Solid State Drive Capacity", ""),
        details.get("Flash Memory Size", ""),
        details.get("Hard Drive Interface", ""),
        title,
    ]).lower()
    if "nvme" in storage_type_combined or "pcie" in storage_type_combined:
        storage_type = "NVMe SSD"
    elif "ssd" in storage_type_combined or "solid state" in storage_type_combined:
        storage_type = "SSD"
    elif "hdd" in storage_type_combined or "hard disk" in storage_type_combined:
        storage_type = "HDD"
    else:
        storage_type = None

    screen_size = _parse_inches(
        details.get("Screen Size") or
        details.get("Standing screen display size") or
        details.get("Display Size") or ""
    )
    cpu = _extract_cpu(details, features) or None
    battery = _parse_hours(
        details.get("Battery Average Life") or
        details.get("Battery Life") or ""
    )
    os_val = (
        details.get("Operating System") or
        details.get("OS") or
        details.get("Platform") or ""
    )
    os_val = str(os_val)[:80] if os_val else None
    weight = _parse_weight_lbs(
        details.get("Item Weight") or
        details.get("Package Weight") or ""
    )
    color = details.get("Color") or None
    if color:
        color = str(color)[:60]

    # ── Description ──────────────────────────────────────────────────────────
    desc_raw = item.get("description") or []
    if isinstance(desc_raw, list):
        # Flatten any nested lists; coerce all elements to str
        flat_parts: list[str] = []
        for part in desc_raw:
            if isinstance(part, list):
                flat_parts.extend(str(x) for x in part)
            else:
                flat_parts.append(str(part))
        description = " ".join(flat_parts).strip()
    else:
        description = str(desc_raw).strip()
    if not description:
        description = " ".join(str(f) for f in features[:5]).strip() or None

    # ── Ratings (2023 only — v2 doesn't include them in metadata) ───────────
    avg_rating = item.get("average_rating") or item.get("rating_number")
    review_count = item.get("rating_number") or None
    if avg_rating is not None:
        try:
            avg_rating = float(avg_rating)
        except (ValueError, TypeError):
            avg_rating = None

    now = datetime.now(timezone.utc).isoformat()

    return {
        "id": _asin_to_uuid(asin),
        "asin": asin,
        "title": title or None,
        "brand": brand or None,
        "price": price,
        "category": category or None,
        "subcategory": subcategory or None,
        "image_url": image_url or None,
        "description": description,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "storage_type": storage_type,
        "cpu": cpu,
        "screen_size": screen_size,
        "battery_life_hours": battery,
        "os": os_val,
        "weight_lbs": weight,
        "color": color,
        "avg_rating": avg_rating,
        "review_count": review_count,
        "source": f"amazon_{source}",
        "imported_at": now,
    }


# ---------------------------------------------------------------------------
# Streaming HTTP reader
# ---------------------------------------------------------------------------

def _stream_jsonl_gz(url: str) -> Iterator[dict]:
    """Stream a gzip-JSONL file over HTTP, yielding parsed dicts."""
    log.info("Streaming %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "idss-dev/1.0"})
    try:
        resp = urllib.request.urlopen(req, context=_SSL, timeout=30)
    except Exception as exc:
        log.error("Failed to open %s: %s", url, exc)
        raise

    gz = gzip.GzipFile(fileobj=resp)
    try:
        for raw_line in gz:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        gz.close()
        resp.close()


# ---------------------------------------------------------------------------
# SQLite schema + writer
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS products (
    id               TEXT PRIMARY KEY,
    asin             TEXT UNIQUE NOT NULL,
    title            TEXT,
    brand            TEXT,
    price            REAL,
    category         TEXT,
    subcategory      TEXT,
    image_url        TEXT,
    description      TEXT,
    ram_gb           INTEGER,
    storage_gb       INTEGER,
    storage_type     TEXT,
    cpu              TEXT,
    screen_size      REAL,
    battery_life_hours REAL,
    os               TEXT,
    weight_lbs       REAL,
    color            TEXT,
    avg_rating       REAL,
    review_count     INTEGER,
    source           TEXT,
    imported_at      TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id      TEXT REFERENCES products(id) ON DELETE CASCADE,
    asin            TEXT NOT NULL,
    rating          INTEGER,
    review_text     TEXT,
    reviewer_name   TEXT,
    review_date     TEXT,
    verified        INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_products_brand      ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_price      ON products(price);
CREATE INDEX IF NOT EXISTS idx_products_ram        ON products(ram_gb);
CREATE INDEX IF NOT EXISTS idx_products_screen     ON products(screen_size);
CREATE INDEX IF NOT EXISTS idx_reviews_asin        ON reviews(asin);
"""

_UPSERT_SQL = """
INSERT INTO products (
    id, asin, title, brand, price, category, subcategory,
    image_url, description, ram_gb, storage_gb, storage_type, cpu,
    screen_size, battery_life_hours, os, weight_lbs, color,
    avg_rating, review_count, source, imported_at
) VALUES (
    :id, :asin, :title, :brand, :price, :category, :subcategory,
    :image_url, :description, :ram_gb, :storage_gb, :storage_type, :cpu,
    :screen_size, :battery_life_hours, :os, :weight_lbs, :color,
    :avg_rating, :review_count, :source, :imported_at
)
ON CONFLICT(asin) DO UPDATE SET
    title = excluded.title,
    brand = COALESCE(excluded.brand, brand),
    price = COALESCE(excluded.price, price),
    ram_gb = COALESCE(excluded.ram_gb, ram_gb),
    storage_gb = COALESCE(excluded.storage_gb, storage_gb),
    storage_type = COALESCE(excluded.storage_type, storage_type),
    cpu = COALESCE(excluded.cpu, cpu),
    screen_size = COALESCE(excluded.screen_size, screen_size),
    battery_life_hours = COALESCE(excluded.battery_life_hours, battery_life_hours),
    os = COALESCE(excluded.os, os),
    weight_lbs = COALESCE(excluded.weight_lbs, weight_lbs),
    avg_rating = COALESCE(excluded.avg_rating, avg_rating),
    review_count = COALESCE(excluded.review_count, review_count),
    imported_at = excluded.imported_at
"""

_REVIEW_INSERT_SQL = """
INSERT OR IGNORE INTO reviews (product_id, asin, rating, review_text, reviewer_name, review_date, verified)
VALUES (:product_id, :asin, :rating, :review_text, :reviewer_name, :review_date, :verified)
"""


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    log.info("SQLite database ready at %s", path)
    return conn


def _sanitize_row(row: dict) -> dict:
    """Coerce any list/dict values to JSON strings so SQLite doesn't choke."""
    out = {}
    for k, v in row.items():
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def upsert_batch(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(_UPSERT_SQL, [_sanitize_row(r) for r in rows])
    conn.commit()


def insert_reviews(conn: sqlite3.Connection, review_rows: list[dict]) -> None:
    conn.executemany(_REVIEW_INSERT_SQL, review_rows)
    conn.commit()


# ---------------------------------------------------------------------------
# Review importer
# ---------------------------------------------------------------------------

def load_reviews(
    source: str,
    category: str,
    known_asins: set[str],
    asin_to_id: dict[str, str],
    max_per_product: int = 10,
) -> list[dict]:
    """Stream review file, keep only reviews for known ASINs."""
    url_tmpl = DATASET_URLS[source]["reviews"]
    url = url_tmpl.format(category=category)
    rows: list[dict] = []
    counts: dict[str, int] = {}

    log.info("Streaming reviews from %s", url)
    for item in _stream_jsonl_gz(url):
        asin = item.get("asin") or ""
        if asin not in known_asins:
            continue
        if counts.get(asin, 0) >= max_per_product:
            continue

        rating = item.get("overall") or item.get("rating")
        try:
            rating = int(float(rating)) if rating is not None else None
        except (ValueError, TypeError):
            rating = None

        review_text = (item.get("reviewText") or item.get("text") or "").strip()[:2000]
        reviewer_name = (item.get("reviewerName") or item.get("user_id") or "").strip()[:100]
        review_date = item.get("reviewTime") or item.get("timestamp") or ""

        rows.append({
            "product_id": asin_to_id.get(asin),
            "asin": asin,
            "rating": rating,
            "review_text": review_text or None,
            "reviewer_name": reviewer_name or None,
            "review_date": str(review_date)[:30] if review_date else None,
            "verified": 1 if item.get("verified") else 0,
        })
        counts[asin] = counts.get(asin, 0) + 1

    log.info("Collected %d review rows for %d products", len(rows), len(counts))
    return rows


# ---------------------------------------------------------------------------
# Main import pipeline
# ---------------------------------------------------------------------------

BATCH_SIZE = 250


def run_import(
    source: str,
    category: str,
    output: Path,
    limit: Optional[int],
    scan_limit: Optional[int],
    laptop_filter: bool,
    include_reviews: bool,
    dry_run: bool,
) -> None:
    """
    limit      — max *accepted* products to write (target DB size).
    scan_limit — max records to read from stream before stopping, regardless of
                 how many were accepted. Useful to prevent scanning the entire
                 1+ GB file when you just want a quick sample.
    """
    url_tmpl = DATASET_URLS[source]["meta"]
    url = url_tmpl.format(category=category)

    conn = None if dry_run else init_db(output)

    rows: list[dict] = []
    total = 0      # accepted
    scanned = 0    # total records read from stream
    skipped = 0

    try:
        for item in _stream_jsonl_gz(url):
            if limit and total >= limit:
                break
            if scan_limit and scanned >= scan_limit:
                log.info("scan-limit %d reached (%d accepted so far)", scan_limit, total)
                break
            scanned += 1

            if laptop_filter and not is_laptop(item, source):
                skipped += 1
                continue

            row = item_to_row(item, source)
            if not row["asin"]:
                skipped += 1
                continue

            rows.append(row)
            total += 1

            if len(rows) >= BATCH_SIZE:
                if dry_run:
                    _print_sample(rows[:3])
                else:
                    upsert_batch(conn, rows)
                    log.info("Upserted %d rows (total=%d, scanned=%d, skipped=%d)",
                             len(rows), total, scanned, skipped)
                rows = []

        # flush remainder
        if rows:
            if dry_run:
                _print_sample(rows[:3])
            else:
                upsert_batch(conn, rows)

    except KeyboardInterrupt:
        log.info("Interrupted — flushing %d buffered rows", len(rows))
        if rows and not dry_run:
            upsert_batch(conn, rows)

    if dry_run:
        log.info("Dry run complete. Scanned %d records, would have written %d products.", scanned, total)
        return

    log.info("Products import complete: %d inserted/updated, %d scanned, %d skipped",
             total, scanned, skipped)

    if include_reviews and total > 0:
        # Collect ASIN→UUID mapping from DB
        cur = conn.execute("SELECT asin, id FROM products WHERE source = ?", (f"amazon_{source}",))
        asin_to_id = {row[0]: row[1] for row in cur.fetchall()}
        known_asins = set(asin_to_id.keys())

        review_rows = load_reviews(source, category, known_asins, asin_to_id)
        if review_rows:
            insert_reviews(conn, review_rows)
            log.info("Reviews import complete: %d rows", len(review_rows))

    conn.close()
    log.info("Done. Database: %s", output.resolve())
    _print_stats(output)


def _print_sample(rows: list[dict]) -> None:
    for r in rows:
        print(
            f"  {r['asin']} | {(r['title'] or '')[:55]:<55} | "
            f"${r['price'] or '?':>7} | RAM={r['ram_gb']}GB | "
            f"Screen={r['screen_size']}\" | CPU={r['cpu'] or '?'}"
        )


def _print_stats(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    n_products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    n_with_price = conn.execute("SELECT COUNT(*) FROM products WHERE price IS NOT NULL").fetchone()[0]
    n_with_ram = conn.execute("SELECT COUNT(*) FROM products WHERE ram_gb IS NOT NULL").fetchone()[0]
    n_reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    conn.close()
    print(f"\n{'─'*50}")
    print(f"  Products     : {n_products:,}")
    print(f"  With price   : {n_with_price:,}  ({100*n_with_price//max(n_products,1)}%)")
    print(f"  With RAM info: {n_with_ram:,}  ({100*n_with_ram//max(n_products,1)}%)")
    print(f"  Reviews      : {n_reviews:,}")
    print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Amazon product dataset → local SQLite DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", choices=["v2", "2023"], default="2023",
        help="Dataset version: 'v2' (2018, ~786K Electronics products) or '2023' (default, richer)",
    )
    parser.add_argument(
        "--category", default="Electronics",
        help="Amazon category name (default: Electronics). "
             "Examples: Books, Clothing_Shoes_and_Jewelry, Home_and_Kitchen",
    )
    parser.add_argument(
        "--output", default="dev_amazon.db", type=Path,
        help="Output SQLite file path (default: dev_amazon.db)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max *accepted* products to write (default: all). Stops after this many laptops.",
    )
    parser.add_argument(
        "--scan-limit", dest="scan_limit", type=int, default=None,
        help="Max records to *read* from the stream before stopping, regardless of how many "
             "were accepted. For dry-runs this defaults to 50000. Set to 0 to disable.",
    )
    parser.add_argument(
        "--no-laptop-filter", action="store_true",
        help="Import all items in the category, not just laptops",
    )
    parser.add_argument(
        "--reviews", action="store_true",
        help="Also import reviews for imported products (streams a second large file)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and print rows without writing to DB",
    )
    args = parser.parse_args()

    laptop_filter = not args.no_laptop_filter

    # For dry-runs, default to a scan-limit so we don't stream the entire file
    scan_limit = args.scan_limit
    if args.dry_run and scan_limit is None:
        scan_limit = 50_000  # scan up to 50K records; should yield ~500 laptops

    log.info(
        "Config: source=%s  category=%s  limit=%s  scan_limit=%s  laptop_filter=%s  output=%s",
        args.source, args.category, args.limit, scan_limit, laptop_filter, args.output,
    )

    run_import(
        source=args.source,
        category=args.category,
        output=args.output,
        limit=args.limit,
        scan_limit=scan_limit,
        laptop_filter=laptop_filter,
        include_reviews=args.reviews,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
