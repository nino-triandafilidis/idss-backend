
# IDSS Backend - Multi-Domain Interactive Decision Support System

## Quick Start (TL;DR)

**Start all local servers (backend + frontend):**

```bash
bash start_all_local.sh
```

**Run all unit tests:**

```bash
bash run_all_tests.sh
# Expected: 571 backend tests pass, 126 frontend tests pass (March 2026)
```

**Deployed URLs:**

- Frontend: <https://idss-web.vercel.app> (Vercel)
- Backend API: see Railway dashboard → your project → deployment URL

---

An LLM-driven Interactive Decision Support System that helps users find products through conversational interviews. The **Universal Agent** detects the user's domain, extracts preferences via structured LLM calls, and generates natural follow-up questions before delivering recommendations. Supports **vehicles**, **laptops**, **books**, and **24,000+ electronics** products.

## Architecture

```
                        Frontend (Port 3000)
                       Next.js Chat Interface
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│                  MCP Server (Port 8001)                   │
│                                                          │
│  POST /chat ─────► agent/                                │
│                    ├── UniversalAgent (LLM brain)         │
│                    │   ├── Domain detection               │
│                    │   ├── Criteria extraction            │
│                    │   ├── Question generation            │
│                    │   └── Post-rec refinement            │
│                    ├── chat_endpoint.py (orchestrator)    │
│                    └── interview/session_manager.py       │
│                                                          │
│  Search dispatch:                                        │
│    vehicles ──► Supabase (direct import, no HTTP)        │
│    laptops  ──► PostgreSQL                               │
│    books    ──► PostgreSQL                               │
└──────────────────────────────────────────────────────────┘
              │                         │
              ▼                         ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│        Supabase         │   │      PostgreSQL          │
│   Vehicle Data          │   │    (mcp_ecommerce)       │
│   ~147k vehicles        │   │  - Electronics (~21k)    │
│   Embeddings + Phrases  │   │  - Books (~66)           │
└─────────────────────────┘   └─────────────────────────┘
```

**Key design:** There is no separate IDSS API server. Vehicle search functions (`idss.recommendation.*`) are imported directly into the MCP server process. Only **one server** (port 8001) is needed.

## Project Structure

```
idss-backend/
├── agent/                           # Agent brain (independent of server)
│   ├── __init__.py                  # Public API re-exports
│   ├── universal_agent.py           # LLM-driven pipeline (domain → extract → question → search)
│   ├── domain_registry.py           # Domain schemas (slots, priorities, allowed values)
│   ├── prompts.py                   # All LLM prompt templates (tune without touching logic)
│   ├── chat_endpoint.py             # /chat orchestrator + search dispatchers
│   └── interview/
│       └── session_manager.py       # Session state + Redis/Neo4j persistence
│
├── mcp-server/                      # HTTP server + tools
│   ├── app/
│   │   ├── main.py                  # FastAPI app (port 8001)
│   │   ├── endpoints.py             # MCP tool-call endpoints
│   │   ├── tools/vehicle_search.py  # IDSS vehicle search wrapper (direct import)
│   │   ├── formatters.py            # Product formatting for frontend
│   │   ├── research_compare.py      # Post-rec research/compare handlers
│   │   ├── database.py              # PostgreSQL connection
│   │   ├── models.py                # SQLAlchemy models
│   │   └── ...                      # Cache, metrics, UCP, etc.
│   ├── scripts/
│   │   ├── seed_diverse.sql         # Creates tables + seed products
│   │   ├── seed_laptops_expanded.sql # Additional laptop data
│   │   ├── seed_books_expanded.sql  # Additional book data
│   │   └── merge_supabase_data.py   # Import ~24k products from Supabase
│   └── tests/
│
├── idss/                            # IDSS Vehicle Recommendation Engine
│   ├── recommendation/              # Embedding similarity, coverage-risk
│   ├── diversification/             # Entropy bucketing
│   └── core/                        # Controller
│
├── config/default.yaml              # IDSS recommendation config
├── requirements.txt
└── .env                             # Environment variables
```

## Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime |
| PostgreSQL | 14+ | Product database (laptops, books, electronics) |
| OpenAI API key | - | LLM calls (domain detection, extraction, question generation) |

**Optional:**

| Software | Purpose |
|----------|---------|
| Redis 6+ / Upstash | Session caching (falls back to in-memory) |

## Quick Start

### 1. Clone and Install

```bash
git clone <repo-url> idss-backend
cd idss-backend

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the **project root** (not inside `mcp-server/`):

```bash
# Required
OPENAI_API_KEY="sk-REPLACE-ME"
DATABASE_URL="postgresql://YOUR_USERNAME@localhost:5432/mcp_ecommerce"

# Supabase (vehicle search)
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your-supabase-anon-key"

# LLM model configuration
OPENAI_MODEL="gpt-5-nano"
# OPENAI_REASONING_EFFORT="low"
```

**Finding your PostgreSQL username:** On Mac, it's usually your system username. Check with:
```bash
whoami
# or
psql -c "\du"
```

**Using Supabase (recommended):** Replace `DATABASE_URL` with your Supabase direct connection string from Project Settings > Database > Connection string > URI.

### 3. Setup Database

**Option A: Use Supabase cloud (recommended — data already loaded)**

If your Supabase already has the 24,150 products, just set `DATABASE_URL` in `.env` and skip to step 4.

**Option B: Seed a local PostgreSQL**

```bash
# Create the database
createdb mcp_ecommerce

# Create tables + seed initial products
cd mcp-server
psql -d mcp_ecommerce -f scripts/seed_diverse.sql
psql -d mcp_ecommerce -f scripts/seed_laptops_expanded.sql
psql -d mcp_ecommerce -f scripts/seed_books_expanded.sql
```

#### Import full product catalog from Supabase (recommended)

This imports ~24,000 real products (laptops, monitors, GPUs, keyboards, etc.) from a shared Supabase database into your local PostgreSQL. It's a one-time operation that takes ~30-60 seconds:

```bash
python scripts/merge_supabase_data.py --skip-redis --skip-kg
```

The script is idempotent — running it again skips already-imported products.

#### Verify

```bash
psql -d mcp_ecommerce -c "SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY count DESC;"
```

With full import you should see:
```
  category   | count
-------------+-------
 Electronics | ~21000+
 Books       |    66
```

Without Supabase import (seed data only):
```
  category   | count
-------------+-------
 Electronics |    37
 Books       |    50
```

### 4. Start the Server

```bash
# Terminal 1: IDSS server (port 8000) — frontend connects here
source .venv/bin/activate
uvicorn idss.api.server:app --reload --port 8000

# Terminal 2: MCP server (port 8001) — optional, same agent
cd mcp-server
uvicorn app.main:app --reload --port 8001
```

First startup preloads vehicle embedding models (~60-120 seconds).

### 5. Verify

```bash
# Health check
curl http://localhost:8000/health

# Test laptop search
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want a gaming laptop"}'

# Test book search
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "looking for a mystery novel"}'

# Test vehicle flow
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need an SUV under 30k"}'
```

## How It Works

### The Universal Agent Pipeline

Every `/chat` message goes through this flow:

1. **Domain Detection** — Classifies into `laptops`, `books`, `vehicles`, or `unknown`
2. **Criteria Extraction** — Extracts slot values using domain-specific schemas
3. **Interview Decision** — Ask another question or show results?
4. **Question Generation** — Natural follow-up with quick replies
5. **Search Dispatch** — Supabase PostgreSQL with progressive filter relaxation
6. **Recommendation Explanation** — Conversational message highlighting standout products
7. **Post-Rec Refinement** — Filter changes, comparisons, research, checkout

### Chat Request

```json
{
  "message": "I want a laptop for gaming",
  "session_id": "optional-session-id",
  "k": 3,
  "user_actions": [
    {"type": "favorite", "product_id": "uuid-here"},
    {"type": "unfavorite", "product_id": "uuid-here"}
  ]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | required | User message |
| `session_id` | string | auto-generated | Session ID for multi-turn conversations |
| `k` | int | 3 | Max interview questions |
| `user_actions` | array | [] | Favorite/unfavorite actions to sync |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Main conversation endpoint |
| `/health` | GET | Health check |
| `/ucp/checkout-sessions` | POST | Create checkout session |
| `/ucp/checkout-sessions/{id}/complete` | POST | Complete checkout |
| `/ucp/checkout-sessions/{id}/cancel` | POST | Cancel checkout |
| `/session/{id}` | GET | Get session state |
| `/session/reset` | POST | Reset/create session |

## Troubleshooting

**"role does not exist"** — Wrong PostgreSQL username in `DATABASE_URL`. Find yours with `whoami` (Mac) or `psql -c "\du"`.

**"Column expression expected, got Price"** — The old `Price` model is now a stub. If a script uses `db.query(Price)`, update it to use `product.price_value` directly.

**"Values of type UUID are not supported"** — Neo4j doesn't accept Python `uuid.UUID` objects. Convert with `str(product.product_id)` before passing to Cypher queries.

**"column does not exist" (e.g. `kg_features`, `product_type`)** — Your table schema is outdated. Drop and recreate: `psql -d mcp_ecommerce -f scripts/seed_diverse.sql` (warning: this drops all data, re-run seed scripts and Supabase import after).

**No products returned** — Check `psql -d mcp_ecommerce -c "SELECT COUNT(*) FROM products;"`. If 0, run the seed scripts (step 3c) and optionally the Supabase import (step 3d).

**Supabase import fails** — The Supabase import connects to a shared remote database. If it fails with a connection error, the remote may be unavailable. The seed data (37 laptops + 50 books) is sufficient to run the system without the Supabase import.

**Redis connection errors** — Redis is optional. The system falls back to in-memory sessions. The warning is harmless.

**UUID vs String IDs** — Supabase uses UUID for `id`. Old code that used string IDs like `"laptop-1"` needs updating to use UUIDs.

**Neo4j port confusion** — Docker maps `7475→7474` and `7688→7687`. Use `localhost:7475` for the browser and `bolt://localhost:7688` for the driver. Do NOT use the default ports (7474/7687) — those may be a separate local Neo4j install.

**Vehicle search returns no results** — Check that `SUPABASE_URL` and `SUPABASE_KEY` are set correctly in your `.env` file.

**Server crashes on startup (ImportError)** — Make sure you're running from the repo root with `--app-dir mcp-server`. The agent package must be importable from the repo root.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for agent LLM calls (do not commit real keys) |
| `DATABASE_URL` | Yes | - | PostgreSQL connection string (e.g. `postgresql://user@localhost:5432/mcp_ecommerce`) |
| `SUPABASE_URL` | Yes | - | Supabase project URL for vehicle search |
| `SUPABASE_KEY` | Yes | - | Supabase anon key for vehicle search |
| `OPENAI_MODEL` | No | gpt-5-nano | Model for all agent LLM calls |
| `OPENAI_REASONING_EFFORT` | No | low | Reasoning effort: low, medium, high |
| `LOG_LEVEL` | No | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `REDIS_HOST` | No | localhost | Redis host for session caching |
| `REDIS_PORT` | No | 6379 | Redis port |
| `NEO4J_URI` | No | - | Neo4j connection for knowledge graph |
