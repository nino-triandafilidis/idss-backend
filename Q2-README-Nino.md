# Q2 — Intent Recognition and Orchestration - Nino

> **Depends on:** PR #21 (`q1-session-memory-nino`) — merge that first.

## What this PR does

Improves intent recognition for 3 existing intents where natural phrasings were missed at Layer 1 (keyword fast-path), plus refactors duplicated filter-merge logic.

| Intent | Fix | Details |
|--------|-----|---------|
| `pros_cons` | +9 keywords | Synonym pairs ("strengths and weaknesses", "advantages and disadvantages") + informal requests ("break it down for me") |
| `add_to_cart` | New regex | `_CASUAL_PURCHASE_RE` catches "I'll take it", "let me get the second one" — no cart-word required |
| `best_value` | +12 keywords | Value idioms ("bang for the buck", "best deal") + recommendation asks ("which do you recommend?") |

## Reproducing results

### Prerequisites

1. **Python 3.11+** and a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Supabase credentials** — copy `.env.example` to `.env` and fill in:
   - `SUPABASE_URL` and `SUPABASE_KEY` (from Supabase dashboard → Settings → API)
   - `OPENAI_API_KEY`
   - `DATABASE_URL` (PostgreSQL connection string)

   If setting up from scratch, see PR #21 README for full Supabase setup and seed instructions.

3. **Seed the database** (if not already done via Q1):
   ```bash
   python mcp-server/scripts/seed_supabase_local.py
   ```

### Running tests

```bash
# Activate venv
source .venv/bin/activate

# Run agent tests (includes 36 Q2 intent tests)
PYTHONPATH=./mcp-server pytest agent/tests/ -v --ignore=agent/tests/test_downgrade_scenarios.py

# Run MCP tests
PYTHONPATH=./mcp-server pytest mcp-server/tests/test_complex_query_routing.py mcp-server/tests/test_conversation_controller.py -v
```

### Running the app locally

```bash
bash start_all_local.sh
# Backend: http://localhost:8001
# Frontend: http://localhost:3000 (requires ../idss-web)
```

## Before / after evidence

### Test output

**Before (main branch):** 77 agent tests, 0 intent recognition tests.

**After (this branch):** 113 agent tests (+36 new), all passing.

```
agent/tests/test_intent_recognition_q2.py::TestProsConsExpanded       — 10 tests PASSED
agent/tests/test_intent_recognition_q2.py::TestCasualPurchaseIntent   — 11 tests PASSED
agent/tests/test_intent_recognition_q2.py::TestBestValueExpanded      — 15 tests PASSED

======================== 113 passed in 101.64s ========================
```

MCP tests: 23/23 passed, 0 regressions.

### Intent coverage: before vs. after

| Phrase | Before (main) | After (this branch) |
|--------|---------------|---------------------|
| "What are the strengths and weaknesses?" | Falls to L2 LLM → misrouted to `compare` | Caught at L1 → `pros_cons` |
| "Advantages and disadvantages?" | Falls to L2 LLM → misrouted to `compare` | Caught at L1 → `pros_cons` |
| "Break it down for me" | Falls to L2 LLM → `targeted_qa` or `other` | Caught at L1 → `pros_cons` |
| "I'll take it" | Falls to L2 → L3 `action` (no product extraction) | Caught at L1 → `add_to_cart` |
| "Let me get the second one" | Falls to L2 → L3 `action` (no ordinal logic) | Caught at L1 → `add_to_cart` |
| "Give me the third" | Falls to L2 → L3 `action` | Caught at L1 → `add_to_cart` |
| "What's the best deal?" | Falls to L2 → `targeted_qa` (wrong format) | Caught at L1 → `best_value` |
| "Which do you recommend?" | Falls to L2 → `targeted_qa` | Caught at L1 → `best_value` |
| "Best bang for the buck?" | Falls to L2 → `targeted_qa` | Caught at L1 → `best_value` |

All "before" cases required an LLM call and risked misclassification. All "after" cases are resolved at L1 with zero latency and deterministic routing.

## Files changed

- `agent/chat_endpoint.py` — keyword expansions for `pros_cons` (+9), `best_value` (+12), casual purchase regex
- `agent/universal_agent.py` — extracted `_normalize_and_merge_criteria()` (eliminates 3x duplicated filter-merge logic)
- `agent/tests/test_intent_recognition_q2.py` — 36 parametrized test cases
