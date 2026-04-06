# N-run evaluation: time estimate (default n=50)

Both pipelines now run **n times** (default **50**) and report **averaged** metrics.

## Agent response (`evaluation/agent_response/run_eval`)

- **Per run:** 50 test cases × (1 agent conversation + 1 judge call) ≈ 50 × 15–30 s ≈ **12–25 min**.
- **50 runs (agent):** ≈ **10–21 hours**.
- **50 runs (baseline), if run separately:** same order.
- **Total (agent + baseline at n=50):** ≈ **20–40 hours** if run sequentially.

Use `--n-runs 2` or `N_RUNS=2` for a quick sanity check (~30–60 min for agent + baseline).

## Recommendations (`evaluation/recommendations/run_evaluation`)

- **Per run:** N items × 4 stages (query→UCP, query→UCP baseline, UCP→recs, baseline) ≈ 4N × 5–20 s. With N=12–50 items: **4–20 min per run**.
- **50 runs:** ≈ **3–17 hours** depending on number of evaluable items.

Use `--n-runs 2` for a quick check (~10–40 min).

## Running both in parallel

Subagents can run the two pipelines in parallel. Wall-clock time is roughly the **maximum** of the two (agent_response is usually the bottleneck).
