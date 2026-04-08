#!/usr/bin/env python3
"""
run_ablation.py — Ablation table generator for IDSS G-Eval results
===================================================================
Compares two saved G-Eval JSON files (full system vs no-KG ablation)
and prints a markdown ablation table showing KG-lift per group and
overall stats.

Usage
-----
# Generate ablation table from two pre-run result files:
    python scripts/run_ablation.py \\
        --full   scripts/geval_results_v21_full.json \\
        --no-kg  scripts/geval_results_v21_nokg.json \\
        --out    scripts/ablation_table_v21.md

# Run both conditions automatically (requires server at --url):
# Step 1: ensure backend is running WITH KG (NEO4J_PASSWORD set)
# Step 2: ensure backend is running WITHOUT KG (unset NEO4J_PASSWORD, restart)
#   --auto flag runs both sequentially and saves the table.
    python scripts/run_ablation.py \\
        --auto \\
        --url http://localhost:8000 \\
        --out  scripts/ablation_table_v21.md

KG-lift definition
------------------
  kg_lift = full_avg_score − no_kg_avg_score

For the paper:
  Positive lift means KG contributes to recommendation quality.
  Report per-group lift to highlight where KG helps most (expert queries,
  catalog_exploration, comparison_direct) vs where it's neutral (follow_up_qa).
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass


# ============================================================================
# Loading helpers
# ============================================================================

def load_results(path: str) -> Dict:
    """Load a saved G-Eval JSON file and return the full dict."""
    with open(path) as f:
        return json.load(f)


def per_group_stats(results: List[Dict]) -> Dict[str, Dict]:
    """Compute per-group avg_score, pass_pct, type_acc from a results list."""
    PASS_THRESHOLD = 0.5
    groups: Dict[str, List[Dict]] = {}
    for r in results:
        g = r.get("group", "unknown")
        groups.setdefault(g, []).append(r)

    out: Dict[str, Dict] = {}
    for g, rs in groups.items():
        n = len(rs)
        avg = sum(r["score"] for r in rs) / n
        pct = 100.0 * sum(1 for r in rs if r["score"] >= PASS_THRESHOLD) / n
        tacc = 100.0 * sum(1 for r in rs if r.get("type_score", 0) == 1.0) / n
        out[g] = {"n": n, "avg": round(avg, 4), "pass_pct": round(pct, 1), "type_acc": round(tacc, 1)}
    return out


def overall_stats(results: List[Dict]) -> Dict:
    PASS_THRESHOLD = 0.5
    n = len(results)
    if n == 0:
        return {"n": 0, "avg": 0.0, "pass_pct": 0.0, "type_acc": 0.0}
    avg = sum(r["score"] for r in results) / n
    pct = 100.0 * sum(1 for r in results if r["score"] >= PASS_THRESHOLD) / n
    tacc = 100.0 * sum(1 for r in results if r.get("type_score", 0) == 1.0) / n
    return {"n": n, "avg": round(avg, 4), "pass_pct": round(pct, 1), "type_acc": round(tacc, 1)}


# ============================================================================
# Table rendering
# ============================================================================

def render_markdown_table(
    full_data: Dict,
    nokg_data: Dict,
    full_path: str,
    nokg_path: str,
) -> str:
    full_results = full_data.get("results", [])
    nokg_results = nokg_data.get("results", [])

    full_grp = per_group_stats(full_results)
    nokg_grp = per_group_stats(nokg_results)
    full_all = overall_stats(full_results)
    nokg_all = overall_stats(nokg_results)

    all_groups = sorted(set(list(full_grp.keys()) + list(nokg_grp.keys())))

    lines: List[str] = []
    lines.append("# IDSS Ablation Study — KG vs No-KG")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- Full system: `{os.path.basename(full_path)}` (KG enabled)")
    lines.append(f"- No-KG:       `{os.path.basename(nokg_path)}` (KG disabled)")
    lines.append(f"- Pass threshold: 0.5")
    lines.append("")

    # --- Overall summary ---
    lift_avg = round(full_all["avg"] - nokg_all["avg"], 4)
    lift_pct = round(full_all["pass_pct"] - nokg_all["pass_pct"], 1)
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Condition | N | Avg Score | Pass% | TypeAcc% | KG-Lift (avg) |")
    lines.append("|-----------|---|-----------|-------|----------|---------------|")
    lines.append(
        f"| Full system (KG on)  | {full_all['n']} | {full_all['avg']:.4f} "
        f"| {full_all['pass_pct']:.1f}% | {full_all['type_acc']:.1f}% | — |"
    )
    lines.append(
        f"| No-KG ablation       | {nokg_all['n']} | {nokg_all['avg']:.4f} "
        f"| {nokg_all['pass_pct']:.1f}% | {nokg_all['type_acc']:.1f}% | — |"
    )
    lift_dir = "+" if lift_avg >= 0 else ""
    lines.append(
        f"| **KG Lift**          | — | **{lift_dir}{lift_avg:.4f}** "
        f"| **{'+' if lift_pct >= 0 else ''}{lift_pct:.1f}pp** | — | — |"
    )
    lines.append("")

    # --- Per-group breakdown ---
    lines.append("## Per-Group Breakdown")
    lines.append("")
    lines.append(
        "| Group | N | Full Avg | Full Pass% | No-KG Avg | No-KG Pass% "
        "| KG-Lift (avg) | KG-Lift (pass%) |"
    )
    lines.append(
        "|-------|---|----------|------------|-----------|-------------|"
        "---------------|-----------------|"
    )

    for g in all_groups:
        fs = full_grp.get(g)
        ns = nokg_grp.get(g)
        if fs is None:
            fs = {"n": 0, "avg": 0.0, "pass_pct": 0.0, "type_acc": 0.0}
        if ns is None:
            ns = {"n": 0, "avg": 0.0, "pass_pct": 0.0, "type_acc": 0.0}
        n = fs["n"] or ns["n"]
        la = round(fs["avg"] - ns["avg"], 4)
        lp = round(fs["pass_pct"] - ns["pass_pct"], 1)
        la_str = f"{'+' if la >= 0 else ''}{la:.4f}"
        lp_str = f"{'+' if lp >= 0 else ''}{lp:.1f}pp"
        lines.append(
            f"| {g:<28} | {n:>3} | {fs['avg']:.4f}   | {fs['pass_pct']:>5.1f}%     "
            f"| {ns['avg']:.4f}    | {ns['pass_pct']:>5.1f}%       "
            f"| {la_str:<14} | {lp_str:<16} |"
        )

    lines.append("")

    # --- Interpretation ---
    lines.append("## Interpretation Notes")
    lines.append("")
    if lift_avg >= 0.02:
        lines.append(
            f"- KG contributes **+{lift_avg:.4f}** avg score across all queries — "
            f"a meaningful positive contribution."
        )
    elif lift_avg > -0.02:
        lines.append(
            f"- KG lift is near-zero (+{lift_avg:.4f}) — KG has neutral effect at this scale. "
            f"Check per-group for selective contribution."
        )
    else:
        lines.append(
            f"- KG shows slight negative lift ({lift_avg:.4f}). "
            f"Consider whether KG is properly configured or if candidates are degrading quality."
        )

    # Highlight top and bottom KG-lift groups
    lift_by_group = []
    for g in all_groups:
        fs = full_grp.get(g, {"avg": 0.0})
        ns = nokg_grp.get(g, {"avg": 0.0})
        la = round(fs["avg"] - ns["avg"], 4)
        lift_by_group.append((g, la))
    lift_by_group.sort(key=lambda x: x[1], reverse=True)

    if lift_by_group:
        top = lift_by_group[:3]
        bot = lift_by_group[-3:]
        lines.append(
            f"- **Highest KG-lift groups**: "
            + ", ".join(f"{g} ({'+' if v >= 0 else ''}{v:.4f})" for g, v in top)
        )
        lines.append(
            f"- **Lowest KG-lift groups**: "
            + ", ".join(f"{g} ({'+' if v >= 0 else ''}{v:.4f})" for g, v in reversed(bot))
        )

    lines.append("")
    lines.append("> G-Eval nondeterminism: ~±0.02-0.03 variance between identical runs.")
    lines.append("> Treat differences < 0.02 as within noise; focus on groups with |lift| > 0.05.")
    lines.append("")

    return "\n".join(lines)


def render_console_table(
    full_data: Dict,
    nokg_data: Dict,
) -> None:
    """Print a compact ASCII table to stdout."""
    full_results = full_data.get("results", [])
    nokg_results = nokg_data.get("results", [])

    full_grp = per_group_stats(full_results)
    nokg_grp = per_group_stats(nokg_results)
    full_all = overall_stats(full_results)
    nokg_all = overall_stats(nokg_results)
    all_groups = sorted(set(list(full_grp.keys()) + list(nokg_grp.keys())))

    BOLD = "\033[1m"
    GRN  = "\033[32m"
    RED  = "\033[31m"
    RST  = "\033[0m"

    def lift_color(v: float) -> str:
        c = GRN if v >= 0 else RED
        return f"{c}{'+' if v >= 0 else ''}{v:.4f}{RST}"

    print(f"\n{'='*78}")
    print(f"  {BOLD}IDSS Ablation Study — KG vs No-KG{RST}")
    print(f"{'='*78}\n")

    # Overall
    la = round(full_all["avg"] - nokg_all["avg"], 4)
    lp = round(full_all["pass_pct"] - nokg_all["pass_pct"], 1)
    print(f"  {'Condition':<28} {'N':>4}  {'Avg':>7}  {'Pass%':>7}  {'TypeAcc%':>9}")
    print(f"  {'─'*60}")
    print(f"  {'Full system (KG on)':<28} {full_all['n']:>4}  {full_all['avg']:>7.4f}  "
          f"{full_all['pass_pct']:>6.1f}%  {full_all['type_acc']:>8.1f}%")
    print(f"  {'No-KG ablation':<28} {nokg_all['n']:>4}  {nokg_all['avg']:>7.4f}  "
          f"{nokg_all['pass_pct']:>6.1f}%  {nokg_all['type_acc']:>8.1f}%")
    print(f"  {'─'*60}")
    lp_c = GRN if lp >= 0 else RED
    print(f"  {'KG-Lift':<28}       {lift_color(la):>7}  "
          f"  {lp_c}{'+' if lp >= 0 else ''}{lp:.1f}pp{RST}")
    print()

    # Per-group
    print(f"  Per-group breakdown:")
    print(f"  {'Group':<28} {'Full':>7}  {'NoKG':>7}  {'Lift':>8}  {'FPass%':>7}  {'NPass%':>7}")
    print(f"  {'─'*72}")
    for g in all_groups:
        fs = full_grp.get(g, {"n": 0, "avg": 0.0, "pass_pct": 0.0})
        ns = nokg_grp.get(g, {"n": 0, "avg": 0.0, "pass_pct": 0.0})
        la_g = round(fs["avg"] - ns["avg"], 4)
        n = fs.get("n") or ns.get("n", 0)
        print(
            f"  {g:<28} {fs['avg']:>7.4f}  {ns['avg']:>7.4f}  {lift_color(la_g):>8}  "
            f"{fs['pass_pct']:>6.1f}%  {ns['pass_pct']:>6.1f}%   (n={n})"
        )
    print()


# ============================================================================
# Auto-run mode: invoke run_geval.py as subprocess
# ============================================================================

def run_geval_subprocess(url: str, save_path: str, no_kg: bool = False) -> bool:
    """
    Run run_geval.py as a subprocess and save results.
    Returns True on success.
    """
    script = os.path.join(os.path.dirname(__file__), "run_geval.py")
    cmd = [sys.executable, script, "--url", url, "--save", save_path]
    if no_kg:
        cmd.append("--no-kg")

    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation table: full system vs no-KG G-Eval comparison"
    )
    parser.add_argument(
        "--full", metavar="FILE",
        help="Path to pre-run G-Eval results JSON (full system, KG enabled)",
    )
    parser.add_argument(
        "--no-kg", metavar="FILE", dest="nokg",
        help="Path to pre-run G-Eval results JSON (no-KG ablation)",
    )
    parser.add_argument(
        "--out", metavar="FILE",
        help="Save markdown ablation table to this path",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help=(
            "Auto-run both conditions via run_geval.py. "
            "Requires server at --url (full system first, then no-KG). "
            "Note: you must restart the backend WITHOUT NEO4J_PASSWORD between runs."
        ),
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Agent base URL (only used with --auto, default: http://localhost:8000)",
    )
    args = parser.parse_args()

    if args.auto:
        # Auto-run mode: run full system first, then no-KG
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = args.full or f"scripts/ablation_full_{ts}.json"
        nokg_path = args.nokg or f"scripts/ablation_nokg_{ts}.json"

        print("\n" + "="*70)
        print("  ABLATION AUTO-RUN")
        print("="*70)
        print(f"\n  Step 1/2: Full system evaluation (KG enabled)")
        print(f"  Make sure backend is running WITH NEO4J_PASSWORD set.")
        input("  Press Enter to start... ")
        if not run_geval_subprocess(args.url, full_path, no_kg=False):
            print("  ERROR: Full-system eval failed.")
            sys.exit(1)

        print(f"\n  Step 2/2: No-KG evaluation")
        print(f"  Restart backend WITHOUT NEO4J_PASSWORD (unset it), then press Enter.")
        input("  Press Enter to start... ")
        if not run_geval_subprocess(args.url, nokg_path, no_kg=True):
            print("  ERROR: No-KG eval failed.")
            sys.exit(1)
    else:
        if not args.full or not args.nokg:
            parser.error(
                "Provide --full and --no-kg result files, or use --auto to run both.\n"
                "Example:\n"
                "  python scripts/run_ablation.py \\\n"
                "      --full scripts/geval_results_v21_full.json \\\n"
                "      --no-kg scripts/geval_results_v21_nokg.json"
            )
        full_path = args.full
        nokg_path = args.nokg

    # Load and compare
    print(f"\n  Loading: {full_path}")
    full_data = load_results(full_path)
    print(f"  Loading: {nokg_path}")
    nokg_data = load_results(nokg_path)

    # Console output
    render_console_table(full_data, nokg_data)

    # Markdown output
    md = render_markdown_table(full_data, nokg_data, full_path, nokg_path)

    out_path = args.out
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"scripts/ablation_table_{ts}.md"

    with open(out_path, "w") as f:
        f.write(md)
    print(f"  Markdown ablation table saved to: {out_path}\n")


if __name__ == "__main__":
    main()
