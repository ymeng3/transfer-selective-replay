"""Aggregate result JSONs into a mean +- std summary table.

Usage:
    python eval/aggregate_results.py results/*.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict


def main():
    by_method = defaultdict(list)
    for path in sys.argv[1:]:
        r = json.load(open(path))
        by_method[r["method"]].append(r)
    print(f"{'method':<8} {'n':>2}  {'Overall':>14}  {'Plas':>14}  {'BWT':>14}")
    for method, rs in sorted(by_method.items()):
        row = [f"{method:<8} {len(rs):>2}"]
        for key in ("overall", "plas", "bwt"):
            vals = [r[key] for r in rs]
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            row.append(f"{mean:+.3f} ± {sd:.3f}" if key == "bwt"
                       else f"{mean:.3f} ± {sd:.3f}")
        print("  ".join(row))


if __name__ == "__main__":
    main()
