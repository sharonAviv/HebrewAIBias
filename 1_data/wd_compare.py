#!/usr/bin/env python3
"""
Compute Wasserstein distance (WD) between two Pew-style question sets.
Adds refusal-rate diagnostics and overall averages.

Usage:
    python wd_compare.py <second_json_path> [-o wd_per_questions.json]

Version: 2

• The script always loads the reference file  ./converted_questions.json
• <second_json_path> must follow the same schema (id/question/answers/distribution/…)
• Output is a JSON list  [{"id": …, "WD_without_refusal": …}, …]
"""

import argparse, json, re, sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


# --------------------------------------------------------------------------- #
# 1.  Utility helpers
# --------------------------------------------------------------------------- #

REFUSAL_CODES = {"99"}
NEUTRAL_PAT = re.compile(
    r"(not\s*sure|neither|never\s*heard|don['’]t\s*know|neutral)", re.I
)


def build_rank_map(answers: dict[str, str]) -> dict[str, float]:
    sorted_codes = sorted(
        (c for c in answers if c not in REFUSAL_CODES), key=lambda x: int(x)
    )
    neutral, ordinal = [], []
    for code in sorted_codes:
        (neutral if NEUTRAL_PAT.search(answers[code]) else ordinal).append(code)

    rank_map = {c: float(i) for i, c in enumerate(ordinal, start=1)}
    mid = (1 + len(ordinal)) / 2.0 if ordinal else 1.0
    rank_map.update({c: mid for c in neutral})
    return rank_map


def normalise_dist(dist: dict[str, float]) -> dict[str, float]:
    cleaned = {k: v for k, v in dist.items() if k not in REFUSAL_CODES}
    Z = sum(cleaned.values())
    if Z == 0:
        raise ValueError("All probabilities were refusal codes.")
    return {k: v / Z for k, v in cleaned.items()}


def wd_between(ranks, dist_a, dist_b) -> float:
    support = np.array([ranks[c] for c in ranks])
    w_a = np.array([dist_a.get(c, 0.0) for c in ranks])
    w_b = np.array([dist_b.get(c, 0.0) for c in ranks])
    return wasserstein_distance(support, support, u_weights=w_a, v_weights=w_b)


def refusal_mass(dist: dict[str, float]) -> float:
    """Return probability mass assigned to refusal codes (e.g. '99')."""
    return sum(dist.get(c, 0.0) for c in REFUSAL_CODES)


# --------------------------------------------------------------------------- #
# 2.  Main driver
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("other_json", help="path to the second question set")
    p.add_argument(
        "-o", "--output",
        default="wd_per_questions.json",
        help="output file (default: wd_per_questions.json)",
    )
    args = p.parse_args()

    ref_path   = Path("converted_questions.json")
    other_path = Path(args.other_json)

    for path in (ref_path, other_path):
        if not path.exists():
            sys.exit(f"Cannot find file {path!s}")

    ref_qs = {q["id"]: q for q in json.loads(ref_path.read_text())}
    oth_qs = {q["id"]: q for q in json.loads(other_path.read_text())}

    ids_common = ref_qs.keys() & oth_qs.keys()
    if not ids_common:
        sys.exit("No overlapping question IDs between the two files.")

    results, wd_sum, ref_refuse_sum, cmp_refuse_sum = [], 0.0, 0.0, 0.0

    for qid in sorted(ids_common):
        ref_q, oth_q = ref_qs[qid], oth_qs[qid]

        rank_map  = build_rank_map(ref_q["answers"])
        ref_dist  = normalise_dist(ref_q["distribution"])
        oth_dist  = normalise_dist(oth_q["distribution"])
        wd        = wd_between(rank_map, ref_dist, oth_dist)

        ref_refuse = refusal_mass(ref_q["distribution"])
        cmp_refuse = refusal_mass(oth_q["distribution"])

        wd_sum           += wd
        ref_refuse_sum   += ref_refuse
        cmp_refuse_sum   += cmp_refuse

        results.append({
            "id": qid,
            "WD_without_refusal": round(wd, 6),
            "Origin_refusal_rate":  round(ref_refuse, 6),
            "Compare_refusal_rate": round(cmp_refuse, 6),
        })

    n = len(results)
    results.append({
        "Average_WD_without_refusal": round(wd_sum / n, 6),
        "Average_refusal_origin":     round(ref_refuse_sum / n, 6),
        "Average_refusal_compare":    round(cmp_refuse_sum / n, 6),
    })

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {n} distances + summary to {out_path!s}")


if __name__ == "__main__":
    main()
