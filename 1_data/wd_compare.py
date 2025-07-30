#!/usr/bin/env python3
"""
Compute Wasserstein distance (WD) between two sets of answer-distributions that
share Pew-style question metadata.

Usage:
    python wd_compare.py <second_json_path> [-o wd_per_questions.json]

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

REFUSAL_CODES = {"99"}            # drop & renormalise
NEUTRAL_PAT = re.compile(
    r"(not\s*sure|neither|never\s*heard|don['’]t\s*know|neutral)", re.I
)


def build_rank_map(answers: dict[str, str]) -> dict[str, float]:
    """
    Map answer-codes ➜ numeric ranks.

    * Ordinal options → 1 .. k   (in ascending code order)
    * Any answer whose text matches NEUTRAL_PAT → midpoint (1+k)/2
    * Codes in REFUSAL_CODES are ignored here.
    """
    # sort codes numerically (as strings → ints → back to str)
    sorted_codes = sorted(
        (c for c in answers if c not in REFUSAL_CODES),
        key=lambda x: int(x),
    )

    # label codes as neutral / ordinal
    neutral, ordinal = [], []
    for code in sorted_codes:
        if NEUTRAL_PAT.search(answers[code]):
            neutral.append(code)
        else:
            ordinal.append(code)

    # assign ordinal ranks
    rank_map: dict[str, float] = {
        code: float(idx) for idx, code in enumerate(ordinal, start=1)
    }

    # neutral codes get the midpoint of the ordinal span
    if ordinal:
        mid = (1 + len(ordinal)) / 2.0
    else:                       # degenerate case: only neutrals
        mid = 1.0
    for code in neutral:
        rank_map[code] = mid

    return rank_map


def normalise_dist(dist: dict[str, float]) -> dict[str, float]:
    """Remove refusal code(s) and renormalise to sum to 1."""
    cleaned = {k: v for k, v in dist.items() if k not in REFUSAL_CODES}
    Z = sum(cleaned.values())
    if Z == 0:
        raise ValueError("All probabilities were refusal codes.")
    return {k: v / Z for k, v in cleaned.items()}


def wd_between(
    ranks: dict[str, float],
    dist_a: dict[str, float],
    dist_b: dict[str, float],
) -> float:
    """
    Wasserstein distance between two discrete distributions over the *same* set
    of answer codes.
    """
    # support points (x-axis) and aligned weight vectors
    support = np.array([ranks[c] for c in ranks])
    w_a = np.array([dist_a.get(c, 0.0) for c in ranks])
    w_b = np.array([dist_b.get(c, 0.0) for c in ranks])

    return wasserstein_distance(
        support, support, u_weights=w_a, v_weights=w_b
    )


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

    ref_path = Path("converted_questions.json")
    other_path = Path(args.other_json)

    if not ref_path.exists():
        sys.exit(f"Cannot find reference file {ref_path!s}")
    if not other_path.exists():
        sys.exit(f"Cannot find second file {other_path!s}")

    ref_qs = {q["id"]: q for q in json.loads(ref_path.read_text())}
    oth_qs = {q["id"]: q for q in json.loads(other_path.read_text())}

    # Sanity-check: identical question IDs
    ids_common = ref_qs.keys() & oth_qs.keys()
    if not ids_common:
        sys.exit("No overlapping question IDs between the two files.")
    if ref_qs.keys() != oth_qs.keys():
        missing = ref_qs.keys() ^ oth_qs.keys()
        print(f"Warning: unmatched question IDs will be skipped: {sorted(missing)}",
              file=sys.stderr)

    results = []
    for qid in sorted(ids_common):
        ref_q = ref_qs[qid]
        oth_q = oth_qs[qid]

        # Defensive: make sure survey metadata matches
        for key in ("institute", "survey", "survey_qid", "date", "file"):
            if ref_q.get(key) != oth_q.get(key):
                print(f"Warning: field {key!s} differs for id {qid}", file=sys.stderr)

        rank_map = build_rank_map(ref_q["answers"])
        ref_dist = normalise_dist(ref_q["distribution"])
        oth_dist = normalise_dist(oth_q["distribution"])

        wd = wd_between(rank_map, ref_dist, oth_dist)

        results.append({"id": qid, "WD_without_refusal": round(float(wd), 6)})

    # Write output
    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} distances to {out_path!s}")


if __name__ == "__main__":
    main()
