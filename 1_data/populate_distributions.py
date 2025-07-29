#!/usr/bin/env python3
"""
populate_distributions.py  —  v7  (2025-07-29)

• Expects the master JSON at  1_data/converted_questions.json
• Each object in the JSON is a single question (no nested surveys)
• The .sav files live in 0_raw_data/<wave>/…  (paths already in "file")

For every question *missing* a "distribution" field, the script

    1. loads the correct .sav once (caches per path)
    2. finds a weight variable
    3. matches survey_qid → SPSS column name (robust heuristics)
    4. computes weighted (or unweighted) percentages
    5. inserts the "distribution" block in-place

At the end it overwrites the original JSON **in place**.
"""

import json, pathlib, sys, warnings, difflib, re, tempfile, shutil
from collections import defaultdict
import pandas as pd

# ── matching helpers ──────────────────────────────────────────────────────────
def _normal(s: str) -> str:
    """Strip non-alphanumerics and upper-case"""
    return ''.join(ch for ch in s.upper() if ch.isalnum())

def _find_weight(df):
    for c in df.columns:
        if c.upper().startswith(("WEIGHT", "FINALW", "PW", "FWEIGHT")):
            return c
    return None

def _best_match(columns, target):
    """Return best matching column name or None; fuzzy/substring tolerant."""
    cols = list(columns)
    tu   = target.upper()
    t8   = tu[:8]
    tn   = _normal(target)

    # pass-1 exact
    for c in cols:
        if c.upper() == tu:
            return c
    # pass-2 8-char trunc
    for c in cols:
        if c.upper() == t8:
            return c
    # pass-3 prefix / suffix containment
    for c in cols:
        cu = c.upper()
        if cu.startswith(tu) or tu.startswith(cu):
            return c
    # pass-4 normalised equality
    for c in cols:
        if _normal(c) == tn:
            return c
    # pass-5 substring of normalised
    for c in cols:
        cn = _normal(c)
        if tn in cn or cn in tn:
            return c
    # pass-6 single very-close fuzzy hit
    near = difflib.get_close_matches(target, cols, n=2, cutoff=0.80)
    if len(near) == 1:
        return near[0]
    return None, difflib.get_close_matches(target, cols, n=5, cutoff=0.60)

def _distribution(df, col, wt):
    counts = (df.groupby(col)[wt].sum() if wt else df[col].value_counts(dropna=True))
    total  = counts.sum()
    return {str(k): round(float(v/total), 3) for k, v in counts.items()}

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    HERE = pathlib.Path(__file__).resolve().parent
    json_path = HERE / "converted_questions.json"
    if not json_path.exists():
        sys.exit(f"✖ JSON not found: {json_path}")

    questions = json.loads(json_path.read_text(encoding="utf-8"))

    # cache: sav-path -> (DataFrame, weight_var)
    cache: dict[pathlib.Path, tuple[pd.DataFrame, str|None]] = {}

    filled = 0
    for q in questions:
        if "distribution" in q:               # already filled
            continue

        sav_path = pathlib.Path(q["file"]).expanduser()

        # load & cache the SAV
        if sav_path not in cache:
            if not sav_path.exists():
                print(f"[warn] missing file {sav_path}")
                cache[sav_path] = (None, None)
                continue
            print(f"\nLoading {sav_path.name} …")
            df = pd.read_spss(sav_path, convert_categoricals=False,
                               dtype_backend="numpy_nullable")
            wt = _find_weight(df)
            print("  weight variable:", wt or "none")
            cache[sav_path] = (df, wt)

        df, wt = cache[sav_path]
        if df is None:
            continue  # file missing

        varname = q["survey_qid"]
        matched = _best_match(df.columns, varname)

        if isinstance(matched, tuple):
            # got (None, suggestions)
            print(f"    ✖ {varname} not matched. Suggestions: {', '.join(matched[1])}")
            continue
        if matched is None:
            print(f"    ✖ {varname} not found")
            continue

        q["distribution"] = _distribution(df, matched, wt)
        filled += 1
        print(f"    ✔ {varname}  →  {matched}")

    # --- atomic overwrite ---
    tmp = tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8",
            dir=json_path.parent, suffix=".tmp")
    json.dump(questions, tmp, indent=4, ensure_ascii=False)
    tmp.close()
    shutil.move(tmp.name, json_path)     # replace original file
    print(f"\n✓ Inserted distributions for {filled} questions.")
    print(f"✓ Updated file saved to {json_path}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
