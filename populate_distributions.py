#!/usr/bin/env python3
"""
populate_distributions.py  –  v2
Fill in missing "Distribution" blocks inside questions.json.
"""

import json
import pathlib
import sys
import pandas as pd


# ---------- helpers ---------------------------------------------------------
def find_weight(df):
    """Return the first column whose name *starts* with common weight prefixes."""
    prefixes = ("WEIGHT", "FINALW", "PW", "FWEIGHT", "FTWEIGHT")
    for col in df.columns:
        if col.upper().startswith(prefixes):
            return col
    return None


def match_var(df_columns, requested):
    """
    Find the column in df_columns that corresponds to `requested`.
    Tries exact case-insensitive match first, then 8-char SPSS truncation,
    then a starts-with check. Returns None if no match.
    """
    req_up = requested.upper()

    # 1. exact, case-insensitive
    for col in df_columns:
        if col.upper() == req_up:
            return col

    # 2. SPSS may have truncated to 8 chars
    req8 = req_up[:8]
    for col in df_columns:
        if col.upper()[:8] == req8:
            return col

    # 3. starts-with (useful when the JSON name is shorter)
    for col in df_columns:
        if col.upper().startswith(req_up) or req_up.startswith(col.upper()):
            return col

    return None


def distribution(df, column, weight=None):
    """
    Return {code:str -> pct:float} rounded to 3 dp.
    """
    if weight:
        counts = df.groupby(column, observed=True)[weight].sum()
        total = counts.sum()
    else:
        counts = df[column].value_counts(dropna=True)
        total = counts.sum()

    pcts = (counts / total).round(3)
    return {str(k): float(v) for k, v in pcts.items()}


# ---------- main ------------------------------------------------------------
def main():
    here = pathlib.Path.cwd()
    json_path = here / "questions.json"
    if not json_path.exists():
        sys.exit("questions.json not found in cwd.")

    surveys = json.loads(json_path.read_text(encoding="utf-8"))

    for survey in surveys:
        sav_path = pathlib.Path(survey["File"]).expanduser().resolve()
        if not sav_path.exists():
            print(f"[WARN] .sav not found: {sav_path}")
            continue

        print(f"\nProcessing {sav_path.name} …")
        df = pd.read_spss(sav_path, convert_categoricals=False, dtype_backend="numpy_nullable")
        weight = find_weight(df)
        print("  Weight variable:", weight or "none (un-weighted)")

        for q in survey["Questions"]:
            if "Distribution" in q:
                continue  # already filled

            req_name = q["Variable_Name"]
            col = match_var(df.columns, req_name)
            if not col:
                print(f"    [SKIP] {req_name} – not found")
                continue

            q["Distribution"] = distribution(df, col, weight)
            print(f"    Added {req_name}  →  {q['Distribution']}")

    out_path = json_path.with_name("questions_filled.json")
    out_path.write_text(json.dumps(surveys, indent=4, ensure_ascii=False))
    print(f"\nDone. Updated file written to {out_path}")


if __name__ == "__main__":
    main()
