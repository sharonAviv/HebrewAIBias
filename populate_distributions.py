#!/usr/bin/env python3
"""
populate_distributions.py  —  v5
Fills missing "Distribution" blocks in questions.json.
Handles:
  • 8-char SPSS truncation
  • dots / underscores
  • wave suffixes like _W84
  • prefixed letter splits (e.g. GAP21Q4_a)
"""

import json, pathlib, sys, warnings, difflib, re
import pandas as pd

def normalise(s: str) -> str:
    """Upper-case, strip non-alphanumerics: 'FAV_US.' -> 'FAVUS'."""
    return ''.join(ch for ch in s.upper() if ch.isalnum())

def find_weight(df):
    for c in df.columns:
        if c.upper().startswith(("WEIGHT", "FINALW", "PW", "FWEIGHT")):
            return c
    return None

def best_match(cols, wanted):
    """Return the matching column *or* None."""
    w_u   = wanted.upper()
    w8    = w_u[:8]
    w_norm = re.sub(r'\W', '', wanted.upper())          # letters & digits only

    # pass-1 exact (case-ins.)
    for c in cols:
        if c.upper() == w_u:
            return c
    # pass-2 8-char SPSS trunc
    for c in cols:
        if c.upper() == w8:
            return c
    # pass-3 prefix / suffix
    for c in cols:
        cu = c.upper()
        if cu.startswith(w_u) or w_u.startswith(cu):
            return c
    # pass-4 normalised equality
    for c in cols:
        if re.sub(r'\W', '', c.upper()) == w_norm:
            return c
    # **NEW** pass-5 substring (normalised)
    for c in cols:
        c_norm = re.sub(r'\W', '', c.upper())
        if w_norm in c_norm or c_norm in w_norm:
            return c
    # **NEW** pass-6 single very-close fuzzy hit
    close = difflib.get_close_matches(wanted, cols, n=2, cutoff=0.80)
    if len(close) == 1:
        return close[0]

    # no good match → return list of suggestions for diagnostics
    return difflib.get_close_matches(wanted, cols, n=5, cutoff=0.60)

def distribution(df, col, wt):
    counts = (df.groupby(col)[wt].sum()
              if wt else df[col].value_counts(dropna=True))
    total  = counts.sum()
    return {str(k): round(float(v/total), 3) for k, v in counts.items()}

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    jfile = pathlib.Path("questions.json")
    if not jfile.exists(): sys.exit("questions.json not in cwd")

    surveys = json.loads(jfile.read_text(encoding="utf-8"))

    for s in surveys:
        sav = pathlib.Path(s["File"]).expanduser()
        if not sav.exists():
            print(f"[warn] {sav} missing"); continue

        print(f"\nProcessing {sav.name}")
        df = pd.read_spss(sav, convert_categoricals=False,
                          dtype_backend="numpy_nullable")
        wt = find_weight(df)
        print("  weight:", wt or "none")

        for q in s["Questions"]:
            if "Distribution" in q: continue

            want = q["Variable_Name"]
            hit  = best_match(df.columns, want)
            if isinstance(hit, list):      # got suggestion list → miss
                print(f"    miss {want}  suggestions: {', '.join(hit) or '—'}")
                continue

            q["Distribution"] = distribution(df, hit, wt)
            print(f"    added {want}  ←  {hit}")

    out = jfile.with_name("questions_filled.json")
    out.write_text(json.dumps(surveys, indent=4, ensure_ascii=False))
    print(f"\n✓ wrote {out}")

if __name__ == "__main__":
    main()
