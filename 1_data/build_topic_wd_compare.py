"""
Build per-topic/per-question WD comparison JSON + topic/global summaries.

Layout (relative to this script at root/1_data/):
- Topics map (default):     root/1_data/topic_to_question.json
  Supported schemas:
    A) [{"Topic": "<name>", "Question_ids": [..]}, ...]
    B) {"<name>": [..], "<name2>": [..], ...}
- WD inputs (default):      root/wd_comparisons/*.json  (fixed schema)
- Output (default):         root/1_data/topic_wd_compare.json

Fixed WD schema (abridged):
{
  "comparisons": [
    {"id": <int>, "WD_survey_vs_model": <float>, "variant": "no_refusal"|"with_refusal", ...},
    ...
  ],
  "summary": {
    "experiment_metadata": {"model": <str>, "language": <str>, "variant": <str>, ...},
    "comparison_stats": {"average_WD_survey_vs_model": <float>, ...}
  }
}

Rules:
- Ignore question id 53 everywhere (no rows, no averaging).
- Topic summaries: for each (model, language, variant) → avg WD over that topic’s questions.
- Global summary: mean of per-file 'average_WD_survey_vs_model', and per-(model,language,variant) aggregation.
- NEW: comparisons & summaries are split into 'no_refusal' and 'with_refusal' lists.

CLI:
  python build_topic_wd_compare.py
  [--topics PATH] [--comparisons-dir PATH] [--out PATH]
  [--only SUBSTR ...]   # keep only files whose *filenames* contain all SUBSTRs (case-insensitive)

Version: 2
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SKIP_QIDS = {"53"}  # skip entirely (no rows, no averaging))
VAR_NO = "no_refusal"
VAR_WITH = "with_refusal"

# ---------- Utilities ----------

def script_paths() -> Tuple[Path, Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent                # root/1_data
    root_dir = script_dir.parent                                # root
    topic_map_path = script_dir / "topic_to_question.json"      # root/1_data/topic_to_question.json
    comparisons_dir = root_dir / "wd_comparisons"               # root/wd_comparisons
    output_path = script_dir / "topic_wd_compare.json"          # root/1_data/topic_wd_compare.json
    return script_dir, root_dir, topic_map_path, comparisons_dir, output_path

def load_json_with_bom_retry(path: Path) -> Any:
    # Try normal UTF-8; on failure (e.g., BOM) retry with 'utf-8-sig'
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

def mean_or_none(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None

def to_qid_str(x: Any) -> str:
    return str(x).strip()

def normalize_topic_map(obj: Any) -> Dict[str, List[str]]:
    """
    Accepts:
      A) list of {"Topic": <str>, "Question_ids": [..]}
      B) dict mapping topic -> [qids]
    Returns {topic: [qid_str, ...]} excluding SKIP_QIDS.
    """
    out: Dict[str, List[str]] = {}
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            topic = item.get("Topic")
            qids = item.get("Question_ids")
            if isinstance(topic, str) and isinstance(qids, list):
                cleaned = [to_qid_str(q) for q in qids if to_qid_str(q) not in SKIP_QIDS]
                out[topic] = cleaned
    elif isinstance(obj, dict):
        for topic, qids in obj.items():
            if isinstance(topic, str) and isinstance(qids, list):
                cleaned = [to_qid_str(q) for q in qids if to_qid_str(q) not in SKIP_QIDS]
                out[topic] = cleaned
    return out

def var_bucket_name(v: str) -> str:
    v = (v or "").strip().lower()
    if v == VAR_NO:
        return VAR_NO
    if v == VAR_WITH:
        return VAR_WITH
    # default any unexpected value to with_refusal bucket to keep output predictable
    return VAR_WITH

# ---------- Core ----------

def build(args: argparse.Namespace) -> int:
    script_dir, root_dir, default_topics, default_comparisons, default_out = script_paths()

    topics_path = Path(args.topics or default_topics)
    comparisons_dir = Path(args.comparisons_dir or default_comparisons)
    out_path = Path(args.out or default_out)

    # 1) Load topic mapping (BOM-tolerant)
    if not topics_path.exists():
        print(f"[ERROR] Missing topic map: {topics_path}", file=sys.stderr)
        return 2
    topic_raw = load_json_with_bom_retry(topics_path)
    topic_to_qids = normalize_topic_map(topic_raw)
    if not topic_to_qids:
        print("[ERROR] topic_to_question.json must be either "
              "[{ 'Topic': str, 'Question_ids': list }, ...] or { topic: list }", file=sys.stderr)
        return 2

    # 2) Prepare output skeleton (comparisons split by variant)
    out: Dict[str, Any] = {"topics": {}, "global_summary": {}}
    for topic, qids in sorted(topic_to_qids.items()):
        out["topics"][topic] = {
            "questions": {
                qid: {"comparisons": {VAR_NO: [], VAR_WITH: []}}
                for qid in sorted(qids, key=lambda s: int(s) if s.isdigit() else s)
            },
            "summary": {"by_model_language_variant": {VAR_NO: [], VAR_WITH: []}},
        }

    # 3) Scan input files (with optional --only filters on filename)
    if not comparisons_dir.exists():
        print(f"[ERROR] Missing wd_comparisons dir: {comparisons_dir}", file=sys.stderr)
        return 2

    files = sorted([p for p in comparisons_dir.glob("*.json") if p.is_file()])
    only_terms = [t.lower() for t in (args.only or [])]
    if only_terms:
        def keep(name: str) -> bool:
            name_l = name.lower()
            return all(term in name_l for term in only_terms)
        files = [p for p in files if keep(p.name)]

    if not files:
        print(f"[WARN] No JSON files found after filtering in {comparisons_dir}", file=sys.stderr)

    # Accumulators
    # Topic averages split by variant:
    #   {(topic, model, lang, variant_bucket): [wd, ...]}
    topic_combo_wds: Dict[Tuple[str, str, str, str], List[float]] = {}

    # Global, file-level averages split by variant bucket:
    #   {(model, lang, variant_bucket): [file_avg, ...]}
    combo_file_level_avgs: Dict[Tuple[str, str, str], List[float]] = {}

    all_file_level_avgs: List[float] = []
    files_with_summary_metric = 0

    for path in files:
        try:
            data = load_json_with_bom_retry(path)
        except Exception as e:
            print(f"[WARN] Failed to parse JSON {path.name}: {e}", file=sys.stderr)
            continue

        # Fixed schema extraction
        try:
            comps = data["comparisons"]
            meta = data["summary"]["experiment_metadata"]
            stats = data["summary"]["comparison_stats"]
        except Exception as e:
            print(f"[WARN] {path.name} missing expected keys: {e}", file=sys.stderr)
            continue

        if not isinstance(comps, list):
            print(f"[WARN] {path.name} 'comparisons' is not a list; skipping", file=sys.stderr)
            continue

        model = str(meta.get("model", "")).strip()
        language = str(meta.get("language", "")).strip()
        file_variant_bucket = var_bucket_name(str(meta.get("variant", "")).strip())

        file_avg = stats.get("average_WD_survey_vs_model", None)
        if isinstance(file_avg, (int, float)):
            files_with_summary_metric += 1
            all_file_level_avgs.append(float(file_avg))
            combo_file_level_avgs.setdefault((model, language, file_variant_bucket), []).append(float(file_avg))

        # Map id -> (wd, variant_bucket) for fast lookup, skipping QID 53
        qid_to_entry: Dict[str, Tuple[Optional[float], str]] = {}
        for entry in comps:
            try:
                qid = to_qid_str(entry["id"])
                if qid in SKIP_QIDS:
                    continue
                wd = entry.get("WD_survey_vs_model", None)
                wd_val = float(wd) if isinstance(wd, (int, float)) else None
                vbucket = var_bucket_name(str(entry.get("variant", meta.get("variant", ""))))
                qid_to_entry[qid] = (wd_val, vbucket)
            except Exception:
                continue

        # Fill per-topic rows (into the variant bucket) and accumulate variant-split topic averages
        for topic, qids in topic_to_qids.items():
            for qid in qids:
                if qid in qid_to_entry:
                    wd_val, vbucket = qid_to_entry[qid]
                    out["topics"][topic]["questions"][qid]["comparisons"][vbucket].append({
                        "model": model,
                        "language": language,
                        "variant": vbucket,
                        "WD_survey_vs_model": wd_val,
                        "source_file": path.name,
                    })
                    if wd_val is not None:
                        topic_combo_wds.setdefault((topic, model, language, vbucket), []).append(wd_val)

    # 4) Topic summaries (ordered high → low) split into the two variant buckets
    for topic in out["topics"]:
        rows_no: List[Dict[str, Any]] = []
        rows_with: List[Dict[str, Any]] = []
        for (t, model, language, vbucket), vals in topic_combo_wds.items():
            if t != topic:
                continue
            avg = mean_or_none(vals)
            if avg is None:
                continue
            row = {
                "model": model,
                "language": language,
                "variant": vbucket,
                "avg_WD_over_topic_questions": avg,
                "n_questions_counted": len(vals),
            }
            if vbucket == VAR_NO:
                rows_no.append(row)
            else:
                rows_with.append(row)

        rows_no.sort(
            key=lambda r: (r["avg_WD_over_topic_questions"], r["n_questions_counted"], r["model"], r["language"]),
            reverse=True,
        )
        rows_with.sort(
            key=lambda r: (r["avg_WD_over_topic_questions"], r["n_questions_counted"], r["model"], r["language"]),
            reverse=True,
        )
        out["topics"][topic]["summary"]["by_model_language_variant"][VAR_NO] = rows_no
        out["topics"][topic]["summary"]["by_model_language_variant"][VAR_WITH] = rows_with

    # 5) Global summary (overall avg is still across all files, regardless of variant)
    out["global_summary"]["meta"] = {
        "files_scanned": len(files),
        "files_with_summary_metric": files_with_summary_metric,
    }
    out["global_summary"]["overall_average_WD_survey_vs_model_across_files"] = mean_or_none(all_file_level_avgs)

    # Split by variant bucket for the per-(model,language,variant) aggregation
    by_combo_no: List[Dict[str, Any]] = []
    by_combo_with: List[Dict[str, Any]] = []
    for (model, language, vbucket), vals in combo_file_level_avgs.items():
        avg = mean_or_none(vals)
        if avg is None:
            continue
        row = {
            "model": model,
            "language": language,
            "variant": vbucket,
            "avg_of_file_level_average_WD": avg,
            "n_files": len(vals),
        }
        if vbucket == VAR_NO:
            by_combo_no.append(row)
        else:
            by_combo_with.append(row)

    by_combo_no.sort(
        key=lambda r: (r["avg_of_file_level_average_WD"], r["n_files"], r["model"], r["language"]),
        reverse=True,
    )
    by_combo_with.sort(
        key=lambda r: (r["avg_of_file_level_average_WD"], r["n_files"], r["model"], r["language"]),
        reverse=True,
    )
    out["global_summary"]["by_model_language_variant"] = {
        VAR_NO: by_combo_no,
        VAR_WITH: by_combo_with,
    }

    # 6) Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✓ Wrote: {out_path}")
    return 0

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build topic WD compare JSON (variant-split).")
    parser.add_argument("--topics", help="Path to topic_to_question.json (default: root/1_data/...)", default=None)
    parser.add_argument("--comparisons-dir", help="Path to wd_comparisons dir (default: root/wd_comparisons)", default=None)
    parser.add_argument("--out", help="Output JSON path (default: root/1_data/topic_wd_compare.json)", default=None)
    parser.add_argument("--only", action="append", help="Substring filter on filenames (repeatable). Case-insensitive.", default=[])
    return parser.parse_args(argv)

def main() -> int:
    return build(parse_args(sys.argv[1:]))

if __name__ == "__main__":
    sys.exit(main())
