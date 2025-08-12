#!/usr/bin/env python3
"""
Compute Wasserstein distance (WD) between converted_questions.json and experimental results.
Compares original survey distributions with model choice probabilities.

Usage:
    python wd_compare.py <experimental_results_path> [-o wd_per_questions.json]

Version: 3 (Modified for experimental results)

• The script always loads the reference file  ./converted_questions.json  
• <experimental_results_path> should be a refusal_results.json from experiments
• Compares original survey distributions vs model choice probabilities
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
REFUSAL_PATTERNS = [
    re.compile(r"I prefer not to answer", re.I),
    re.compile(r"אני מעדיף לא לענות", re.I),  # Hebrew refusal
]
NEUTRAL_PAT = re.compile(
    r"(not\s*sure|neither|never\s*heard|don['']t\s*know|neutral)", re.I
)


def is_refusal_choice(choice_text: str) -> bool:
    """Check if a choice text represents a refusal option."""
    return any(pattern.search(choice_text) for pattern in REFUSAL_PATTERNS)


def build_rank_map(answers: dict[str, str]) -> dict[str, float]:
    """Build rank map from numeric answer codes to ranking positions."""
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
    """Normalize distribution excluding refusal codes/choices."""
    cleaned = {k: v for k, v in dist.items() if k not in REFUSAL_CODES and not is_refusal_choice(k)}
    Z = sum(cleaned.values())
    if Z == 0:
        raise ValueError("All probabilities were refusal codes.")
    return {k: v / Z for k, v in cleaned.items()}


def convert_choice_probs_to_numeric(choice_probs: dict[str, float], answers: dict[str, str]) -> dict[str, float]:
    """Convert experimental choice probabilities (text keys) to numeric keys for comparison."""
    numeric_dist = {}
    
    for choice_text, prob in choice_probs.items():
        # Skip refusal choices
        if is_refusal_choice(choice_text):
            continue
            
        # Try to match choice text to numeric code
        # Remove number prefix like "1. " from choice text
        clean_choice = re.sub(r'^\d+\.\s*', '', choice_text).strip()
        
        # Find matching numeric code by comparing answer text
        for code, answer_text in answers.items():
            if code in REFUSAL_CODES:
                continue
            if clean_choice.lower() == answer_text.lower().strip():
                numeric_dist[code] = prob
                break
        else:
            # If no exact match, try partial match
            for code, answer_text in answers.items():
                if code in REFUSAL_CODES:
                    continue
                if clean_choice.lower() in answer_text.lower() or answer_text.lower() in clean_choice.lower():
                    numeric_dist[code] = prob
                    break
    
    return numeric_dist


def wd_between(ranks, dist_a, dist_b) -> float:
    support = np.array([ranks[c] for c in ranks])
    w_a = np.array([dist_a.get(c, 0.0) for c in ranks])
    w_b = np.array([dist_b.get(c, 0.0) for c in ranks])
    return wasserstein_distance(support, support, u_weights=w_a, v_weights=w_b)


def refusal_mass(dist: dict[str, float]) -> float:
    """Return probability mass assigned to refusal codes/choices."""
    total_refusal = 0.0
    for k, v in dist.items():
        if k in REFUSAL_CODES or is_refusal_choice(str(k)):
            total_refusal += v
    return total_refusal


def load_experimental_results(exp_path: Path) -> dict:
    """Load experimental results and convert to dict keyed by question ID."""
    with open(exp_path, 'r', encoding='utf-8') as f:
        results_list = json.load(f)
    
    # Convert list to dict keyed by ID, handling both int and float IDs
    exp_results = {}
    for item in results_list:
        qid = item.get("id")
        if qid is not None:
            # Convert float IDs to int for consistency
            exp_results[int(qid) if isinstance(qid, float) else qid] = item
    
    return exp_results


def parse_experiment_path(exp_path: Path) -> tuple[str, str, str, str, str]:
    """
    Parse experiment path to extract metadata.
    Expected format: results/refusal_{variant}_{language}_{model}_tmp{temp}_seed{seed}/refusal_results.json
    
    Returns: (variant, language, model, temp, seed)
    """
    # Get the parent directory name
    dir_name = exp_path.parent.name
    
    # Parse directory name using regex - handle compound variant names like "no_refusal"
    pattern = r'refusal_((?:no_refusal|with_refusal|[^_]+))_([^_]+)_(.+?)_tmp([0-9.]+)_seed(\d+)'
    match = re.match(pattern, dir_name)
    
    if match:
        variant, language, model, temp, seed = match.groups()
        return variant, language, model, temp, seed
    else:
        # More sophisticated fallback parsing
        if dir_name.startswith('refusal_'):
            # Remove the 'refusal_' prefix
            remaining = dir_name[8:]  
            parts = remaining.split('_')
            
            if len(parts) >= 5:
                # Find tmp and seed parts
                temp_idx = seed_idx = -1
                for i, part in enumerate(parts):
                    if part.startswith('tmp'):
                        temp_idx = i
                    elif part.startswith('seed'):
                        seed_idx = i
                
                if temp_idx > 0 and seed_idx > 0:
                    # Handle compound variants like "no_refusal"
                    if parts[0] == 'no' and parts[1] == 'refusal':
                        variant = 'no_refusal'
                        language = parts[2]
                        model_parts = parts[3:temp_idx]
                    elif parts[0] == 'with' and parts[1] == 'refusal':
                        variant = 'with_refusal'
                        language = parts[2]
                        model_parts = parts[3:temp_idx]
                    else:
                        variant = parts[0]
                        language = parts[1] 
                        model_parts = parts[2:temp_idx]
                    
                    model = '_'.join(model_parts)
                    temp = parts[temp_idx][3:]  # Remove 'tmp' prefix
                    seed = parts[seed_idx][4:]  # Remove 'seed' prefix
                    
                    return variant, language, model, temp, seed
        
        return "unknown", "unknown", "unknown", "unknown", "unknown"


# --------------------------------------------------------------------------- #
# 2.  Main driver
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("experimental_results", help="path to experimental results file (refusal_results.json)")
    p.add_argument(
        "-o", "--output",
        default=None,
        help="output file (default: auto-generated based on experiment metadata)",
    )
    p.add_argument(
        "--reference", 
        default="1_data/converted_questions.json",
        help="path to reference questions file (default: 1_data/converted_questions.json)"
    )
    args = p.parse_args()

    ref_path = Path(args.reference)
    exp_path = Path(args.experimental_results)

    # Validate paths
    for path in (ref_path, exp_path):
        if not path.exists():
            sys.exit(f"Cannot find file {path!s}")

    # Parse experiment metadata from path
    variant, language, model, temp, seed = parse_experiment_path(exp_path)
    
    # Generate default output filename if not specified
    if args.output is None:
        safe_model_name = re.sub(r'[^\w\-_.]', '_', model)
        output_filename = f"wd_survey_vs_{safe_model_name}_{variant}_{language}_tmp{temp}_seed{seed}.json"
    else:
        output_filename = args.output

    print(f"Processing experiment: {variant} variant, {language} language, {model} model")

    # Auto-detect reference file based on language if using default
    if args.reference == "1_data/converted_questions.json" and language == "hebrew":
        ref_path = Path("1_data/translated_questions.json")
        print(f"Auto-detected Hebrew experiment, using Hebrew reference: {ref_path}")
        
    # Load reference questions (survey data)
    try:
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_list = json.load(f)
    except UnicodeDecodeError:
        # Try with different encodings for Hebrew files
        for encoding in ['utf-8-sig', 'windows-1255', 'cp1255', 'iso-8859-8']:
            try:
                with open(ref_path, 'r', encoding=encoding) as f:
                    ref_list = json.load(f)
                print(f"Loaded reference file using {encoding} encoding")
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        else:
            sys.exit(f"Could not load reference file {ref_path} with any encoding")
    
    # Handle both int and float IDs in reference questions  
    ref_qs = {}
    for q in ref_list:
        qid = q.get("id")
        if qid is not None:
            # Convert float IDs to int for consistency
            ref_qs[int(qid) if isinstance(qid, float) else qid] = q

    # Load experimental results 
    exp_qs = load_experimental_results(exp_path)

    # Find common question IDs
    ids_common = set(ref_qs.keys()) & set(exp_qs.keys())
    if not ids_common:
        sys.exit("No overlapping question IDs between survey and experimental results.")

    print(f"Found {len(ids_common)} common questions out of {len(ref_qs)} survey questions and {len(exp_qs)} experimental results")

    results, wd_sum, ref_refuse_sum, exp_refuse_sum = [], 0.0, 0.0, 0.0
    skipped_questions = 0

    for qid in sorted(ids_common):
        ref_q = ref_qs[qid]
        exp_q = exp_qs[qid]

        # Extract choice probabilities from experimental results
        choice_probs = exp_q.get("refusal_analysis", {}).get("choice_probabilities", {})
        
        if not choice_probs:
            print(f"Warning: No choice probabilities found for question {qid}, skipping...")
            skipped_questions += 1
            continue

        try:
            # Build rank map from reference answers
            rank_map = build_rank_map(ref_q["answers"])
            
            # Normalize reference distribution (exclude refusal)
            ref_dist = normalise_dist(ref_q["distribution"])
            
            # Convert experimental choice probabilities to numeric format and normalize
            exp_numeric_probs = convert_choice_probs_to_numeric(choice_probs, ref_q["answers"])
            exp_dist = normalise_dist(exp_numeric_probs)
            
            # Calculate Wasserstein distance
            wd = wd_between(rank_map, ref_dist, exp_dist)

            # Calculate refusal masses
            ref_refuse = refusal_mass(ref_q["distribution"])
            exp_refuse = refusal_mass(choice_probs)

            wd_sum += wd
            ref_refuse_sum += ref_refuse
            exp_refuse_sum += exp_refuse

            results.append({
                "id": qid,
                "question": ref_q["question"][:100] + "..." if len(ref_q["question"]) > 100 else ref_q["question"],
                "WD_survey_vs_model": round(wd, 6),
                "survey_refusal_rate": round(ref_refuse, 6),
                "model_refusal_rate": round(exp_refuse, 6),
                "refusal_rate_difference": round(exp_refuse - ref_refuse, 6),
                "variant": exp_q.get("variant", "unknown")
            })
            
        except (ValueError, KeyError) as e:
            print(f"Warning: Error processing question {qid}: {e}, skipping...")
            skipped_questions += 1
            continue

    if not results:
        sys.exit("No valid comparisons could be made between survey and experimental results.")

    n = len(results)
    
    # Add summary with experiment metadata
    summary = {
        "summary": {
            "experiment_metadata": {
                "variant": variant,
                "language": language, 
                "model": model,
                "temperature": temp,
                "seed": seed,
                "experiment_path": str(exp_path)
            },
            "comparison_stats": {
                "total_questions_compared": n,
                "questions_skipped": skipped_questions,
                "average_WD_survey_vs_model": round(wd_sum / n, 6),
                "average_survey_refusal_rate": round(ref_refuse_sum / n, 6),
                "average_model_refusal_rate": round(exp_refuse_sum / n, 6),
                "average_refusal_rate_difference": round((exp_refuse_sum - ref_refuse_sum) / n, 6)
            }
        }
    }
    
    # Combine results and summary
    output_data = {
        "comparisons": results,
        **summary
    }

    out_path = Path(output_filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Wrote {n} comparisons + summary to {out_path!s}")
    print(f"Skipped {skipped_questions} questions due to missing data or errors")
    print(f"Average WD: {summary['summary']['comparison_stats']['average_WD_survey_vs_model']}")


if __name__ == "__main__":
    main()