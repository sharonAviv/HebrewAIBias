#!/usr/bin/env python3
# group_questions.py
"""
Group Pew-research questions by semantic similarity.

Usage:
    python group_questions.py \\ 
        --input  converted_questions.json \\
        --output question_groups.json \\
        --threshold 0.80 (default)

Version: 1
"""

import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------------------------------------------------
# 1. Parse CLI arguments
# ---------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Group similar questions")
    p.add_argument("--input",  required=True, help="path/to/converted_questions.json")
    p.add_argument("--output", required=True, help="path/to/question_groups.json")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="cosine-similarity threshold for grouping (default 0.80)")
    return p.parse_args()

# ---------------------------------------------------------------------
# 2. Simple “union-find” building groups on the fly
# ---------------------------------------------------------------------
def greedy_group(ids, embeddings, threshold: float):
    """
    Greedy clustering: each new item joins the first existing
    group whose representative exceeds the threshold; otherwise
    it starts a new group.

    Parameters
    ----------
    ids         : list[int]
    embeddings  : torch.Tensor  shape = (N, dim)
    threshold   : float         cosine similarity threshold

    Returns
    -------
    list[list[int]]  – groups of question IDs
    """
    groups, reps = [], []            # reps = representative embedding per group

    for idx, emb in zip(ids, embeddings):
        placed = False
        for g_i, rep in enumerate(reps):
            score = util.cos_sim(emb, rep).item()
            if score >= threshold:
                groups[g_i].append(idx)
                # update group representative by simple average
                reps[g_i] = (rep * (len(groups[g_i]) - 1) + emb) / len(groups[g_i])
                placed = True
                break
        if not placed:               # start a new group
            groups.append([idx])
            reps.append(emb)
    return groups

# ---------------------------------------------------------------------
# 3. Main routine
# ---------------------------------------------------------------------
def main():
    args = get_args()

    # --- load questions ------------------------------------------------
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    ids, questions = zip(*[(d["id"], d["question"]) for d in data])

    # --- embed with a lightweight model --------------------------------
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(questions, convert_to_tensor=True, device=model.device)

    # --- group by similarity ------------------------------------------
    groups = greedy_group(ids, embeddings, threshold=args.threshold)

    # --- write result --------------------------------------------------
    out = [{"questions_ids": sorted(g)} for g in groups]
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✓ saved {len(out)} groups to {args.output}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)     # no gradients needed
    main()
