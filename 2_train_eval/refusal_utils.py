"""
Fixed refusal probability utilities with proper logprob calculation.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def extract_choice_logprobs_fixed(model, tokenizer, prompt, multiple_choice_options):
    """
    Extract log probabilities for multiple choice options with proper normalization.
    
    This version fixes several issues:
    1. Uses average logprob per token to avoid length bias
    2. Calculates proper conditional probabilities P(choice | prompt)
    3. Returns raw logprobs without forced normalization
    
    :param model: The HuggingFace model instance
    :param tokenizer: The tokenizer instance  
    :param prompt: The input prompt string
    :param multiple_choice_options: List of multiple choice answer strings
    :return: Dictionary mapping options to their log probabilities
    """
    try:
        option_logprobs = {}
        
        for option in multiple_choice_options:
            logger.debug(f"Evaluating option: '{option}'")
            
            # Method 1: Direct continuation probability
            # Calculate P(option | prompt) using model's next-token predictions
            logprob = calculate_continuation_logprob(model, tokenizer, prompt, option)
            option_logprobs[option] = logprob
            
            logger.debug(f"Option '{option}' logprob: {logprob:.6f}")
        
        logger.debug(f"Raw option logprobs: {option_logprobs}")
        
        # Return raw logprobs - let caller decide how to normalize
        return option_logprobs
        
    except Exception as e:
        logger.error(f"Error extracting choice logprobs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def calculate_continuation_logprob(model, tokenizer, prompt, continuation):
    """
    Calculate log probability of continuation given prompt.
    Uses proper conditional probability: P(continuation | prompt)
    """
    # Tokenize prompt and full sequence
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    full_sequence = prompt + " " + continuation
    full_tokens = tokenizer(full_sequence, return_tensors="pt").to(model.device)
    
    prompt_length = prompt_tokens['input_ids'].shape[1]
    full_length = full_tokens['input_ids'].shape[1]
    
    if full_length <= prompt_length:
        logger.warning(f"Continuation '{continuation}' resulted in no new tokens")
        return float('-inf')
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**full_tokens, use_cache=False)
        logits = outputs.logits
    
    # Calculate log probability for continuation tokens
    total_logprob = 0.0
    continuation_length = full_length - prompt_length
    
    for pos in range(prompt_length, full_length):
        if pos < logits.shape[1]:
            # Get log probabilities at previous position (autoregressive)
            log_probs = F.log_softmax(logits[0, pos - 1, :], dim=-1)
            
            # Get the actual token at current position
            actual_token_id = full_tokens['input_ids'][0, pos].item()
            
            # Add log probability of this token
            token_logprob = log_probs[actual_token_id].item()
            total_logprob += token_logprob
            
            logger.debug(f"Position {pos}, token {tokenizer.decode([actual_token_id])}, logprob: {token_logprob:.6f}")
    
    # Average logprob per token to avoid length bias
    avg_logprob = total_logprob / continuation_length if continuation_length > 0 else float('-inf')
    
    logger.debug(f"Continuation '{continuation}': total={total_logprob:.6f}, avg={avg_logprob:.6f}, tokens={continuation_length}")
    
    return avg_logprob


def calculate_refusal_metrics(choice_logprobs, variant, refusal_key=None):
    """
    Calculate refusal probability metrics from choice logprobs.
    
    :param choice_logprobs: Dict mapping choices to their log probabilities
    :param variant: 'no_refusal' or 'with_refusal'
    :param refusal_key: Key identifying the refusal option
    :return: Dictionary with refusal analysis
    """
    if variant != "with_refusal" or not refusal_key:
        return {
            "variant": variant,
            "has_refusal": False,
            "choice_probabilities": convert_logprobs_to_probs(choice_logprobs)
        }
    
    # Find refusal option
    refusal_logprob = None
    refusal_option = None
    
    for choice, logprob in choice_logprobs.items():
        if refusal_key in choice or "prefer not to answer" in choice.lower():
            refusal_logprob = logprob
            refusal_option = choice
            break
    
    if refusal_logprob is None:
        logger.warning("Refusal option not found in choice_logprobs")
        return {
            "variant": variant,
            "has_refusal": False,
            "error": "Refusal option not found",
            "choice_probabilities": convert_logprobs_to_probs(choice_logprobs)
        }
    
    # Convert logprobs to probabilities using softmax
    choice_probs = convert_logprobs_to_probs(choice_logprobs)
    
    refusal_prob = choice_probs[refusal_option]
    
    # Calculate non-refusal probability
    non_refusal_prob = sum(prob for choice, prob in choice_probs.items() 
                          if choice != refusal_option)
    
    # Refusal ratio as requested: exp(logprob_refusal) / sum(exp(logprob_i) for all choices)
    refusal_ratio = refusal_prob  # This is already the normalized probability
    
    # Alternative ratio: refusal vs non-refusal
    refusal_vs_others = refusal_prob / non_refusal_prob if non_refusal_prob > 0 else float('inf')
    
    return {
        "variant": variant,
        "has_refusal": True,
        "refusal_option": refusal_option,
        "refusal_logprob": refusal_logprob,
        "refusal_probability": refusal_prob,
        "refusal_ratio": refusal_ratio,
        "non_refusal_probability": non_refusal_prob,
        "refusal_vs_others_ratio": refusal_vs_others,
        "choice_probabilities": choice_probs,
        "raw_logprobs": choice_logprobs
    }


def convert_logprobs_to_probs(logprobs_dict):
    """
    Convert log probabilities to probabilities using softmax normalization.
    This is the correct way to get P(choice_i) from logprobs.
    """
    if not logprobs_dict:
        return {}
    
    # Convert to tensor for numerical stability
    logprobs_tensor = torch.tensor(list(logprobs_dict.values()), dtype=torch.float32)
    
    # Apply softmax to get probabilities
    probs_tensor = F.softmax(logprobs_tensor, dim=0)
    
    # Convert back to dictionary
    probs_dict = {choice: float(prob) for choice, prob in 
                  zip(logprobs_dict.keys(), probs_tensor)}
    
    return probs_dict


def compare_refusal_variants(no_refusal_results, with_refusal_results):
    """
    Compare results between no_refusal and with_refusal variants.
    
    :param no_refusal_results: Results from baseline experiment
    :param with_refusal_results: Results from refusal experiment
    :return: Comparison analysis
    """
    comparison = {
        "total_questions": len(no_refusal_results),
        "baseline_predictions": [],
        "refusal_predictions": [],
        "probability_shifts": [],
        "refusal_statistics": {
            "questions_with_refusal_selected": 0,
            "avg_refusal_probability": 0.0,
            "high_refusal_questions": []  # Questions with >0.3 refusal probability
        }
    }
    
    refusal_probs = []
    
    for i, (baseline, refusal) in enumerate(zip(no_refusal_results, with_refusal_results)):
        if baseline["id"] != refusal["id"]:
            logger.warning(f"ID mismatch at index {i}: {baseline['id']} vs {refusal['id']}")
            continue
        
        baseline_analysis = baseline.get("refusal_analysis", {})
        refusal_analysis = refusal.get("refusal_analysis", {})
        
        comparison["baseline_predictions"].append(baseline.get("selected_answer"))
        comparison["refusal_predictions"].append(refusal.get("selected_answer"))
        
        # Track refusal statistics
        if refusal_analysis.get("has_refusal", False):
            refusal_prob = refusal_analysis.get("refusal_probability", 0.0)
            refusal_probs.append(refusal_prob)
            
            # Check if refusal was selected
            selected = refusal.get("selected_answer", "")
            if "prefer not to answer" in selected.lower():
                comparison["refusal_statistics"]["questions_with_refusal_selected"] += 1
            
            # Track high refusal probability questions
            if refusal_prob > 0.3:
                comparison["refusal_statistics"]["high_refusal_questions"].append({
                    "question_id": refusal["id"],
                    "question": refusal["question"][:100] + "...",
                    "refusal_probability": refusal_prob,
                    "selected_answer": selected
                })
    
    if refusal_probs:
        comparison["refusal_statistics"]["avg_refusal_probability"] = sum(refusal_probs) / len(refusal_probs)
        comparison["refusal_statistics"]["median_refusal_probability"] = sorted(refusal_probs)[len(refusal_probs)//2]
        comparison["refusal_statistics"]["max_refusal_probability"] = max(refusal_probs)
    
    return comparison