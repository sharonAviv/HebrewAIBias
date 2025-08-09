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
        return convert_logprobs_to_probs(choice_logprobs)
    
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
        return convert_logprobs_to_probs(choice_logprobs)
    
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


def extract_choice_logprobs_openai(llm, prompt_template, user_input, mc_options):
    """
    Extract choice probabilities from OpenAI model with single call approach.
    
    The key insight: Make one call and extract the logprob of each choice number
    from the top_logprobs at the first token position. This gives us the model's
    natural preference for each option.
    
    Args:
        llm: The OpenAI LLM instance
        prompt_template: The prompt template  
        user_input: Dictionary with question and answers
        mc_options: List of multiple choice options (e.g., ["1. Yes", "2. No"])
        
    Returns:
        Dictionary mapping options to raw logprobs (like local model)
    """
    try:
        logger.debug(f"Extracting choice logprobs for {len(mc_options)} options")
        
        # Make a single call to get the model's natural response with logprobs
        chain = prompt_template | llm.bind(logprobs=True, top_logprobs=20)
        
        response = chain.invoke({
            "question": user_input['question'],
            "answers": user_input['answers']
        })
        
        # Extract logprobs from response
        content_logprobs = response.response_metadata.get("logprobs", {}).get("content", [])
        
        if not content_logprobs:
            logger.warning("No logprobs found in OpenAI response")
            return {option: -5.0 for option in mc_options}
        
        # Get the first token entry which should contain the choice
        first_token_data = content_logprobs[0]
        top_logprobs = first_token_data.get('top_logprobs', [])
        
        logger.debug(f"First token: '{first_token_data.get('token', '')}' with {len(top_logprobs)} alternatives")
        
        # Create mapping from choice numbers to options
        import re
        choice_to_option = {}
        for option in mc_options:
            match = re.match(r'(\d+)\.\s*(.*)', option)
            if match:
                choice_num = match.group(1)
                choice_to_option[choice_num] = option
        
        # Extract logprobs for each choice from the top_logprobs
        option_logprobs = {}
        
        # Check all top logprobs (including the actual token generated)
        all_token_data = [first_token_data] + top_logprobs
        
        for token_data in all_token_data:
            token = token_data.get('token', '').strip()
            logprob = token_data.get('logprob', -15.0)
            
            # Check if this token matches any of our choice numbers
            if token in choice_to_option:
                option = choice_to_option[token]
                if option not in option_logprobs:  # Take first occurrence (highest probability)
                    option_logprobs[option] = logprob
                    logger.debug(f"Found token '{token}' -> option '{option}': {logprob:.6f}")
        
        # For any missing options, assign very low probability
        for option in mc_options:
            if option not in option_logprobs:
                option_logprobs[option] = -15.0
                logger.debug(f"Option '{option}' not found, assigned: -15.0")
        
        logger.debug(f"Final option logprobs: {option_logprobs}")
        return option_logprobs
        
    except Exception as e:
        logger.error(f"Error extracting OpenAI choice probabilities: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {option: -5.0 for option in mc_options}


def find_best_option_match(response_content, content_logprobs, mc_options):
    """
    Find which multiple choice option best matches the response and return its logprob.
    
    Args:
        response_content: The response text from OpenAI
        content_logprobs: The logprobs from OpenAI response
        mc_options: List of multiple choice options
        
    Returns:
        Tuple of (best_matching_option, logprob) or (None, None) if no match
    """
    response_lower = response_content.lower().strip()
    best_match = None
    best_logprob = None
    
    # Method 1: Direct text matching
    import re
    for option in mc_options:
        # Extract choice number and text
        match = re.match(r'(\d+)\.\s*(.*)', option)
        if match:
            choice_num, choice_text = match.groups()
            
            # Check if response starts with choice number
            if response_lower.startswith(choice_num):
                # Find logprob of the choice number token
                for entry in content_logprobs:
                    token = entry.get('token', '').strip()
                    if token == choice_num:
                        return option, entry.get('logprob', -5.0)
                # If exact token not found, use first token logprob
                if content_logprobs:
                    return option, content_logprobs[0].get('logprob', -5.0)
            
            # Check if response contains choice text
            if choice_text.lower().strip() in response_lower:
                # Calculate average logprob for relevant tokens
                relevant_logprobs = []
                choice_words = choice_text.lower().split()
                
                for entry in content_logprobs:
                    token = entry.get('token', '').lower().strip()
                    for word in choice_words:
                        if word in token or token in word:
                            relevant_logprobs.append(entry.get('logprob', -5.0))
                            break
                
                if relevant_logprobs:
                    avg_logprob = sum(relevant_logprobs) / len(relevant_logprobs)
                    return option, avg_logprob
    
    # Method 2: Token-level analysis if no direct match
    if content_logprobs:
        for option in mc_options:
            match = re.match(r'(\d+)\.\s*(.*)', option)
            if match:
                choice_num, choice_text = match.groups()
                
                # Look for choice number in any token
                for entry in content_logprobs:
                    token = entry.get('token', '').strip()
                    if choice_num in token:
                        return option, entry.get('logprob', -5.0)
    
    # No match found
    return None, None


def calculate_token_option_match(token, option, logprob):
    """
    Calculate how well a token matches a multiple choice option.
    
    Args:
        token: The token from OpenAI response
        option: The choice option (e.g., "1. Yes")  
        logprob: The logprob of this token
        
    Returns:
        Float score (logprob if match, -inf if no match)
    """
    if not token or logprob == float('-inf'):
        return float('-inf')
    
    token_lower = token.lower().strip()
    
    # Extract choice number and text from option
    match = re.match(r'(\d+)\.\s*(.*)', option)
    if match:
        choice_num, choice_text = match.groups()
        choice_text_lower = choice_text.lower().strip()
        
        # Direct number match (highest priority)
        if token_lower == choice_num or token_lower == choice_num + '.':
            return logprob
        
        # Choice text match
        if token_lower == choice_text_lower:
            return logprob
        
        # Partial word match
        choice_words = choice_text_lower.split()
        for word in choice_words:
            if word in token_lower or token_lower in word:
                return logprob * 0.8  # Slight penalty for partial match
    
    return float('-inf')


def analyze_response_content(response_content, mc_options):
    """
    Fallback method to analyze response content when token matching fails.
    
    Args:
        response_content: The full response text
        mc_options: List of multiple choice options
        
    Returns:
        Dictionary with estimated logprobs for each option
    """
    choice_logprobs = {}
    response_lower = response_content.lower().strip()
    
    import re
    for option in mc_options:
        # Extract choice number and text
        match = re.match(r'(\d+)\.\s*(.*)', option)
        if match:
            choice_num, choice_text = match.groups()
            
            # Check for exact matches in response
            if response_lower.startswith(choice_num):
                choice_logprobs[option] = -1.0  # High probability
            elif choice_text.lower() in response_lower:
                choice_logprobs[option] = -2.0  # Medium probability  
            else:
                choice_logprobs[option] = -4.0  # Low probability
        else:
            choice_logprobs[option] = -4.0  # Low probability
    
    return choice_logprobs


def calculate_option_logprob_openai(content_logprobs, target_option, response_content):
    """
    Calculate logprob for a specific option from OpenAI response.
    
    Args:
        content_logprobs: List of logprob entries from OpenAI
        target_option: The target option (e.g., "1. Yes")
        response_content: The actual response content
        
    Returns:
        Float logprob value for this option
    """
    try:
        # Extract the choice number from target option
        match = re.match(r'(\d+)\.\s*(.*)', target_option)
        if not match:
            return -5.0
            
        choice_num, choice_text = match.groups()
        response_lower = response_content.lower().strip()
        
        # Method 1: Look for the choice number at the start of response
        if response_lower.startswith(choice_num):
            # Find logprob of the choice number token
            for entry in content_logprobs:
                token = entry.get('token', '').strip()
                if token == choice_num or token.endswith(choice_num):
                    return entry.get('logprob', -5.0)
        
        # Method 2: Look for choice text in response
        choice_words = choice_text.lower().split()
        relevant_logprobs = []
        
        for entry in content_logprobs:
            token = entry.get('token', '').lower().strip()
            logprob = entry.get('logprob')
            
            if logprob is not None:
                # Check if token matches any part of the choice
                for word in choice_words:
                    if word in token or token in word:
                        relevant_logprobs.append(logprob)
                        break
        
        if relevant_logprobs:
            # Return average logprob of relevant tokens
            return sum(relevant_logprobs) / len(relevant_logprobs)
        
        # Method 3: Default based on response similarity
        if any(word.lower() in response_lower for word in choice_text.split()):
            return -2.0  # Moderate probability
        else:
            return -5.0  # Low probability
            
    except Exception as e:
        logger.error(f"Error calculating option logprob: {e}")
        return -5.0


def find_token_logprob(content_logprobs, target_token):
    """Find logprob for a specific token in content logprobs"""
    target_lower = target_token.lower().strip()
    
    for entry in content_logprobs:
        token = entry.get('token', '').lower().strip()
        if token == target_lower or token == target_lower + '.' or token.endswith(target_lower):
            return entry.get('logprob', None)
    
    return None


def estimate_text_logprob(content_logprobs, target_text):
    """Estimate logprob for a text span by averaging relevant tokens"""
    target_words = target_text.lower().split()
    relevant_logprobs = []
    
    for entry in content_logprobs:
        token = entry.get('token', '').lower().strip()
        
        # Check if token is part of target text
        for word in target_words:
            if word in token or token in word:
                logprob = entry.get('logprob')
                if logprob is not None:
                    relevant_logprobs.append(logprob)
                break
    
    if relevant_logprobs:
        return sum(relevant_logprobs) / len(relevant_logprobs)
    return None


def analyze_choice_tokens(content_logprobs, mc_options):
    """Analyze all tokens to find best matches for multiple choice options"""
    choice_logprobs = {}
    
    # Extract all tokens and their logprobs
    all_tokens = []
    for entry in content_logprobs:
        token = entry.get('token', '')
        logprob = entry.get('logprob')
        if token and logprob is not None:
            all_tokens.append((token.lower().strip(), logprob))
    
    # For each option, find the best matching token(s)
    for option in mc_options:
        option_lower = option.lower()
        best_logprob = float('-inf')
        
        # Look for choice number
        match = re.match(r'(\d+)\.\s*(.*)', option)
        if match:
            choice_num = match.group(1)
            
            # Find tokens that match the choice number
            for token, logprob in all_tokens:
                if token == choice_num or token.startswith(choice_num):
                    best_logprob = max(best_logprob, logprob)
        
        # If no good match found, use a reasonable default
        if best_logprob == float('-inf'):
            best_logprob = -3.0  # Low but reasonable probability
        
        choice_logprobs[option] = best_logprob
    
    return choice_logprobs


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