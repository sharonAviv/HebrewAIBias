"""
Refusal Probability Experiment for Multiple Choice Questions

This script implements two variants of refusal probability measurement:
1. Standard logprobs without refusal option (baseline)
2. Logprobs with refusal option and refusal probability calculation

Formula: refusal_prob = exp(logprob_refusal) / sum(exp(logprob_i) for all choices)
"""

import os
import yaml
import json
import argparse
import pandas as pd
import torch
import numpy as np
from transformers import set_seed
from langchain_core.prompts import ChatPromptTemplate
from utils import (
    get_logger,
    set_keys,
    parse_response,
    get_answers,
    PROMPTS
)
from models import get_model, is_openai_model, extract_logprobs
from refusal_utils import (
    extract_choice_logprobs_fixed,
    calculate_refusal_metrics,
    compare_refusal_variants
)


def main():
    logger = get_logger(__name__)
    args = parse_args()
    logger.info(f"Args: {args}")
    
    config = load_config(args)
    set_keys(yaml.safe_load(open(r"./2_train_eval/keys.yaml.example")))
    set_seed(config["seed"])
    
    data = pd.read_json(config["data_file"])
    
    # Run both refusal experiments for all models
    for model_config in config["models"]:
        current_config = {
            **config,
            **model_config
        }
        
        if current_config["debug"]:
            current_data = data.sample(
                n=current_config["num_samples"], 
                random_state=current_config["seed"]
            )
        else:
            current_data = data
        
        # Run both variants
        for variant in ["no_refusal", "with_refusal"]:
            variant_config = {**current_config, "variant": variant}
            exp_dir = setup_experiment_dir(variant_config)
            run_refusal_experiment(variant_config, current_data, exp_dir, logger)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=r"./2_train_eval/config_refusal.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=["no_refusal", "with_refusal", "both"], default="both")
    return parser.parse_args()


def load_config(args):
    """Load and merge config"""
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    return config


def setup_experiment_dir(config):
    """Create experiment directory and save config"""
    model_name = config["model"].split("/")[-1]
    variant = config.get("variant", "unknown")
    exp_name = f"refusal_{variant}_{model_name}_tmp{config['temperature']}_seed{config['seed']}"
    exp_dir = os.path.join(config["logs_dir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    return exp_dir


def run_refusal_experiment(config, data, exp_dir, logger):
    """Run refusal probability experiment"""
    variant = config.get("variant", "no_refusal")
    logger.info(f"=== Running Refusal Experiment: {variant.upper()} ===")
    
    # Load model
    logger.info(f"Loading model: {config['model']}")
    model_result = get_model(
        config["model"],
        config["temperature"],
        config.get("use_rate_limiter", False),
        requests_per_second=config.get("requests_per_second"),
        check_every_n_seconds=config.get("check_every_n_seconds"),
        local_model_id=config.get("local")
    )
    
    # Handle different return formats
    if isinstance(model_result, tuple):
        base_llm, raw_model, tokenizer = model_result
        is_local_model = True
        logger.info("Local model loaded successfully")
    else:
        base_llm = model_result
        raw_model = None
        tokenizer = None
        is_local_model = False
        logger.info("OpenAI model loaded successfully")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPTS["system"]),
        ("human", PROMPTS["user"].format(question="{question}", answers="{answers}"))
    ])

    # Process data and create inputs with/without refusal
    user_inputs = get_refusal_inputs(data, variant)
    responses = initialize_refusal_responses(data, variant)
    
    logger.info(f"Processing {len(user_inputs)} questions with variant: {variant}")
    
    # Determine if this is an OpenAI model
    is_openai = is_openai_model(config["model"], config.get("local"))
    
    # Process each question
    for i, user_input in enumerate(user_inputs):
        logger.info(f"Processing question {i+1}/{len(user_inputs)}")
        logger.debug(f"Question: {user_input['question']}")
        
        try:
            if is_local_model:
                # Use FIXED local model probability extraction
                full_prompt = f"{PROMPTS['system']}\n\n{PROMPTS['user'].format(question=user_input['question'], answers=user_input['answers'])}"
                mc_options = [f"{k}. {v}" for k, v in user_input["allowed_answers"].items()]
                
                logger.debug(f"Full prompt: {full_prompt[:200]}...")
                logger.debug(f"MC options: {mc_options}")
                
                # Extract logprobs using FIXED method
                choice_logprobs = extract_choice_logprobs_fixed(
                    raw_model, tokenizer, full_prompt, mc_options
                )
                
                if choice_logprobs:
                    # Calculate refusal metrics using FIXED method
                    refusal_analysis = calculate_refusal_metrics(
                        choice_logprobs, 
                        variant, 
                        user_input.get("refusal_key")
                    )
                    
                    # Store results
                    responses[i]["choice_logprobs"] = choice_logprobs
                    responses[i]["refusal_analysis"] = refusal_analysis
                    
                    # Select best option based on probability (not logprob)
                    choice_probs = refusal_analysis.get("choice_probabilities", {})
                    if choice_probs:
                        best_option = max(choice_probs.items(), key=lambda x: x[1])[0]
                        best_prob = choice_probs[best_option]
                        responses[i]["selected_answer"] = best_option
                        
                        logger.info(f"Selected: {best_option} (prob: {best_prob:.4f})")
                        
                        if variant == "with_refusal" and refusal_analysis.get("has_refusal"):
                            refusal_prob = refusal_analysis.get("refusal_probability", 0)
                            logger.info(f"Refusal probability: {refusal_prob:.4f}")
                    else:
                        responses[i]["selected_answer"] = None
                else:
                    logger.warning("No choice logprobs extracted")
                    responses[i]["choice_logprobs"] = {}
                    responses[i]["refusal_analysis"] = {}
                    responses[i]["selected_answer"] = None
                
            else:
                # OpenAI model handling (similar structure but with structured output)
                logger.warning("OpenAI refusal experiment not fully implemented yet")
                responses[i]["choice_logprobs"] = {}
                responses[i]["refusal_analysis"] = {}
                responses[i]["selected_answer"] = None
                
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            responses[i]["choice_logprobs"] = {}
            responses[i]["refusal_analysis"] = {}
            responses[i]["selected_answer"] = None

    # Save results
    save_refusal_results(responses, exp_dir, variant, logger)


def get_refusal_inputs(data, variant):
    """Create inputs with or without refusal option"""
    user_inputs = []
    
    for _, row in data.iterrows():
        answers = row["answers"]
        
        if variant == "with_refusal":
            # Add refusal option
            refusal_key = str(max(int(k) for k in answers.keys()) + 1)
            answers_with_refusal = {**answers, refusal_key: "I prefer not to answer"}
            
            formatted_answers = "\n".join([f"{k}. {v}" for k, v in answers_with_refusal.items()])
            
            user_inputs.append({
                "question": row["question"],
                "answers": formatted_answers,
                "allowed_answers": answers_with_refusal,
                "refusal_key": refusal_key
            })
        else:
            # No refusal option (baseline)
            formatted_answers = "\n".join([f"{k}. {v}" for k, v in answers.items()])
            
            user_inputs.append({
                "question": row["question"],
                "answers": formatted_answers,
                "allowed_answers": answers,
                "refusal_key": None
            })
    
    return user_inputs


def initialize_refusal_responses(data, variant):
    """Initialize response storage for refusal experiment"""
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "variant": variant,
            "original_answers": row["answers"],
            "selected_answer": None,
            "choice_logprobs": {},
            "refusal_analysis": {}
        }
        for _, row in data.iterrows()
    ]



def save_refusal_results(responses, exp_dir, variant, logger):
    """Save refusal experiment results"""
    results_file = os.path.join(exp_dir, "refusal_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to {results_file}")
    
    # Generate summary statistics
    generate_refusal_summary(responses, variant, logger)


def generate_refusal_summary(responses, variant, logger):
    """Generate and log summary statistics"""
    logger.info(f"=== REFUSAL EXPERIMENT SUMMARY ({variant.upper()}) ===")
    
    total_questions = len(responses)
    successful_responses = sum(1 for r in responses if r["selected_answer"] is not None)
    
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Successful responses: {successful_responses}")
    logger.info(f"Success rate: {successful_responses/total_questions*100:.1f}%")
    
    if variant == "with_refusal":
        # Refusal-specific statistics
        refusal_selections = sum(1 for r in responses 
                               if r["selected_answer"] and "prefer not to answer" in r["selected_answer"].lower())
        
        refusal_probs = [r["refusal_analysis"].get("refusal_probability", 0) 
                        for r in responses if r["refusal_analysis"] and r["refusal_analysis"].get("has_refusal")]
        
        if refusal_probs:
            avg_refusal_prob = np.mean(refusal_probs)
            median_refusal_prob = np.median(refusal_probs)
            max_refusal_prob = np.max(refusal_probs)
            
            logger.info(f"Questions where refusal was selected: {refusal_selections}")
            logger.info(f"Average refusal probability: {avg_refusal_prob:.4f}")
            logger.info(f"Median refusal probability: {median_refusal_prob:.4f}")
            logger.info(f"Maximum refusal probability: {max_refusal_prob:.4f}")
            
            # Show top 3 questions with highest refusal probability
            valid_responses = [r for r in responses if r["refusal_analysis"].get("has_refusal")]
            sorted_responses = sorted(valid_responses, 
                                    key=lambda x: x["refusal_analysis"].get("refusal_probability", 0), 
                                    reverse=True)
            
            logger.info("Top 3 questions with highest refusal probability:")
            for i, resp in enumerate(sorted_responses[:3], 1):
                refusal_prob = resp["refusal_analysis"].get("refusal_probability", 0)
                question_preview = resp["question"][:100] + "..." if len(resp["question"]) > 100 else resp["question"]
                logger.info(f"  {i}. Refusal prob: {refusal_prob:.4f} - {question_preview}")
                
                # Show probability breakdown for this question
                choice_probs = resp["refusal_analysis"].get("choice_probabilities", {})
                if choice_probs:
                    sorted_choices = sorted(choice_probs.items(), key=lambda x: x[1], reverse=True)
                    logger.info(f"     Choice probabilities:")
                    for choice, prob in sorted_choices:
                        marker = "🚫" if "prefer not to answer" in choice.lower() else "  "
                        logger.info(f"     {marker} {choice}: {prob:.4f}")
        else:
            logger.info("No refusal probabilities calculated")
    
    # Show some example probability distributions
    logger.info("\n=== SAMPLE PROBABILITY DISTRIBUTIONS ===")
    for i, resp in enumerate(responses[:3]):
        logger.info(f"\nQuestion {i+1}: {resp['question'][:80]}...")
        analysis = resp.get("refusal_analysis", {})
        choice_probs = analysis.get("choice_probabilities", {})
        
        if choice_probs:
            sorted_choices = sorted(choice_probs.items(), key=lambda x: x[1], reverse=True)
            for choice, prob in sorted_choices:
                marker = "👑" if prob == max(choice_probs.values()) else "  "
                logger.info(f"  {marker} {choice}: {prob:.4f}")
        else:
            logger.info("  No probability data available")


if __name__ == "__main__":
    main()