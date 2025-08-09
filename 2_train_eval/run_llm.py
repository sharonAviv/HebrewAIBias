"""
Running LLM inference on survey questions with separate inference/evaluation modes.
"""

import os
import yaml
import json
import argparse
import pandas as pd
import torch
from transformers import set_seed
from langchain_core.prompts import ChatPromptTemplate
from utils import (
    get_logger,
    set_keys,
    parse_response,
    get_answers,
    # run_individual,
    PROMPTS
)
from models import get_model, get_schema, is_openai_remote_model, extract_logprobs, extract_logprobs_from_local_model


def main():
    logger = get_logger(__name__)
    args = parse_args()
    logger.info(f"Args: {args}")
    
    config = load_config(args)
    set_keys(yaml.safe_load(open(r"./2_train_eval/keys.yaml.example")))
    set_seed(config["seed"])
    
    data = pd.read_json(config["data_file"])
    
    # Run selected mode for all models (new loop)
    for model_config in config["models"]:
        current_config = {
            **config,          # Global settings (prompt_type, seed, mode)
            **model_config     # Model-specific configurations
        }
        
        if current_config["debug"]:
            current_data = data.sample(
                n=current_config["num_samples"], 
                random_state=current_config["seed"]
            )
        else:
            current_data = data
        
        model_dir = setup_model_dir(current_config)
        for variant in ["no_refusal", "with_refusal"]:
            exp_dir = os.path.join(model_dir, variant)
            os.makedirs(exp_dir, exist_ok=True)
            if current_config["mode"] == "inference":
                run_inference(current_config, current_data, exp_dir, variant, logger)
            else:
                run_evaluation(current_config, current_data, exp_dir, logger)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=r"./2_train_eval/config_llm.yaml")
    parser.add_argument("--keys_file", help="Path to keys YAML file", default=r"./2_train_eval/keys.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=["inference", "evaluation"])
    parser.add_argument("--responses_dir")
    return parser.parse_args()

def load_config(args):
    """Load and merge config"""
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    return config

def setup_model_dir(config):
    """Create experiment directory and save config"""
    model_name = config["model"].split("/")[-1]
    exp_name = f"{model_name}_{config['prompt_type']}_tmp{config['temperature']}_seed{config['seed']}"
    exp_dir = os.path.join(config["logs_dir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    return exp_dir

def run_inference(config, data, exp_dir, variant, logger):
    """Run LLM inference and save responses"""
    from enum import Enum
    from pydantic import BaseModel, create_model

    # ---- FIX: base class must inherit BaseModel
    class _SurveyBase(BaseModel):
        # For Pydantic v2, a dict here is fine; ensures we get enum values (strings)
        model_config = {"use_enum_values": True}

    # 1) Base LLM (no global structured schema)
    logger.info(f"Loading model: {config['model']}")
    logger.info(f"Temperature: {config['temperature']}")
    logger.info(f"Local model ID: {config.get('local', 'None')}")
    
    model_result = get_model(
        config["model"],
        config["temperature"],
        config.get("use_rate_limiter", False),
        requests_per_second=config.get("requests_per_second"),
        check_every_n_seconds=config.get("check_every_n_seconds"),
        local_model_id=config.get("local")  # Pass local model ID from YAML
    )
    
    # Handle different return formats
    if isinstance(model_result, tuple):
        # Local model returns (llm, model, tokenizer)
        base_llm, raw_model, tokenizer = model_result
        is_local_model = True
        logger.info("Local model loaded successfully")
    else:
        # OpenAI model returns just the llm
        base_llm = model_result
        raw_model = None
        tokenizer = None
        is_local_model = False
        logger.info("OpenAI model loaded successfully")

    # 2) Prompt (reuse for all questions)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPTS["system"]),
        ("human", PROMPTS["user"].format(question="{question}", answers="{answers}"))
    ])

    # 3) Inputs & storage
    user_inputs = get_answers(data, variant)        # builds "answers" and "allowed_answers"
    responses = initialize_responses(data, variant) # your existing accumulator
    
    logger.info(f"Processing {len(user_inputs)} questions with variant: {variant}")
    
    # 4) Determine if this is an OpenAI model for logprob handling
    is_openai = is_openai_remote_model(config["model"], config.get("local"))

    # Helper: per-question schema with Enum via create_model
    def _make_schema_for_question(allowed_answers: dict[str, str]):
        # Display strings exactly as shown to the model ("1. Dogs", "2. Cats", ...)
        values = [f"{k}. {v}" for k, v in allowed_answers.items()]
        AnswerEnum = Enum(
            "AnswerEnum",
            {f"OPT_{i}": val for i, val in enumerate(values, start=1)}
        )
        # Build the model dynamically; __base__ MUST be a BaseModel subclass (fixed above)
        return create_model(
            "SurveyResponse",
            answer=(AnswerEnum, ...),
            __base__=_SurveyBase,
            __module__=__name__,
        )

    # 5) Run
    logger.info(f"=== Run {run_index + 1}/{total_runs} ===")

    for i, user_input in enumerate(user_inputs):
        logger.info(f"Processing question {i+1}/{len(user_inputs)}")
        logger.debug(f"Question: {user_input['question']}")
        
        try:
            SchemaForQ = _make_schema_for_question(user_input["allowed_answers"])
            
            if is_local_model:
                logger.debug("Using local model for inference")
                # For local models, use direct probability extraction
                # Format the prompt for the model
                full_prompt = f"{PROMPTS['system']}\n\n{PROMPTS['user'].format(question=user_input['question'], answers=user_input['answers'])}"
                logger.debug(f"Full prompt: {full_prompt[:250]}...")
                
                
                # Get multiple choice options (formatted as they appear to the model)
                mc_options = [f"{k}. {v}" for k, v in user_input["allowed_answers"].items()]
                logger.debug(f"MC options: {mc_options}")
                
                # Extract probabilities for multiple choice options
                logger.debug("Extracting probabilities from local model...")
                choice_logprobs = extract_logprobs_from_local_model(
                    raw_model, tokenizer, full_prompt, mc_options
                )
                
                # Select the option with highest probability
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
                # For OpenAI models, use structured output as before
                llm_q = base_llm.with_structured_output(
                    SchemaForQ, include_raw=True, method="function_calling"
                )
                chain_q = prompt | llm_q

                # Invoke (no pydantic context/config needed)
                resp = chain_q.invoke({
                    "question": user_input["question"],
                    "answers": user_input["answers"],
                })

                # Store raw for traceability (best-effort)
                raw_text = ""
                try:
                    raw_msg = resp.get("raw") if isinstance(resp, dict) else None
                    raw_text = getattr(raw_msg, "content", "") or str(resp)
                except Exception:
                    raw_text = str(resp)
                responses[i]["raw_response"].append(raw_text)

                # Extract and store logprobs
                logprobs = extract_logprobs(
                    resp.get("raw") if isinstance(resp, dict) else resp, 
                    is_openai=is_openai
                )
                responses[i]["logprobs"].append(logprobs)

                # Parse & store the answer (string)
                parsed = parse_response(resp)   # your helper returns {"answer": "..."}
                ans = parsed.get("answer")
                # If exporter returned an Enum member, coerce to value
                try:
                    from enum import Enum as _PyEnum
                    if isinstance(ans, _PyEnum):
                        ans = ans.value
                except Exception:
                    pass
                responses[i]["responses"].append(ans if isinstance(ans, str) else None)

        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            responses[i]["responses"].append(None)
            responses[i]["raw_response"].append("")
            responses[i]["logprobs"].append(None)

    logger.info("=== INFERENCE COMPLETE ===")
    logger.info(f"Processed {len(responses)} questions")
    
    # Summary statistics
    total_responses = sum(len(r["responses"]) for r in responses)
    successful_responses = sum(1 for r in responses for resp in r["responses"] if resp is not None)
    logger.info(f"Total responses: {total_responses}")
    logger.info(f"Successful responses: {successful_responses}")
    logger.info(f"Success rate: {successful_responses/total_responses*100:.1f}%")
    
    save_responses(responses, exp_dir, logger)



def initialize_responses(data, variant):
    """Initialize response storage structure"""
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "variant": variant,
            "original_answers": row["answers"],
            "selected_answer": None,
            "logprobs": {},
            "distribution": {}
        }
        for _, row in data.iterrows()
    ]

def process_responses(raw_responses, responses, logger):
    """Process and store LLM responses"""
    for i, resp in enumerate(raw_responses):
        if resp is None:
            responses[i]["responses"].append(None)
            responses[i]["raw_response"].append("")
            continue
            
        responses[i]["raw_response"].append(
            resp.content if hasattr(resp, "content") else str(resp)
        )
        try:
            # Extract just the answer from the structured output
            if isinstance(resp, dict) and "parsed" in resp:
                parsed = resp["parsed"]
                responses[i]["responses"].append(parsed.answer if parsed else None)
            else:
                responses[i]["responses"].append(None)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            responses[i]["responses"].append(None)

def save_responses(responses, exp_dir, logger):
    """Save responses to file and display summary"""
    responses_file = os.path.join(exp_dir, "responses.json")
    with open(responses_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved responses to {responses_file}")
    
    # Display summary of responses
    logger.info("=== RESPONSE SUMMARY ===")
    for i, response_data in enumerate(responses):
        logger.info(f"Question {i+1}: {response_data['question'][:80]}...")
        logger.info(f"  Total runs: {len(response_data['responses'])}")
        
        for run_idx, (resp, probs) in enumerate(zip(
            response_data['responses'], 
            response_data['logprobs']  # Now contains actual probabilities
        )):
            logger.info(f"  Run {run_idx+1}: {resp}")
            if isinstance(probs, dict) and probs:
                # Show all options with probabilities, sorted by probability
                sorted_options = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"    Probability rankings:")
                total_prob = sum(probs.values())
                for rank, (option, prob) in enumerate(sorted_options, 1):
                    marker = "👑" if rank == 1 else "  "
                    percentage = (prob / total_prob * 100) if total_prob > 0 else 0
                    logger.info(f"    {marker} {rank}. {option}: {prob:.6f} ({percentage:.1f}%)")
                logger.info(f"    Total probability sum: {total_prob:.6f}")
            else:
                logger.info(f"    No probability data available")
        logger.info("")

def run_evaluation(config, data, exp_dir, logger):
    """Load responses for evaluation"""
    responses_path = os.path.join(config["responses_dir"], "responses.json")
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    
    logger.info(f"Loaded {len(responses)} responses from {responses_path}")
    logger.info("Evaluation step - implement evaluation logic here")
    
    #TODO
    pass

if __name__ == "__main__":
    main()