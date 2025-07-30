"""
Running LLM inference on survey questions with separate inference/evaluation modes.
"""

import os
import yaml
import json
import argparse
import pandas as pd
from transformers import set_seed
from langchain_core.prompts import ChatPromptTemplate
from utils import (
    get_logger,
    set_keys,
    parse_response,
    get_user_inputs,
    run_batch,
    run_individual
)
from models import get_model, get_schema

def main():
    # Initialize
    logger = get_logger(__name__)
    args = parse_args()
    logger.info(f"Args: {args}")
    
    # Load config and keys
    config = load_config(args)
    set_keys(yaml.safe_load(open("keys.yaml")))
    set_seed(config["seed"])
    
    # Prepare experiment directory
    exp_dir = setup_experiment_dir(config)
    
    # Load data
    data = pd.read_json(config["data_file"])
    if config["debug"]:
        data = data.sample(n=config["num_samples"], random_state=config["seed"])
    
    # Run selected mode
    if config["mode"] == "inference":
        run_inference(config, data, exp_dir, logger)
    else:
        run_evaluation(config, data, exp_dir, logger)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config_llm.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", required=True, choices=["inference", "evaluation"])
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

def setup_experiment_dir(config):
    """Create experiment directory and save config"""
    model_name = config["model"].split("/")[-1]
    exp_name = f"survey_{model_name}_{config['prompt_type']}_tmp{config['temperature']}_seed{config['seed']}"
    exp_dir = os.path.join(config["logs_dir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    return exp_dir

def run_inference(config, data, exp_dir, logger):
    """Run LLM inference and save responses"""
    llm = get_model(
        config["model"],
        config["temperature"],
        config.get("use_rate_limiter", False),
        requests_per_second=config.get("requests_per_second"),
        check_every_n_seconds=config.get("check_every_n_seconds")
    )
    
    # Apply structured output if supported
    schema = get_schema(config["model"], "classification")
    if schema is not None:
        llm = llm.with_structured_output(schema, include_raw=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", config["system_prompt"]),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    user_inputs = get_user_inputs(data)
    responses = initialize_responses(data)
    
    for run_index in range(config.get("sc_runs", 1)):
        logger.info(f"Run {run_index + 1}/{config.get('sc_runs', 1)}")
        
        raw_responses = (
            run_batch(chain, user_inputs, run_index, logger) if config["batched"]
            else run_individual(chain, user_inputs, run_index, exp_dir, logger)
        )
        
        process_responses(raw_responses, responses, logger)
    
    save_responses(responses, exp_dir, logger)

def initialize_responses(data):
    """Initialize response storage structure"""
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "responses": [],
            "raw_response": []
        }
        for _, row in data.iterrows()
    ]

def process_responses(raw_responses, responses, logger):
    """Process and store LLM responses"""
    for i, resp in enumerate(raw_responses):
        if resp is None:
            responses[i]["responses"].append({})
            responses[i]["raw_response"].append("")
            continue
            
        responses[i]["raw_response"].append(
            resp.content if hasattr(resp, "content") else str(resp)
        )
        try:
            responses[i]["responses"].append(parse_response(resp))
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            responses[i]["responses"].append({})

def save_responses(responses, exp_dir, logger):
    """Save responses to file"""
    with open(os.path.join(exp_dir, "responses.json"), "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved responses to {exp_dir}")

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