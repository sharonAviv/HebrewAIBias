####################################################################################################
# Imports
import os
import logging
from ast import literal_eval
import pandas as pd
from pydantic import BaseModel, Field
from pandas import DataFrame
import json
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


MERGE_COLUMNS = ["debug", "model", "prompt_type", "seed", "sc_runs", "temperature", "shots"]

class Label(BaseModel):
    """Label"""
    label: str = Field(
        description="The predicted label for the classification task"
    )
    
    
####################################################################################################
# Functions

def set_keys(keys: dict):
    """
    Set API keys as environment variables.
    :param keys: dictionary with keys
    """
    for key, value in keys.items():
        os.environ[key] = value


def get_logger(name):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        encoding="utf-8",
    )
    logger = logging.getLogger(name)
    return logger


def run_batch(chain, user_inputs, run_index, logger):
    """Execute batch processing."""
    try:
        raw_responses = chain.batch(user_inputs)
        logger.info(f"Generated {len(raw_responses)} raw responses for run {run_index}")
        return raw_responses
    except Exception as e:
        logger.error(f"Error during batch run {run_index}: {e}")
        return [None] * len(user_inputs)
    
PROMPTS = {
    "system": """You will be asked a multiple-choice question. Return the answer you most agree with as-is (for example, "1. Dogs"). Do not include ANY additional text or explanations.\n""",
    "user": "Question: {question}\nMultiple-choice answers:\n{answers}"
}

def get_answers(data: pd.DataFrame) -> list[dict]:
    """
    Get the multiple choice answers for each question in the survey.
    param data: data with 'question' and 'answers' columns
    returns: list of dicts, each containing question, answers string, and allowed_answers
    """
    user_inputs = []

    for _, row in data.iterrows():
        answer_string = "\n".join(
            f"{key}. {value}" 
            for key, value in (
                (k, v.replace('\n', ' ')) 
                for k, v in row['answers'].items() 
                if k != '99'
            )
        )
        
        user_inputs.append({
            "question": row["question"],
            "answers": answer_string,
            "allowed_answers": {k: v for k, v in row['answers'].items() if k != '99'}
        })

    return user_inputs

def get_answers(data: pd.DataFrame, variant: str) -> list[dict]:
    """Get the multiple choice answers for each question in the survey.
    If experiment variant is 'refusal', an additional answer 'I prefer not to answer' is provided."""
    user_inputs = []
    
    for _, row in data.iterrows():
        answers = row["answers"]

        refusal_key = len(answers) + 1
        answers_with_refusal = {**answers, refusal_key: "I prefer not to answer"}
        
        formatted_answers = "\n".join([f"{k}. {v}" for k, v in answers_with_refusal.items()])
        
        user_inputs.append({
            "question": row["question"],
            "answers": formatted_answers,
            "allowed_answers": answers_with_refusal if variant == "with_refusal" else answers,
            "refusal_key": refusal_key
        })
    
    return user_inputs

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
        "refusal_probability": refusal_prob,
        "choice_probabilities": choice_probs,
        "raw_logprobs": choice_logprobs
    }
    
def parse_response(resp):
    """Normalize outputs from structured/unstructured calls."""
    # If using structured output with include_raw=True, resp is a dict
    if isinstance(resp, dict) and "parsed" in resp:
        parsed = resp["parsed"]
        try:
            return parsed.model_dump()  # pydantic v2
        except Exception:
            try:
                return parsed.dict()     # pydantic v1 fallback
            except Exception:
                return parsed
    # Fallback: return an empty dict; you can extend if you keep unstructured LLMs
    return {}

def json_file_to_dataframe(file_path):
    """
    Read JSON survey data and convert to DataFrame with answers as lists of tuples.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        pd.DataFrame: Each question as one row with answers in a list column
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    rows = []
    
    for item in json_data:
        answers = item.get('answers', {})
        distributions = item.get('distribution', {})
        
        # Prepare answer tuples (code, text, distribution)
        answer_list = [
            (
                code,
                answers.get(code, "REFUSE TO ANSWER" if code == '99' else None),
                dist
            )
            for code, dist in distributions.items()
        ]
        
        rows.append({
            'id': item['id'],
            'question': item['question'],
            'institute': item['institute'],
            'survey': item['survey'],
            'survey_qid': item['survey_qid'],
            'date': item['date'],
            'file': item['file'],
            'answers': answer_list  # List of (code, text, distribution) tuples, excluding refusal to answer
        })
    
    return pd.DataFrame(rows)