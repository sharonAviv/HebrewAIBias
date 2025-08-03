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
    returns: laist of strings, each string represents multiple line-seperated answers from a single question.
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
            "answers": answer_string
        })

    return user_inputs

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