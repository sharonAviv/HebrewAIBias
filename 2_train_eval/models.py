####################################################################################################
# Imports
import os

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field, field_validator 
from typing import Optional, List





###############################################################################
# Schemas

class SurveyResponse(BaseModel):
    answer: str
    allowed_answers: dict  # Format: {"1": "Yes", "2": "No", ...}

    @field_validator("answer")
    def validate_answer(cls, v: str, values: dict) -> str:
        if "allowed_answers" not in values:
            raise ValueError("Allowed answers not provided")
        
        # Format allowed answers as "1. Yes", "2. No", etc.
        allowed_formatted = {
            f"{num}. {text}" for num, text in values["allowed_answers"].items()
        }
        
        if v not in allowed_formatted:
            raise ValueError(
                f"Invalid answer. Must be one of: {sorted(allowed_formatted)}"
            )
        return v


import sys
sys.path.append("../") 

####################################################################################################

####################################################################################################
# Constants
BATCH_SIZE = 16

MODEL_CONFIGS = {
    "gpt-4o-mini": {
        "display_name": "GPT-4o-mini",
        "model_id": "gpt-4o-mini", 
        "structured_output": "pydantic", 
        "provider": "openai", 
    },
    # "EMMA-500-7B": {
    #     "display_name": "EMMA-500-7B",
    #     "model_id": "MaLA-LM/emma-500-llama2-7b",
    #     "structured_output": None,
    #     "provider": "local",
    # },
    # "LLaMAX-3-8B": {
    #     "display_name": "LLaMAX-3-8B",
    #     "model_id": "LLaMAX/LLaMAX3-8B",
    #     "structured_output": None,
    #     "provider": "local",
    # }
}

####################################################################################################

####################################################################################################
# Models

def get_schema(model_name: str):
    print(f"Getting schema for model {model_name}")
    structured_output = MODEL_CONFIGS[model_name].get("structured_output", None)
    if structured_output is None:
        return None
    if structured_output == "pydantic":
        return SurveyResponse
    else:
        # Inform the user and return None
        print(f"Model {model_name} does not support structured output.")
        return None



def get_model(model_name: str, temperature: float = 0.0, use_rate_limiter: bool = False, requests_per_second=0.8, check_every_n_seconds=1.5) -> LLMChain:
    """
    Get the model based on the model name.
    :param model_name: The name of the model to get.
    :param temperature: The temperature for the model.
    :param use_rate_limiter: Whether to use a rate limiter for the model.
    :return: The model instance as an LLMChain.
    """
    assert model_name in MODEL_CONFIGS, f"Model {model_name} not found in MODEL_CONFIGS"
    if use_rate_limiter:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_every_n_seconds,  # Wake up every X ms to check whether allowed to make a request,
            max_bucket_size=1,  # Controls the maximum burst size.
        )
    else:
        rate_limiter = None

    model_configs = MODEL_CONFIGS[model_name]
    model_name = model_configs["model_id"]
    provider = model_configs["provider"]

    if provider == "local":
        os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs={"temperature": temperature, "do_sample":True, "max_new_tokens": 4096},
            #model_kwargs={"token": access_token_read},
            batch_size=BATCH_SIZE,
            device_map="auto",
        )
        return llm
    elif provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature, logprobs=True, top_logprobs=4)
    else:
        raise ValueError(f"Unknown model {model_name}")
