####################################################################################################
# Imports
import os
import torch
import torch.nn.functional as F

from langchain.chains import LLMChain

from langchain_openai import ChatOpenAI
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict  
import sys
sys.path.append("../") 
from enum import Enum




###############################################################################
# Schemas

class SurveyResponse(BaseModel):
    answer: str
    # Remove allowed_answers from fields since we'll pass it via context
    # Add a Config class to handle arbitrary types during validation
    class Config:
        arbitrary_types_allowed = True

    @field_validator("answer")
    def validate_answer(cls, v: str, info: ValidationInfo) -> str:
        # Get allowed_answers from validation context
        allowed_answers = info.context.get("allowed_answers")
        if not allowed_answers:
            raise ValueError("Allowed answers not provided in context")
        
        # Format allowed answers as "1. Yes", "2. No", etc.
        allowed_formatted = {
            f"{num}. {text}" for num, text in allowed_answers.items()
        }
        
        if v not in allowed_formatted:
            raise ValueError(
                f"Invalid answer. Must be one of: {sorted(allowed_formatted)}"
            )
        return v

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


def is_openai_remote_model(model_name: str, local_model_id: str = None) -> bool:
    """
    Check if a model is OpenAI-based for logprob handling.
    :param model_name: The model name from config.
    :param local_model_id: Override model ID for local models.
    :return: True if OpenAI model, False otherwise.
    """
    if local_model_id:
        return False  # Local models are not OpenAI
    
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]["provider"] == "openai"
    
    return False


def extract_logprobs_from_local_model(model, tokenizer, prompt, multiple_choice_options, schema):
    """
    Extract log probabilities for multiple choice options from local model.
    Uses conditional probability: P(option | prompt) for each option.
    :param model: The HuggingFace model instance.
    :param tokenizer: The tokenizer instance.
    :param prompt: The input prompt string.
    :param multiple_choice_options: List of multiple choice answer strings.
    :return: Dictionary mapping options to log probabilities.
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


def extract_logprobs(response, is_openai: bool = True):
    """
    Extract log probabilities from model response.
    OpenAI uses 'logprobs' in response_metadata, others use different formats.
    :param response: The model response.
    :param is_openai: Whether this is an OpenAI model.
    :return: Extracted logprobs or None.
    """
    try:
        if is_openai:
            # OpenAI format: response_metadata["logprobs"]["content"]
            if hasattr(response, 'response_metadata') and 'logprobs' in response.response_metadata:
                return response.response_metadata["logprobs"]["content"]
        else:
            # Non-OpenAI models might store logprobs differently
            # This is a placeholder - adjust based on actual model output format
            if hasattr(response, 'logprobs'):
                return response.logprobs
            elif hasattr(response, 'response_metadata') and 'logprobs' in response.response_metadata:
                return response.response_metadata['logprobs']
    except Exception as e:
        print(f"Error extracting logprobs: {e}")
    
    return None



def get_model(model_name: str, temperature: float = 0.0, use_rate_limiter: bool = False, requests_per_second=0.8, check_every_n_seconds=1.5, local_model_id: str = None):
    """
    Get the model based on the model name.
    :param model_name: The name of the model to get.
    :param temperature: The temperature for the model.
    :param use_rate_limiter: Whether to use a rate limiter for the model.
    :param local_model_id: Override model ID for local models (from YAML config).
    :return: For local models: (llm, model, tokenizer) tuple. For OpenAI: just the llm.
    """
    if use_rate_limiter:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_every_n_seconds,  # Wake up every X ms to check whether allowed to make a request,
            max_bucket_size=1,  # Controls the maximum burst size.
        )
    else:
        rate_limiter = None

    # If local_model_id is provided, use local model regardless of MODEL_CONFIGS
    if local_model_id:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Check GPU availability and configure device
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} GPU(s)")
            device_map = "auto"
            torch_dtype = torch.float16
            # Only set CUDA_VISIBLE_DEVICES if you want to restrict to specific GPUs
            # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only
        else:
            logger.warning("CUDA not available, using CPU")
            device_map = None
            torch_dtype = torch.float32
        
        logger.info(f"Loading model: {local_model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        # Load model with proper device handling
        model = AutoModelForCausalLM.from_pretrained(
            local_model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        logger.info(f"Model loaded successfully on device: {model.device if hasattr(model, 'device') else 'multiple devices'}")
        
        # Create HuggingFace pipeline with same device settings
        llm = HuggingFacePipeline.from_model_id(
            model_id=local_model_id,
            task="text-generation",
            pipeline_kwargs={
                "temperature": temperature, 
                "do_sample": True, 
                "max_new_tokens": 4096,
                "torch_dtype": torch_dtype
            },
            batch_size=BATCH_SIZE,
            device_map=device_map,
        )
        logger.info("HuggingFace pipeline created successfully")
        return llm, model, tokenizer
    
    # Fallback to existing MODEL_CONFIGS logic
    assert model_name in MODEL_CONFIGS, f"Model {model_name} not found in MODEL_CONFIGS"
    model_configs = MODEL_CONFIGS[model_name]
    model_id = model_configs["model_id"]
    provider = model_configs["provider"]

    if provider == "local":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Check GPU availability and configure device
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} GPU(s)")
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            logger.warning("CUDA not available, using CPU")
            device_map = None
            torch_dtype = torch.float32
        
        logger.info(f"Loading model: {model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        # Load model with proper device handling
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        logger.info(f"Model loaded successfully on device: {model.device if hasattr(model, 'device') else 'multiple devices'}")

        # Create HuggingFace pipeline with same device settings
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={
                "temperature": temperature, 
                "do_sample": True, 
                "max_new_tokens": 4096,
                "torch_dtype": torch_dtype
            },
            batch_size=BATCH_SIZE,
            device_map=device_map,
        )
        logger.info("HuggingFace pipeline created successfully")
        return llm, model, tokenizer
    elif provider == "openai":
        return ChatOpenAI(model=model_id, temperature=temperature, logprobs=True, top_logprobs=4)
    else:
        raise ValueError(f"Unknown model {model_name}")
