import os
import platform
from typing import Optional

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

# Default model for local usage - using Llama 3 small model
DEFAULT_MODEL = "meta-llama/Llama-3-7b"  # Update this with actual Llama 3 model name when released

# Token limits - Llama models typically have 4096 context window
MAX_INPUT_LENGTH = 1024  # Reserve space for input
MAX_OUTPUT_LENGTH = 512  # Limit response length
TOTAL_MAX_LENGTH = 2048  # Keep well under model's limit for safety

def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and platform.processor() == 'arm':
        return "mps"  # For M1/M2 Macs
    return "cpu"

def get_llm(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_length: int = TOTAL_MAX_LENGTH
) -> HuggingFacePipeline:
    """
    Initialize and return a LangChain-compatible LLM.
    
    Args:
        model_name: Name of the model to use (default: Llama-3-7b)
        device: Device to run the model on ('cpu', 'cuda', or 'mps'). If None, best device is auto-detected
        max_length: Maximum length of generated text
    
    Returns:
        HuggingFacePipeline: LangChain-compatible LLM
    """
    model_name = model_name or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    device = device or get_device()
    
    print(f"Using device: {device}")
    
    # Initialize tokenizer with truncation settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=TOTAL_MAX_LENGTH,
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )
    
    # Configure model loading based on device
    model_kwargs = {
        "trust_remote_code": True,
        "use_auth_token": True,  # Required for Llama models
    }
    
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["load_in_8bit"] = True
    elif device == "mps":
        model_kwargs["torch_dtype"] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Move model to MPS device if needed
    if device == "mps":
        model = model.to(device)

    # Create text generation pipeline with proper sampling settings
    pipe_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": MAX_OUTPUT_LENGTH,  # Control output length
        "do_sample": True,  # Enable sampling
        "temperature": 0.7,  # Control randomness
        "top_p": 0.95,      # Nucleus sampling
        "repetition_penalty": 1.15,
        "truncation": True,  # Enable truncation
        "return_full_text": False,  # Only return generated text
    }
    
    # Add device configuration for CUDA
    if device == "cuda":
        pipe_kwargs["device"] = 0
    elif device == "cpu":
        pipe_kwargs["device"] = -1
    # For MPS, device is already set by moving the model

    pipe = pipeline(
        "text-generation",
        **pipe_kwargs
    )

    # Create LangChain wrapper
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm
