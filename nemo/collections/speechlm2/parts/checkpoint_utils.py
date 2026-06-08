# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
from peft import PeftModel

from nemo.utils import logging


def validate_model_components(model) -> Tuple[bool, str]:
    """Validate that all critical model components are properly loaded and initialized.

    Args:
        model: SALM model instance to validate

    Returns:
        Tuple[bool, str]: (is_valid, message) where is_valid is True if all checks pass
    """
    issues = []

    # Check LLM
    if not hasattr(model, 'llm') or model.llm is None:
        issues.append("LLM component not found or is None")
    else:
        try:
            if not hasattr(model.llm, 'config'):
                issues.append("LLM missing config attribute")
            if not hasattr(model.llm, 'parameters'):
                issues.append("LLM missing parameters")
        except Exception as e:
            issues.append(f"Error checking LLM: {e}")

    # Check embeddings
    if not hasattr(model, 'embed_tokens') or model.embed_tokens is None:
        issues.append("Embedding tokens not found or is None")
    else:
        try:
            if model.embed_tokens.num_embeddings == 0:
                issues.append("Embedding vocab size is 0")
        except Exception as e:
            issues.append(f"Error checking embeddings: {e}")

    # Check perception (audio encoder)
    if not hasattr(model, 'perception') or model.perception is None:
        issues.append("Audio perception module not found or is None")
    else:
        try:
            if not hasattr(model.perception, 'encoder'):
                issues.append("Perception missing encoder")
            if not hasattr(model.perception, 'preprocessor'):
                issues.append("Perception missing preprocessor")
        except Exception as e:
            issues.append(f"Error checking perception: {e}")

    # Check tokenizer
    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        issues.append("Tokenizer not found or is None")

    # Check LoRA state consistency
    has_lora_config = hasattr(model, 'cfg') and 'lora' in model.cfg
    is_peft = isinstance(model.llm, PeftModel)

    if has_lora_config and not is_peft:
        issues.append("Model config has LoRA but LLM is not a PeftModel")
    elif is_peft and not has_lora_config:
        issues.append("LLM is a PeftModel but model config has no LoRA settings")

    # Return results
    is_valid = len(issues) == 0
    message = "Model validation passed" if is_valid else "Model validation failed: " + "; ".join(issues)
    return is_valid, message


def check_model_weights(model, sample_size: int = 5) -> Tuple[bool, str]:
    """Check that model parameters are properly initialized (not NaN or Inf).

    Args:
        model: SALM model instance
        sample_size: Number of parameters to sample for checking

    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    issues = []

    # Collect parameters
    params = list(model.parameters())
    if len(params) == 0:
        return False, "Model has no parameters"

    # Sample parameters to check
    import random

    sample_indices = random.sample(range(len(params)), min(sample_size, len(params)))
    for idx in sample_indices:
        param = params[idx]
        if torch.isnan(param).any():
            issues.append(f"Parameter {idx} contains NaN values")
            break
        if torch.isinf(param).any():
            issues.append(f"Parameter {idx} contains Inf values")
            break

    is_valid = len(issues) == 0
    message = "Weight check passed" if is_valid else "Weight check failed: " + "; ".join(issues)
    return is_valid, message


def validate_fine_tuned_checkpoint(model) -> bool:
    """Comprehensive validation for fine-tuned SALM models.

    Args:
        model: SALM model instance

    Returns:
        bool: True if all validations pass
    """
    all_valid = True

    # Component validation
    valid, msg = validate_model_components(model)
    logging.info(f"Component validation: {msg}")
    all_valid = all_valid and valid

    # Weight validation
    valid, msg = check_model_weights(model)
    logging.info(f"Weight validation: {msg}")
    all_valid = all_valid and valid

    # LoRA-specific validation
    if isinstance(model.llm, PeftModel):
        if not hasattr(model.llm, 'peft_config') or len(model.llm.peft_config) == 0:
            logging.error("PeftModel has no adapter configurations")
            all_valid = False
        else:
            adapter_names = list(model.llm.peft_config.keys())
            logging.info(f"LoRA adapters loaded: {adapter_names}")
            active_adapter = model.llm.active_adapters if hasattr(model.llm, 'active_adapters') else 'unknown'
            logging.info(f"Active adapter: {active_adapter}")

    return all_valid


def get_model_info(model) -> dict:
    """Get comprehensive information about loaded model.

    Args:
        model: SALM model instance

    Returns:
        dict: Dictionary containing model information
    """
    info = {
        'has_lora': isinstance(model.llm, PeftModel),
        'vocab_size': model.text_vocab_size if hasattr(model, 'text_vocab_size') else 'unknown',
        'model_dtype': next(model.parameters()).dtype if len(list(model.parameters())) > 0 else 'unknown',
        'device': next(model.parameters()).device if len(list(model.parameters())) > 0 else 'unknown',
        'num_parameters': sum(p.numel() for p in model.parameters()),
    }

    if info['has_lora'] and hasattr(model.llm, 'peft_config'):
        info['lora_adapters'] = list(model.llm.peft_config.keys())

    return info
