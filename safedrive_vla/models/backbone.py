"""InternVL3 VLM backbone (Hugging Face port) and vocabulary extension."""

from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from safedrive_vla.action_tokenizer import action_token_strings
from safedrive_vla.constants import SPECIAL_TOKENS


def resolve_model_id(variant: str) -> str:
    """``OpenGVLab/InternVL3-1B`` -> its Hugging Face port ``OpenGVLab/InternVL3-1B-hf``."""
    return variant if variant.endswith("-hf") else f"{variant}-hf"


def extend_vocabulary(tokenizer, num_action_tokens: int) -> int:
    """Add the special tokens, then the contiguous action tokens; returns the
    token id of ``<action_0>``. The call order fixes the token ids."""
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    action_tokens = action_token_strings(num_action_tokens)
    tokenizer.add_tokens(action_tokens, special_tokens=False)
    first = tokenizer.convert_tokens_to_ids(action_tokens[0])
    last = tokenizer.convert_tokens_to_ids(action_tokens[-1])
    if last - first + 1 != num_action_tokens:
        raise RuntimeError("Action tokens must be contiguous in the vocabulary")
    return int(first)


def load_processor(variant: str, num_action_tokens: int) -> Tuple[object, int]:
    """Processor with the extended vocabulary, and the id of ``<action_0>``."""
    processor = AutoProcessor.from_pretrained(resolve_model_id(variant), trust_remote_code=True)
    action_start_id = extend_vocabulary(processor.tokenizer, num_action_tokens)
    processor.tokenizer.padding_side = "left"
    return processor, action_start_id


def load_vlm(variant: str, dtype: torch.dtype = torch.bfloat16):
    model_id = resolve_model_id(variant)
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_id, config=hf_config, trust_remote_code=True, dtype=dtype)
    return model, hf_config


def lm_hidden_size(hf_config) -> int:
    if isinstance(getattr(hf_config, "hidden_size", None), int):
        return int(hf_config.hidden_size)
    return int(hf_config.text_config.hidden_size)
