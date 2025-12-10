import gc
from dataclasses import dataclass, field
import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

@dataclass
class UncertaintyConfig:
    ref_model_path: str = field(metadata={'help': 'Model path for which we want to calculate the uncertainty from.'})
    uncertainty_device: int = field(default=0, metadata={'help': 'GPU device where the model will be loaded.'})
    first_k: int = field(default=0, metadata={'help': 'Calculate the uncertainty of first k pairs. For k <= 0, uncertainty for all samples is computed.'})


def get_uncertainity_scores(preference_pairs: dict[str: dict[str: str]], config: UncertaintyConfig) -> np.ndarray:
    """
    TODO: Improve the throughput for higher batches of responses. Right now it is just a pair of responses.
    """
    model = LlamaForCausalLM.from_pretrained(config.ref_model_path, device_map=f'cuda:{config.uncertainty_device}', dtype=torch.bfloat16)
    model.requires_grad_(False)
    
    if model.generation_config.temperature is None:
        model.generation_config.temperature = 1.0

    # already set
    model.generation_config.do_sample = True

    tokenizer = AutoTokenizer.from_pretrained(config.ref_model_path)

    model.set_attn_implementation('eager')

    samples_attn_scores = []

    first_k = min(config.first_k, len(preference_pairs)) if config.first_k > 0 else len(preference_pairs)

    pbar_prefernces = tqdm(list(preference_pairs.items())[:first_k])

    for idx, (prompt, pairs) in enumerate(pbar_prefernces, start=1):
        pbar_prefernces.set_description(f'Prompt {idx} uncertainty')
        best_response, worst_response = pairs['best_response'], pairs['worst_response']

        inputs = tokenizer([best_response, worst_response], return_tensors='pt', truncation=True, max_length=2048, padding=True)

        kwargs = {
            "input_ids": inputs['input_ids'].to(model.device),
            "attention_mask": inputs['attention_mask'].to(model.device),
            "use_cache": False,
            "past_key_values": None,
            "output_attentions": True,
            "output_hidden_states": False,
            "return_dict": True,
        }

        with torch.no_grad():
            output = model(**kwargs)

        attns = output.attentions

        # No squeezing
        attn_ = [x.to(torch.float32).detach().cpu() for x in attns]

        prompt_len = len(tokenizer(prompt)['input_ids'])
        mask_sum = inputs['attention_mask'].sum(dim=-1)

        is_single_sequence = not mask_sum.shape

        if is_single_sequence:
            # mask sum is a scalar.
            span_idx = [[prompt_len, mask_sum]]
        else:
            span_idx = [[prompt_len, total_len] for total_len in mask_sum]

        sample_attn_scores = []

        for sample_idx in range(len(span_idx)):
            layers_attn_scores = []
            for block_idx in range(len(attns)):
                eigen_score = 0.0
                for head_idx in range(1, len(attn_[block_idx][sample_idx])):
                    # attn_ is a tuple of attention matrices. With size Batch x head x Sequence len x Sequence len
                    _KER = attn_[block_idx][sample_idx][head_idx]
                    start, end = span_idx[sample_idx]
                    KER = _KER[start: end, start: end]

                    eigen_score += KER.diagonal().log().mean()
                layers_attn_scores.append(eigen_score.item())

            sample_attn_scores.append(layers_attn_scores)

        samples_attn_scores.append(sample_attn_scores)

        torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return np.array(samples_attn_scores)


def get_uncertainity_scores_efficient_v1(preference_pairs: dict[str: dict[str: str]], config: UncertaintyConfig) -> np.ndarray:
    """
    TODO: Improve the throughput for higher batches of responses. Right now it is just a pair of responses.
    """
    model = LlamaForCausalLM.from_pretrained(config.ref_model_path, device_map=f'cuda:{config.uncertainty_device}', dtype=torch.bfloat16)
    model.requires_grad_(False)
    
    if model.generation_config.temperature is None:
        model.generation_config.temperature = 1.0

    # already set
    model.generation_config.do_sample = True

    tokenizer = AutoTokenizer.from_pretrained(config.ref_model_path)

    model.set_attn_implementation('eager')

    samples_attn_scores = []

    first_k = min(config.first_k, len(preference_pairs)) if config.first_k > 0 else len(preference_pairs)

    pbar_prefernces = tqdm(list(preference_pairs.items())[:first_k])

    for idx, (prompt, pairs) in enumerate(pbar_prefernces, start=1):
        pbar_prefernces.set_description(f'Prompt {idx} uncertainty')
        best_response, worst_response = pairs['best_response'], pairs['worst_response']

        inputs = tokenizer([best_response, worst_response], return_tensors='pt', truncation=True, max_length=2048, padding=True)

        kwargs = {
            "input_ids": inputs['input_ids'].to(model.device),
            "attention_mask": inputs['attention_mask'].to(model.device),
            "use_cache": False,
            "past_key_values": None,
            "output_attentions": True,
            "output_hidden_states": False,
            "return_dict": True,
        }

        with torch.no_grad():
            output = model(**kwargs)

        attns = output.attentions

        # No squeezing
        attn_ = [x.to(torch.float32).detach().cpu() for x in attns]

        prompt_len = len(tokenizer(prompt)['input_ids'])
        mask_sum = inputs['attention_mask'].sum(dim=-1)

        is_single_sequence = not mask_sum.shape

        if is_single_sequence:
            # mask sum is a scalar.
            span_idx = [[prompt_len, mask_sum]]
        else:
            span_idx = [[prompt_len, total_len] for total_len in mask_sum]

        sample_attn_scores = []

        for sample_idx in range(len(span_idx)):
            layers_attn_scores = []
            for block_idx in range(len(attns)):
                eigen_score = 0.0
                for head_idx in range(1, len(attn_[block_idx][sample_idx])):
                    # attn_ is a tuple of attention matrices. With size Batch x head x Sequence len x Sequence len
                    _KER = attn_[block_idx][sample_idx][head_idx]
                    start, end = span_idx[sample_idx]
                    KER = _KER[start: end, start: end]

                    eigen_score += KER.diagonal().log().mean()
                layers_attn_scores.append(eigen_score.item())

            sample_attn_scores.append(layers_attn_scores)

        samples_attn_scores.append(sample_attn_scores)

        torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return np.array(samples_attn_scores)


def get_uncertainity_scores_efficient_v2(preference_pairs: dict[str: dict[str: str]], config: UncertaintyConfig) -> np.ndarray:
    """
    TODO: Improve the throughput for higher batches of responses. Right now it is just a pair of responses.
    """
    model = LlamaForCausalLM.from_pretrained(config.ref_model_path, device_map=f'cuda:{config.uncertainty_device}', dtype=torch.bfloat16)
    model.requires_grad_(False)
    
    if model.generation_config.temperature is None:
        model.generation_config.temperature = 1.0

    # already set
    model.generation_config.do_sample = True

    tokenizer = AutoTokenizer.from_pretrained(config.ref_model_path)

    model.set_attn_implementation('eager')

    samples_attn_scores = []

    first_k = min(config.first_k, len(preference_pairs)) if config.first_k > 0 else len(preference_pairs)

    pbar_prefernces = tqdm(list(preference_pairs.items())[:first_k])

    for idx, (prompt, pairs) in enumerate(pbar_prefernces, start=1):
        pbar_prefernces.set_description(f'Prompt {idx} uncertainty')
        best_response, worst_response = pairs['best_response'], pairs['worst_response']

        inputs = tokenizer([best_response, worst_response], return_tensors='pt', truncation=True, max_length=2048, padding=True)

        kwargs = {
            "input_ids": inputs['input_ids'].to(model.device),
            "attention_mask": inputs['attention_mask'].to(model.device),
            "use_cache": False,
            "past_key_values": None,
            "output_attentions": True,
            "output_hidden_states": False,
            "return_dict": True,
        }

        with torch.no_grad():
            output = model(**kwargs)

        attns = output.attentions

        # No squeezing
        attn_ = torch.stack([x.to(torch.float32).detach().cpu() for x in attns])

        prompt_len = len(tokenizer(prompt)['input_ids'])
        mask_sum = inputs['attention_mask'].sum(dim=-1)

        is_single_sequence = not mask_sum.shape

        if is_single_sequence:
            # mask sum is a scalar.
            span_idx = [[prompt_len, mask_sum]]
        else:
            span_idx = [[prompt_len, total_len] for total_len in mask_sum]

        sample_attn_scores = []

        for sample_idx in range(len(span_idx)):
            # layers_attn_scores = []
            # attn_ -> layers x samples x heads x seq. len x seq. len
            start, end = span_idx[sample_idx]
            # sample_KER -> layers x heads x span x span
            sample_KER = attn_[:, sample_idx, :, start: end, start: end]
            # log det mean over tokens (after taking diagonal, it's the last dim) and then sum over heads (2nd dim from the left side).
            layers_attn_scores = torch.diagonal(sample_KER, dim1=-2, dim2=-1).log().mean(dim=-1).sum(dim=1)
            sample_attn_scores.append(layers_attn_scores)

        samples_attn_scores.append(sample_attn_scores)

        torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return np.array(samples_attn_scores)
