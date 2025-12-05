import os
import yaml
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from DPO_alignment import H4ArgumentParser
from trl import DPOConfig, ModelConfig, DPOTrainer
# from DPO_utils import MyDPOTrainer

from data import PreferenceSamplerConfig, PreferenceSampler

@dataclass
class ScriptArguments:
    ref_model: Optional[str] = field(default="")
    reward_model: Optional[str] = field(default="")
    train_dir: Optional[str] = field(default="")
    eval_dir: Optional[str] = field(default="")

    device: Optional[str] = field(default="cuda:1")
    ref_device: Optional[int] = field(default=1)
    rm_device: Optional[int] = field(default=1)

    eos_padding: Optional[bool] = field(default=True)
    margin_scale: Optional[float] = field(default=1.0)
    sanity_check: Optional[bool] = field(default=False)
    max_training_samples: Optional[int] = field(default=-1)
    choose_type: Optional[str] = field(default="max_random")
    ignore_bias_buffers: Optional[bool] = field(default=False)
    eot_token: Optional[str] = field(default="<|eot_id|>")
    len_penalty: Optional[float] = field(default=0.0)

    # === PreferenceSamplerConfig fields ===
    samples_drawn_size: int = field(default=20)
    num_return_sequences: int = field(default=4)
    generation_seed: Optional[int] = field(default=42)
    dataset_dir: str = field(default="RLHFlow/ultrafeedback_iter1")
    ref_gpu_utlization: float = field(default=0.5)
    rm_batch_size: int = field(default=8)

    # Corruption Params
    corrupt_preferences: bool = field(default=True)
    corruption_percentage: float = field(default=0.2)
    corruption_seed: int = field(default=42)


    def __post_init__(self):
        if self.corrupt_preferences:
            if self.corrupt_preferences > 1.0 or self.corrupt_preferences < 0.0:
                self.corrupt_preferences = 0.2  


def corrupt_data(pos: list, neg: list, prompts: list, margin: list, args: ScriptArguments) -> tuple[list, list, list, list]:
    """
    Advaserially flip labels of some percentage of examples.
    """
    n = len(prompts)
    corrupted_idx = int(args.corruption_percentage * n)
    np.random.seed(args.corruption_seed)
    permuted_indices = np.random.permutation(np.arange(n))[:corrupted_idx]

    for idx in permuted_indices:
        pos[idx], neg[idx], margin[idx] = neg[idx], pos[idx], -margin[idx]

    return pos, neg, prompts, margin


def prepare_data(
    args: ScriptArguments,
    samples_preference_pair: dict,
    sanity_check: bool = False,
    margin_scale=1,
    eot_token="",
    is_train=True
) -> Dataset:

    pos = []
    neg = []
    prompts = []
    margin = []
    
    for prompt_key, pairs in samples_preference_pair.items():
        prompts.append(prompt_key)
        best_resp = pairs["best_response"].removeprefix(prompt_key)
        worst_resp = pairs["worst_response"].removeprefix(prompt_key)
        best_reward = float(pairs["best_reward"])
        worst_reward = float(pairs["worst_reward"])
        pos.append(best_resp)
        neg.append(worst_resp)
        margin.append((best_reward - worst_reward) * margin_scale)

    if args.corrupt_preferences and is_train:
        pos, neg, prompts, margin = corrupt_data(pos, neg, prompts, margin, args)
            
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


if __name__ == "__main__":
    parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse()

    training_args.learning_rate = float(training_args.learning_rate)
    
    if model_config.model_name_or_path is None:
            model_config.model_name_or_path = script_args.ref_model

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation="flash_attention_2",
        dtype=torch.float16,)
    # ).to(script_args.device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = model_config.model_name_or_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",)
    # ).to(script_args.device)

    model_ref.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.vocab_size += 1
        model_ref.config.vocab_size += 1
        model.config.pad_token_id = tokenizer.pad_token_id
        model_ref.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model_ref.resize_token_embeddings(len(tokenizer))
    
    pref_config = PreferenceSamplerConfig(
        samples_drawn_size=script_args.samples_drawn_size,
        num_return_sequences=script_args.num_return_sequences,
        generation_seed=script_args.generation_seed,
        dataset_dir=script_args.dataset_dir,
        ref_model_path=script_args.ref_model,
        ref_device=script_args.ref_device,
        ref_gpu_utlization=script_args.ref_gpu_utlization,
        rm_path=script_args.reward_model,
        rm_batch_size=script_args.rm_batch_size,
        rm_device=script_args.rm_device,
    )
    pref_sampler = PreferenceSampler(config=pref_config)

    import json

    # TODO: Sampling needs to be done outside this script. We are using accelerate to do DPO training

    # outputs_policy_1, outputs_policy_2 = pref_sampler.generate_responses()
    # pref_pairs = pref_sampler.rejection_sampling(outputs_policy_1, outputs_policy_2)

    with open('preference_samples.json') as f:
        pref_pairs = json.load(f)

    items = list(pref_pairs.items())[:2100]

    train_pref_pairs = {k: v for (k, v) in items[:2000]}
    eval_pref_pairs = {k: v for (k, v) in items[2000:]}

    train_dataset = prepare_data(
        args=script_args,
        samples_preference_pair=train_pref_pairs,
        sanity_check=script_args.sanity_check,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
        is_train=True
    )
    
    eval_dataset = prepare_data(
        args=script_args,
        samples_preference_pair=eval_pref_pairs,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
        is_train=False
    )
    
    # eval_dataset = train_dataset.select(range(min(len(train_dataset), 100)))

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = os.getcwd()
    os.environ["WANDB_PROJECT"] = "dpo_offline_run"

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
        # max_length=training_args.max_length,
        # max_prompt_length=training_args.max_prompt_length,
        # loss_type=training_args.loss_type,
    )
    print("begin to train")

    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)