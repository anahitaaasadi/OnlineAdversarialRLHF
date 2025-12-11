import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from DPO_alignment import H4ArgumentParser
from trl import DPOConfig, ModelConfig, DPOTrainer
from trl.data_utils import maybe_apply_chat_template
# from DPO_utils import MyDPOTrainer

from data import PreferenceSamplerConfig, PreferenceSampler
from uncertainty import UncertaintyConfig, get_uncertainity_scores

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
    samples_drawn_size: int = field(default=2100)
    train_samples_drawn_size: int = field(default=2000)
    num_return_sequences: int = field(default=4)
    generation_seed: Optional[int] = field(default=42)
    dataset_dir: str = field(default="RLHFlow/ultrafeedback_iter1")
    ref_gpu_utlization: float = field(default=0.5)
    rm_batch_size: int = field(default=8)
    sample_save_dir: str = field(default='samples')

    # Corruption Params
    corrupt_preferences: bool = field(default=True)
    corruption_percentage: float = field(default=0.2)
    corruption_seed: int = field(default=42)

    # Mitigation params
    mitigate_corruption: bool = field(default=False)


    def __post_init__(self):
        if self.corrupt_preferences:
            if self.corrupt_preferences > 1.0 or self.corrupt_preferences < 0.0:
                self.corrupt_preferences = 0.2  


def corrupt_data(pos: list, neg: list, prompts: list, margin: list, args: ScriptArguments) -> tuple[list, list, list, list]:
    n = len(prompts)
    corrupted_idx = int(args.corruption_percentage * n)
    np.random.seed(args.corruption_seed)
    permuted_indices = np.random.permutation(np.arange(n))[:corrupted_idx]

    for idx in permuted_indices:
        pos[idx], neg[idx], margin[idx] = neg[idx], pos[idx], -margin[idx]

    return pos, neg, prompts, margin


def corrupt_scores(scores: np.ndarray, args: ScriptArguments) -> np.ndarray:
    n = scores.shape[0]
    corrupted_idx = int(args.corruption_percentage * n)
    np.random.seed(args.corruption_seed)
    permuted_indices = np.random.permutation(np.arange(n))[:corrupted_idx]
    # Make deep copies.
    accepted_scores = np.array(scores[permuted_indices, 0, :])
    rejected_scores = np.array(scores[permuted_indices, 1, :])

    corrupted_scores = np.array(scores)
    corrupted_scores[permuted_indices, 0, :] = rejected_scores
    corrupted_scores[permuted_indices, 1, :] = accepted_scores

    return corrupted_scores


def attack_mitigation(pos: list, neg: list, prompts: list, margin: list, corrupted_scores: list) -> tuple[list, list, list, list]:
    threshold = 1.01
    for idx in range(len(prompts)):
        score = corrupted_scores[idx]

        threshold_bool = (score[0, :] / score[1, :]).mean(axis=-1) <= threshold
        inv_threshold_bool = (score[0, :] / score[1, :]).mean(axis=-1) >= 1/threshold

        should_flip = threshold_bool and not inv_threshold_bool

        if should_flip:
            pos[idx], neg[idx], margin[idx] = neg[idx], pos[idx], -margin[idx]

    return pos, neg, prompts, margin



def prepare_data(
    args: ScriptArguments,
    samples_preference_pair: dict,
    sanity_check: bool = False,
    margin_scale=1,
    eot_token="",
    is_train=True,
    uncertainty_scores: np.ndarray | None = None
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

        if args.mitigate_corruption and uncertainty_scores is not None:
            corrupted_scores = corrupt_scores(scores, args)
            pos, neg, prompts, margin = attack_mitigation(pos, neg, prompts, margin, corrupted_scores)
            
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


def load_data(args: ScriptArguments) -> tuple[dict, dict]:
    train_samples = {}
    eval_samples = {}
    for sample_fp in sorted([os.path.join('samples', file) for file in os.listdir('samples') if file.endswith('.json')], key=os.path.getctime):
        with open(sample_fp) as fp:
            pref_pairs = list(json.load(fp).items())
            train_pref_pairs = {k: v for (k, v) in pref_pairs[:args.train_samples_drawn_size]}
            eval_pref_pairs = {k: v for (k, v) in pref_pairs[args.train_samples_drawn_size:]}
            train_samples = train_samples | train_pref_pairs
            eval_samples = eval_samples | eval_pref_pairs


    return train_samples, eval_samples


def load_uncertainty_scores() -> np.ndarray:
    fp = max([os.path.join('samples', file) for file in os.listdir('samples') if file.endswith('.npz')], key=os.path.getctime)
    return np.load(fp)['arr_0']


if __name__ == "__main__":
    parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse()

    training_args.learning_rate = float(training_args.learning_rate)
    
    if model_config.model_name_or_path is None:
            model_config.model_name_or_path = script_args.ref_model

    if os.path.exists(training_args.output_dir):
        final_checkpoints_path = [os.path.join(training_args.output_dir, file) for file in os.listdir(training_args.output_dir) if file.startswith('final_checkpoint_')]
        n = len(final_checkpoints_path)

        if n > 0:
            model_config.model_name_or_path = max(final_checkpoints_path, key=os.path.getctime)
    else:
        n = 0

    print(f'Loading model from {model_config.model_name_or_path}')
    print(f'Mitigating attacks : {script_args.mitigate_corruption}')

    train_pref_pairs, eval_pref_pairs = load_data(script_args)
    scores = load_uncertainty_scores()

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation="flash_attention_2",
        dtype=torch.float16,)
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

    train_dataset = prepare_data(
        args=script_args,
        samples_preference_pair=train_pref_pairs,
        sanity_check=script_args.sanity_check,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
        is_train=True,
        uncertainty_scores=scores
    )
    
    eval_dataset = prepare_data(
        args=script_args,
        samples_preference_pair=eval_pref_pairs,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
        is_train=False
    )

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = os.getcwd()
    os.environ["WANDB_PROJECT"] = "iter_DPO"

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )
    print("begin to train")

    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, f"final_checkpoint_{n + 1}")
    dpo_trainer.model.save_pretrained(output_dir)
    dpo_trainer.processing_class.save_pretrained(output_dir)

    with open(f'log_history_{n + 1}.json', mode='w') as f:
        json.dump(dpo_trainer.state.log_history, fp=f, indent=4)