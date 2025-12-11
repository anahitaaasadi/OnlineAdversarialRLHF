import os
import json
import yaml
import subprocess
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

    # Unlearn corrupted data if corruption was applied
    if script_args.corrupt_preferences and script_args.mitigate_corruption:
        print("\n" + "="*80)
        print("Starting unlearning process for corrupted data")
        print("="*80)
        
        # Step 1: Identify corrupted samples
        n_samples = len(train_pref_pairs)
        corrupted_idx_count = int(script_args.corruption_percentage * n_samples)
        np.random.seed(script_args.corruption_seed)
        corrupted_indices = np.random.permutation(np.arange(n_samples))[:corrupted_idx_count]
        
        # Save corrupted samples for unlearning
        corrupted_data_dir = os.path.join("data", "forget_samples")
        os.makedirs(corrupted_data_dir, exist_ok=True)
        
        corrupted_samples_path = os.path.join(corrupted_data_dir, f"corrupted_samples_iter_{n + 1}.json")
        forget_data_path = os.path.join(corrupted_data_dir, f"forget_data_iter_{n + 1}.json")
        
        # Prepare corrupted samples in the format expected by IHL
        prompts_list = list(train_pref_pairs.keys())
        corrupted_samples = []
        retain_samples = []
        
        for idx, (prompt_key, pairs) in enumerate(train_pref_pairs.items()):
            sample = {
                "question": prompt_key,
                "answer": pairs["best_response"].removeprefix(prompt_key),
                "paraphrased_answer": pairs["worst_response"].removeprefix(prompt_key)
            }
            
            if idx in corrupted_indices:
                corrupted_samples.append(sample)
            else:
                retain_samples.append(sample)
        
        # Save forget and retain data
        forget_dataset = {
            "forget": corrupted_samples,
            "retain": retain_samples
        }
        
        with open(forget_data_path, 'w') as f:
            json.dump(forget_dataset, f, indent=2)
        
        print(f"Saved {len(corrupted_samples)} corrupted samples to {forget_data_path}")
        print(f"Saved {len(retain_samples)} retain samples")
        
        # Step 2: Measure importance using IHL_measure_importance logic
        print("\n" + "-"*80)
        print("Step 1: Measuring parameter importance")
        print("-"*80)
        
        from IHL_measure_importance import main as measure_importance_main
        from omegaconf import OmegaConf
        
        # Create config for importance measurement
        importance_config = {
            "model_family": "phi",
            "model_path": output_dir,
            "data_path": forget_data_path,
            "split": "forget",
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "num_epochs": 1,
            "lr": 1e-5,
            "weight_decay": 0.01,
            "save_model": False,
            "eval_only": False,
            "eval_while_train": False,
            "seed": 42,
            "forget_loss": "grad_diff",
            "LoRA": {
                "r": 0,
                "alpha": 32,
                "dropout": 0.05,
                "targets": "all"
            },
            "npo_coeff": 1.0,
            "grad_diff_coeff": 1.0,
            "KL_coeff": 1.0,
            "ref_policy": "fine_tuned",
            "beta": 0.1,
            "eval": {
                "model_path": output_dir,
                "model_family": "phi",
                "save_dir": output_dir,
                "data_path": [forget_data_path],
                "split": "forget",
                "split_list": ["forget"],
                "eval_task": ["eval_log"],
                "question_key": ["question"],
                "answer_key": ["answer"],
                "base_answer_key": ["paraphrased_answer"]
            }
        }
        
        importance_save_path = f"./importances/phi_forget{str(script_args.corruption_percentage).replace('.', '')}_iter_{n + 1}.pt"
        os.makedirs("./importances", exist_ok=True)
        
        try:
            # Run importance measurement
            print(f"Computing importance scores and saving to {importance_save_path}")
            # Note: This would need to be run in a subprocess or refactored to be callable
            # For now, we'll prepare the command to run
            import subprocess
            importance_cmd = [
                "python", "IHL_measure_importance.py",
                f"model_family=phi",
                f"model_path={output_dir}",
                f"data_path={forget_data_path}",
                f"split=forget",
                "batch_size=8",
                "gradient_accumulation_steps=4",
                "num_epochs=1",
                f"lr=1e-5",
                "forget_loss=grad_diff"
            ]
            
            print(f"Running: {' '.join(importance_cmd)}")
            result = subprocess.run(importance_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Warning: Importance measurement failed: {result.stderr}")
            else:
                print("Importance measurement completed successfully")
                
        except Exception as e:
            print(f"Warning: Could not run importance measurement: {e}")
            importance_save_path = None
        
        # Step 3: Forget using IHL_forget
        print("\n" + "-"*80)
        print("Step 2: Forgetting corrupted samples")
        print("-"*80)
        
        forget_save_dir = os.path.join("llm_weights", f"forget_corrupted_iter_{n + 1}")
        
        forget_config = {
            "model_family": "phi",
            "model_path": output_dir,
            "data_path": forget_data_path,
            "split": "forget",
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "num_epochs": 3,
            "lr": 1e-5,
            "weight_decay": 0.01,
            "save_model": True,
            "save_dir": forget_save_dir,
            "overwrite_dir": True,
            "eval_only": False,
            "eval_while_train": False,
            "seed": 42,
            "forget_loss": "grad_ascent",
            "importance_file": importance_save_path if importance_save_path and os.path.exists(importance_save_path) else None,
            "LoRA": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "targets": "all"
            },
            "npo_coeff": 1.0,
            "grad_diff_coeff": 1.0,
            "KL_coeff": 1.0,
            "ref_policy": "fine_tuned",
            "beta": 0.1,
            "eval": {
                "model_path": output_dir,
                "model_family": "phi",
                "save_dir": forget_save_dir,
                "data_path": [forget_data_path],
                "split": "forget",
                "split_list": ["forget"],
                "eval_task": ["eval_log"],
                "question_key": ["question"],
                "answer_key": ["answer"],
                "base_answer_key": ["paraphrased_answer"]
            }
        }
        
        # Save config for reference
        config_path = os.path.join(forget_save_dir, "config.yaml")
        os.makedirs(forget_save_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(forget_config, f)
        
        try:
            # Run forgetting
            forget_cmd = [
                "python", "IHL_forget.py",
                f"model_family=phi",
                f"model_path={output_dir}",
                f"data_path={forget_data_path}",
                f"split=forget",
                f"save_dir={forget_save_dir}",
                "batch_size=8",
                "gradient_accumulation_steps=4",
                "num_epochs=3",
                f"lr=1e-5",
                "forget_loss=grad_ascent",
                "save_model=true",
                "overwrite_dir=true"
            ]
            
            if importance_save_path and os.path.exists(importance_save_path):
                forget_cmd.append(f"importance_file={importance_save_path}")
            
            print(f"Running: {' '.join(forget_cmd)}")
            result = subprocess.run(forget_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Warning: Forgetting failed: {result.stderr}")
            else:
                print(f"Forgetting completed successfully. Model saved to {forget_save_dir}")
                
        except Exception as e:
            print(f"Warning: Could not run forgetting: {e}")
        
        print("\n" + "="*80)
        print("Unlearning process completed")
        print("="*80)
