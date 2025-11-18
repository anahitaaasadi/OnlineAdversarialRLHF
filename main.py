import argparse
import gc
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import PreferenceSampler, PreferenceSamplerConfig
from filter import (
    flip_labels_uncertainty_targeting_text,
    filter_by_margin,
    filter_by_agreement,
)
from DPO.DPO_utils import MyDPOTrainer
from trl import DPOConfig
from IHL.IHL_Loss import CustomTrainerForgetting
from IHL.data_module import convert_raw_data_to_model_format, custom_data_collator


@dataclass
class OnlineRLHFConfig:
    # base models
    policy_model_name: str = "RLHFlow/LLaMA3-SFT"
    ref_model_name: str = "RLHFlow/LLaMA3-SFT"
    rm_model_name: str = "sfairXC/FsfairX-LLaMA3-RM-v0.1"

    # data / sampler
    ulf_dataset_dir: str = "RLHFlow/ultrafeedback_iter1"
    samples_per_iter: int = 16
    num_return_sequences: int = 4

    # online RLHF
    num_iterations: int = 2
    flip_rate: float = 0.2        # fraction of prefs to corrupt
    margin_tau: float = 0.1       # threshold for filter_by_margin

    # DPO hyperparams
    dpo_output_dir: str = "checkpoints/dpo_online"
    dpo_max_length: int = 2048
    dpo_max_prompt_length: int = 1024
    dpo_learning_rate: float = 5e-7
    dpo_train_batch_size: int = 1
    dpo_num_epochs: int = 1
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"

    # IHL / unlearning hyperparams
    ihl_output_dir: str = "checkpoints/ihl_online"
    ihl_max_length: int = 512
    ihl_learning_rate: float = 5e-7
    ihl_train_batch_size: int = 1
    ihl_num_epochs: int = 1
    ihl_loss_type: str = "IHL"   # This corresponds to your CustomTrainerForgetting

    # misc
    device: str = "cuda:0"
    save_dir: str = "BestMods"
    

def build_pair_dataset_from_sampler_output(preference_pairs: Dict[int, Dict[str, Any]]) -> Dataset:
    """
    Convert PreferenceSampler.rejection_sampling output into a DPO-style Dataset.

    preference_pairs[idx] = {
        "best_response": (prompt_str, best_response_str),
        "worst_response": (prompt_str, worst_response_str),
    }
    """
    prompts, chosen, rejected = [], [], []
    for _, pair in preference_pairs.items():
        prompt, best_resp = pair["best_response"]
        _, worst_resp = pair["worst_response"]
        prompts.append(prompt)
        chosen.append(best_resp)
        rejected.append(worst_resp)

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        }
    )


def compute_logps_for_pairs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: str = "cuda:0",
):
    """
    Compute log p_pi(chosen | prompt) and log p_pi(rejected | prompt) for each example.

    Returns:
        lp_c: tensor [B]
        lp_r: tensor [B]
    """
    model.to(device)
    model.eval()

    lp_c_list = []
    lp_r_list = []

    with torch.no_grad():
        for example in dataset:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            # concatenate prompt + response, then compute token-level loss for response tokens
            def seq_logp(response: str) -> float:
                text = prompt + response
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)

                # shift labels to compute log p(response | full prefix)
                labels = tokens["input_ids"].clone()
                # we don't mask out prompt tokens here; we want total log p of whole string.
                outputs = model(**tokens, labels=labels)
                # HF gives mean loss over non-masked tokens; multiply by length to get total log prob
                # loss is -log p / len, so:
                total_logp = -outputs.loss.item() * labels.numel()
                return total_logp

            lp_c_list.append(seq_logp(chosen))
            lp_r_list.append(seq_logp(rejected))

    lp_c = torch.tensor(lp_c_list, device=device)
    lp_r = torch.tensor(lp_r_list, device=device)
    return lp_c, lp_r


def apply_label_flips(dataset: Dataset, flip_mask: torch.Tensor) -> Dataset:
    """
    For indices where flip_mask[i] == True, swap chosen/rejected.
    """
    prompts = dataset["prompt"]
    chosen = dataset["chosen"]
    rejected = dataset["rejected"]

    chosen_new = []
    rejected_new = []

    for i in range(len(prompts)):
        if flip_mask[i]:
            chosen_new.append(rejected[i])
            rejected_new.append(chosen[i])
        else:
            chosen_new.append(chosen[i])
            rejected_new.append(rejected[i])

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosen_new,
            "rejected": rejected_new,
        }
    )


def split_clean_corrupt(dataset: Dataset, clean_mask: torch.Tensor, corrupt_mask: torch.Tensor):
    idx_clean = torch.where(clean_mask)[0].cpu().numpy().tolist()
    idx_corrupt = torch.where(corrupt_mask)[0].cpu().numpy().tolist()

    clean_ds = dataset.select(idx_clean) if len(idx_clean) > 0 else Dataset.from_dict({"prompt": [], "chosen": [], "rejected": []})
    corrupt_ds = dataset.select(idx_corrupt) if len(idx_corrupt) > 0 else Dataset.from_dict({"prompt": [], "chosen": [], "rejected": []})
    return clean_ds, corrupt_ds


def dpo_train_step(
    cfg: OnlineRLHFConfig,
    policy_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
) -> AutoModelForCausalLM:
    """
    One "round" of DPO over the current clean dataset.
    """
    if len(train_dataset) == 0:
        print("DPO step skipped (no clean data).")
        return policy_model

    dpo_args = DPOConfig(
        output_dir=cfg.dpo_output_dir,
        per_device_train_batch_size=cfg.dpo_train_batch_size,
        learning_rate=cfg.dpo_learning_rate,
        num_train_epochs=cfg.dpo_num_epochs,
        beta=cfg.dpo_beta,
        max_length=cfg.dpo_max_length,
        max_prompt_length=cfg.dpo_max_prompt_length,
        remove_unused_columns=False,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        loss_type=cfg.dpo_loss_type,
    )

    trainer = MyDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        beta=cfg.dpo_beta,
        max_length=cfg.dpo_max_length,
        max_prompt_length=cfg.dpo_max_prompt_length,
        loss_type=cfg.dpo_loss_type,
    )

    trainer.train()
    return trainer.model


def build_ihl_dataset_from_pairs(
    corrupt_ds: Dataset,
    tokenizer: AutoTokenizer,
    model_family: str,
    max_length: int,
) -> Dataset:
    """
    Turn (prompt, chosen, rejected) pairs into a simple QA-style dataset
    to feed into IHL unlearning.

    Very simple choice: use prompt as question, chosen as 'answer'.
    You can later try variants like answer=chosen or answer=rejected
    depending on what you want to forget.
    """
    questions = []
    answers = []

    for ex in corrupt_ds:
        questions.append(ex["prompt"])
        answers.append(ex["chosen"])

    # We build tensors manually using convert_raw_data_to_model_format
    input_ids_list = []
    labels_list = []
    attn_list = []

    # You need a model_configs YAML on disk that get_model_identifiers_from_yaml(model_family) can read.
    # Here we assume the YAML is already in place and model_family points to it.
    from IHL.utils import get_model_identifiers_from_yaml
    model_configs = get_model_identifiers_from_yaml(model_family)

    for q, a in zip(questions, answers):
        inp, lab, attn = convert_raw_data_to_model_format(
            tokenizer,
            max_length,
            q,
            a,
            model_configs,
        )
        input_ids_list.append(inp)
        labels_list.append(lab)
        attn_list.append(attn)

    return Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attn_list,
        }
    )


def ihl_train_step(
    cfg: OnlineRLHFConfig,
    policy_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    ihl_dataset: Dataset,
) -> AutoModelForCausalLM:
    """
    One unlearning step using CustomTrainerForgetting with an IHL-style loss.

    Note: CustomTrainerForgetting in IHL_Loss.py expects some extra kwargs:
    - forget_loss
    - oracle_model
    - eval_cfg
    - seed, npo_coeff, grad_diff_coeff, KL_coeff, ref_policy, beta
    For now we use simple placeholders/easy defaults.
    """
    if len(ihl_dataset) == 0:
        print("IHL step skipped (no corrupt data).")
        return policy_model

    from transformers import TrainingArguments

    train_args = TrainingArguments(
        output_dir=cfg.ihl_output_dir,
        per_device_train_batch_size=cfg.ihl_train_batch_size,
        learning_rate=cfg.ihl_learning_rate,
        num_train_epochs=cfg.ihl_num_epochs,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    # Minimal eval_cfg stub – you can later plug in your full eval config for GSM8K / LC Alpaca 2
    class DummyEvalCfg:
        def __init__(self):
            self.data_path = []
            self.split_list = []
            self.question_key = []
            self.answer_key = []
            self.eval_task = []
            self.base_answer_key = []
            self.perturbed_answer_key = []
            self.save_dir = cfg.ihl_output_dir
            self.split = "forget10_split"
            self.ds_size = None
            self.batch_size = cfg.ihl_train_batch_size
            self.model_family = "llama2-7b"
            self.generation = type("GenCfg", (), {"max_length": cfg.ihl_max_length, "max_new_tokens": 128})

    eval_cfg = DummyEvalCfg()

    # CustomTrainerForgetting expects the dataset to yield (input_ids, labels, attn_mask)
    def collator(examples):
        input_ids = torch.stack([e["input_ids"] for e in examples])
        labels = torch.stack([e["labels"] for e in examples])
        attn_mask = torch.stack([e["attention_mask"] for e in examples])
        return input_ids, labels, attn_mask

    trainer = CustomTrainerForgetting(
        model=policy_model,
        args=train_args,
        train_dataset=ihl_dataset,
        data_collator=collator,
        forget_loss=cfg.ihl_loss_type,
        oracle_model=ref_model,
        eval_cfg=eval_cfg,
        seed=42,
        npo_coeff=1.0,
        grad_diff_coeff=1.0,
        KL_coeff=0.0,
        ref_policy="fine_tuned",
        beta=0.1,
    )

    trainer.train()
    return trainer.model


# =======================
# 3. Online RLHF loop
# =======================

def run_online_rlhf(cfg: OnlineRLHFConfig):
    device = cfg.device
    
    # 1. Load models / tokenizer
    # Clean up GPU memory before loading models
    gc.collect()
    torch.cuda.empty_cache()

    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    policy_model.config.use_cache = False

    # Clean up again before loading ref_model
    if 'ref_model' in locals():
        del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.ref_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    ref_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Track cumulative clean & corrupt datasets
    clean_history: List[Dataset] = []
    corrupt_history: List[Dataset] = []

    for t in range(1, cfg.num_iterations + 1):
        print(f"\n================ Iteration {t} ================\n")

        # 2.1 Sample preferences from current policy
        sampler_cfg = PreferenceSamplerConfig(
            rm_device=0,
            samples_drawn_size=cfg.samples_per_iter,
            num_return_sequences=cfg.num_return_sequences,
            generation_seed=42 + t,
            dataset_dir=cfg.ulf_dataset_dir,
            model_path=cfg.policy_model_name,
            rm_path=cfg.rm_model_name,
        )
        sampler = PreferenceSampler(sampler_cfg)
        responses = sampler.generate_responses()
        preference_pairs = sampler.rejection_sampling(responses)

        dpo_ds_t = build_pair_dataset_from_sampler_output(preference_pairs)
        print(f"[Iter {t}] Collected {len(dpo_ds_t)} preference pairs")

        # 2.2 Compute policy & ref log-probs
        lp_c_pi, lp_r_pi = compute_logps_for_pairs(policy_model, tokenizer, dpo_ds_t, device=device)
        lp_c_ref, lp_r_ref = compute_logps_for_pairs(ref_model, tokenizer, dpo_ds_t, device=device)

        # 2.3 Simulate adversarial corruption (flip labels)
        flip_mask = flip_labels_uncertainty_targeting_text(lp_c_pi, lp_r_pi, flip_rate=cfg.flip_rate)
        dpo_ds_t_flipped = apply_label_flips(dpo_ds_t, flip_mask)
        print(f"[Iter {t}] Corrupted {flip_mask.sum().item()} / {len(dpo_ds_t)} pairs")

        # 2.4 Detect clean vs corrupt using margin or agreement
        clean_mask, corrupt_mask, margins = filter_by_margin(lp_c_pi, lp_r_pi, tau=cfg.margin_tau)
        # Or alternative: clean_mask, corrupt_mask = filter_by_agreement(...)

        clean_ds_t, corrupt_ds_t = split_clean_corrupt(dpo_ds_t_flipped, clean_mask, corrupt_mask)
        print(f"[Iter {t}] Detected {len(clean_ds_t)} clean, {len(corrupt_ds_t)} corrupt pairs")

        if len(clean_ds_t) > 0:
            clean_history.append(clean_ds_t)
        if len(corrupt_ds_t) > 0:
            corrupt_history.append(corrupt_ds_t)

        # 2.5 DPO step on all clean data so far
        if len(clean_history) > 0:
            train_clean = concatenate_datasets(clean_history) if len(clean_history) > 1 else clean_history[0]
        else:
            train_clean = Dataset.from_dict({"prompt": [], "chosen": [], "rejected": []})

        print(f"[Iter {t}] Running DPO on {len(train_clean)} clean pairs...")
        policy_model = dpo_train_step(cfg, policy_model, ref_model, tokenizer, train_clean)

        # 2.6 IHL unlearning step on *current* corrupt data
        if len(corrupt_ds_t) > 0:
            print(f"[Iter {t}] Running IHL unlearning on {len(corrupt_ds_t)} corrupt pairs...")
            ihl_dataset = build_ihl_dataset_from_pairs(
                corrupt_ds_t,
                tokenizer=tokenizer,
                model_family="llama2-7b",      # <-- point this to your YAML config
                max_length=cfg.ihl_max_length,
            )
            policy_model = ihl_train_step(cfg, policy_model, ref_model, ihl_dataset)
        else:
            print(f"[Iter {t}] Skipping IHL (no corrupt data this round).")

        # 2.7 Save model checkpoint after this iteration
        save_path = os.path.join(cfg.save_dir, f"iteration_{t}")
        os.makedirs(save_path, exist_ok=True)
        print(f"[Iter {t}] Saving model to {save_path}...")
        policy_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[Iter {t}] Model saved successfully.")

        # 2.8 Clean up GPU memory before next iteration
        if t < cfg.num_iterations:
            print(f"[Iter {t}] Cleaning up GPU memory...")
            # Move models to CPU to free GPU memory
            policy_model.to('cpu')
            ref_model.to('cpu')
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()
            # Reload policy model for next iteration
            del policy_model
            torch.cuda.empty_cache()
            gc.collect()
            policy_model = AutoModelForCausalLM.from_pretrained(
                save_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            policy_model.config.use_cache = False
            # Move ref model back to GPU
            ref_model.to(device)
            print(f"[Iter {t}] GPU memory cleanup complete.")

    print("\nFinished online RLHF loop. You can now run GSM8K / Alpaca eval using evaluate_util.py etc.")


# =======================
# 4. CLI entry point
# =======================

def parse_args() -> OnlineRLHFConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_model_name", type=str, default="RLHFlow/LLaMA3-SFT")
    parser.add_argument("--ref_model_name", type=str, default="RLHFlow/LLaMA3-SFT")
    parser.add_argument("--rm_model_name", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--ulf_dataset_dir", type=str, default="RLHFlow/ultrafeedback_iter1")
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--samples_per_iter", type=int, default=128)
    parser.add_argument("--flip_rate", type=float, default=0.2)
    parser.add_argument("--margin_tau", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    cfg = OnlineRLHFConfig(
        policy_model_name=args.policy_model_name,
        ref_model_name=args.ref_model_name,
        rm_model_name=args.rm_model_name,
        ulf_dataset_dir=args.ulf_dataset_dir,
        num_iterations=args.num_iterations,
        samples_per_iter=args.samples_per_iter,
        flip_rate=args.flip_rate,
        margin_tau=args.margin_tau,
        device=args.device,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run_online_rlhf(cfg)
