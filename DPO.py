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


def attack_mitigation(
    pos: list, 
    neg: list, 
    prompts: list, 
    margin: list, 
    corrupted_scores: np.ndarray,
    model,
    tokenizer,
    model_config,
    device: str = "cuda:0"
) -> tuple[list, list, list, list]:
    """
    Identify corrupted samples and unlearn them using IHL framework.
    Returns the original data unchanged (unlearning modifies the model in-place).
    """
    import json
    import tempfile
    from functools import reduce
    from pathlib import Path
    from peft import LoraConfig, get_peft_model
    import transformers
    import deepspeed
    import copy
    from tqdm import tqdm
    
    # Identify corrupted samples
    threshold = 1.01
    corrupted_indices = []
    
    for idx in range(len(prompts)):
        score = corrupted_scores[idx]
        threshold_bool = (score[0, :] / score[1, :]).mean(axis=-1) <= threshold
        inv_threshold_bool = (score[0, :] / score[1, :]).mean(axis=-1) >= 1/threshold
        should_unlearn = threshold_bool and not inv_threshold_bool
        
        if should_unlearn:
            corrupted_indices.append(idx)
    
    if len(corrupted_indices) == 0:
        print("No corrupted samples detected for unlearning.")
        return pos, neg, prompts, margin
    
    print(f"Identified {len(corrupted_indices)} corrupted samples for unlearning.")
    
    # Create forget dataset with corrupted samples
    forget_data = []
    for idx in corrupted_indices:
        forget_data.append({
            "question": prompts[idx],
            "answer": neg[idx]  # The incorrectly preferred response (currently labeled as positive)
        })
    
    # Create retain dataset with clean samples
    retain_data = []
    for idx in range(len(prompts)):
        if idx not in corrupted_indices:
            retain_data.append({
                "question": prompts[idx],
                "answer": pos[idx]
            })
    
    # Save datasets to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='_forget.json', delete=False) as f:
        json.dump(forget_data, f)
        forget_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_retain.json', delete=False) as f:
        json.dump(retain_data, f)
        retain_file = f.name
    
    try:
        # ========== STEP 1: Measure Importance ==========
        print("Step 1: Measuring importance scores...")
        
        from IHL_data_module import TextForgetDatasetQA
        from IHL_dataloader import custom_data_collator_forget
        
        # Create combined dataset for importance measurement
        combined_data = forget_data + retain_data[:len(forget_data)]  # Balance the dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='_combined.json', delete=False) as f:
            json.dump(combined_data, f)
            combined_file = f.name
        
        importance_dataset = TextForgetDatasetQA(
            combined_file,
            tokenizer=tokenizer,
            model_family=model_config.get('model_family', 'phi'),
            max_length=500,
            split='forget01',
            loss_type='grad_diff'
        )
        
        # Setup for importance measurement
        batch_size = 8
        gradient_accumulation_steps = 1
        
        importance_args = transformers.TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=1,  # Just one pass for importance
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            ddp_find_unused_parameters=False,
        )
        
        # Create a minimal trainer just to get dataloader
        from IHL_dataloader import CustomTrainerForgetting
        
        temp_trainer = CustomTrainerForgetting(
            model=model,
            train_dataset=importance_dataset,
            eval_dataset=importance_dataset,
            args=importance_args,
            data_collator=custom_data_collator_forget,
            forget_loss='grad_diff',
        )
        
        # Find linear layer names for importance tracking
        def find_all_linear_names(model):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')
            return list(lora_module_names)
        
        target_modules = find_all_linear_names(model)
        
        # Initialize importance dictionaries
        importance_f = {}
        importance_r = {}
        for name, param in model.named_parameters():
            for t in target_modules:
                if t in name and 'weight' in name:
                    importance_f[name] = torch.zeros_like(param.data, device='cpu')
                    importance_r[name] = torch.zeros_like(param.data, device='cpu')
        
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        f_cnt = 0
        r_cnt = 0
        
        # Measure importance
        dataloader = temp_trainer.get_train_dataloader()
        for step, inputs in enumerate(tqdm(dataloader, desc="Measuring importance")):
            if step >= len(forget_data) // batch_size + 1:  # Limit iterations
                break
                
            forget_input, retain_input = inputs
            
            # Forget samples
            input_ids, labels, attention_mask = forget_input
            output = model(
                input_ids=input_ids.to(device),
                labels=labels.to(device),
                attention_mask=attention_mask.to(device)
            )
            if output.loss is not None:
                output.loss.backward()
                cnt = torch.sum(labels != -100)
                for n, param in model.named_parameters():
                    if n in importance_f and param.grad is not None:
                        importance_f[n] += (param.grad.pow(2) * cnt.item()).detach().cpu()
                    if param.grad is not None:
                        param.grad = None
                f_cnt += cnt.item()
            
            # Retain samples
            input_ids, labels, attention_mask = retain_input
            output = model(
                input_ids=input_ids.to(device),
                labels=labels.to(device),
                attention_mask=attention_mask.to(device)
            )
            if output.loss is not None:
                output.loss.backward()
                cnt = torch.sum(labels != -100)
                for n, param in model.named_parameters():
                    if n in importance_r and param.grad is not None:
                        importance_r[n] += (param.grad.pow(2) * cnt.item()).detach().cpu()
                    if param.grad is not None:
                        param.grad = None
                r_cnt += cnt.item()
        
        print(f"Importance measurement complete: f_cnt={f_cnt}, r_cnt={r_cnt}")
        
        # ========== STEP 2: Apply FILA (Importance-Weighted LoRA Initialization) ==========
        print("Step 2: Applying FILA with importance-weighted LoRA...")
        
        # Configure LoRA
        lora_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        lora_r = 8
        lora_alpha = 32
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        # Initialize LoRA weights with importance
        def get_module_by_name(module, access_string):
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)
        
        importances = {
            n: torch.div(importance_f[n] / (f_cnt + 1e-10), 1e-5 + (importance_r[n] / (r_cnt + 1e-10)))
            for n in importance_f.keys()
        }
        
        initialized_layers = 0
        for old_name, importance in importances.items():
            if not any([target_name in old_name for target_name in lora_targets]):
                continue
            
            name = old_name.replace("module.", '')
            lora_A_path = 'base_model.model.' + name.replace(".weight", '') + '.lora_A.default'
            lora_B_path = 'base_model.model.' + name.replace(".weight", '') + '.lora_B.default'
            base_layer_path = 'base_model.model.' + name.replace(".weight", '') + '.base_layer'
            scaling_path = 'base_model.model.' + name.replace(".weight", '') + '.scaling'
            
            try:
                lora_A = get_module_by_name(model, lora_A_path)
                lora_B = get_module_by_name(model, lora_B_path)
                base_layer = get_module_by_name(model, base_layer_path)
                scaling = get_module_by_name(model, scaling_path)
                
                orig_shape = base_layer.weight.shape
                W = base_layer.weight.data.reshape(orig_shape)
                dtype = W.dtype
                W = W.to(torch.float32)
                
                # Importance-weighted low-rank approximation
                row_importance = importance.sum(dim=1).sqrt().to(W.device)
                U, S, V = torch.svd_lowrank(row_importance[:, None] * W, q=lora_r)
                S = S / scaling['default']
                
                new_lora_A = (V * torch.sqrt(S)).t()
                new_lora_B = (1 / (row_importance + 1e-5))[:, None] * (U * torch.sqrt(S))
                new_residual = base_layer.weight.data.reshape(orig_shape) - scaling['default'] * new_lora_B @ new_lora_A
                
                lora_A.weight.data = new_lora_A.contiguous().to(dtype)
                lora_B.weight.data = new_lora_B.contiguous().to(dtype)
                base_layer.weight.data = new_residual.contiguous().to(dtype)
                
                initialized_layers += 1
            except (AttributeError, KeyError) as e:
                continue
        
        print(f"FILA: Initialized {initialized_layers} LoRA layers")
        
        # ========== STEP 3: Unlearning via Gradient Ascent ==========
        print("Step 3: Performing unlearning via gradient ascent...")
        
        forget_dataset = TextForgetDatasetQA(
            forget_file,
            tokenizer=tokenizer,
            model_family=model_config.get('model_family', 'phi'),
            max_length=500,
            split='forget01',
            loss_type='grad_ascent'
        )
        
        unlearn_args = transformers.TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            max_steps=min(20, len(corrupted_indices) // 4 + 1),
            learning_rate=1e-5,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            ddp_find_unused_parameters=False,
            weight_decay=0.01,
        )
        
        unlearn_trainer = CustomTrainerForgetting(
            model=model,
            train_dataset=forget_dataset,
            eval_dataset=forget_dataset,
            args=unlearn_args,
            data_collator=custom_data_collator_forget,
            forget_loss='grad_ascent',
            grad_diff_coeff=1.0,
        )
        
        unlearn_trainer.train()
        print("Unlearning complete.")
        
    finally:
        # Cleanup temporary files
        import os
        for f in [forget_file, retain_file, combined_file]:
            if os.path.exists(f):
                os.unlink(f)
    
    return pos, neg, prompts, margin



def prepare_data(
    args: ScriptArguments,
    samples_preference_pair: dict,
    sanity_check: bool = False,
    margin_scale=1,
    eot_token="",
    is_train=True,
    uncertainty_scores: np.ndarray | None = None,
    model=None,
    tokenizer=None,
    model_config: dict | None = None,
    device: str = "cuda:0"
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
            corrupted_scores = corrupt_scores(uncertainty_scores, args)
            pos, neg, prompts, margin = attack_mitigation(
                pos, neg, prompts, margin, corrupted_scores,
                model, tokenizer, model_config, device
            )
            
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
        uncertainty_scores=scores,
        model=model,
        tokenizer=tokenizer,
        model_config={'model_family': 'phi'},
        device=script_args.device
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
