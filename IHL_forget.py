import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, HfArgumentParser
import deepspeed
import json
import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel, load_peft_weights, set_peft_model_state_dict
from pathlib import Path
from omegaconf import OmegaConf
from functools import reduce
import random
import numpy as np

from IHL_data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA
from IHL_dataloader import CustomTrainerForgetting, custom_data_collator_forget
from IHL_utils import get_model_identifiers_from_DPO_yaml

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            print(name)
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main():
    parser = HfArgumentParser()
    parser.add_argument('config_file', help='DPO.py config file', default=None)
    args = parser.parse_args()

    if args.config_file is not None:
        config = get_model_identifiers_from_DPO_yaml(args.config_file)
        iter_dpo_checkpoint_path = config['output_dir']
        final_checkpoints_path = [os.path.join(iter_dpo_checkpoint_path, file) for file in os.listdir(iter_dpo_checkpoint_path) if file.startswith('final_checkpoint_')]
        n = len(final_checkpoints_path)

        if n > 0:
            ref_model_path = max(final_checkpoints_path, key=os.path.getctime)
            config['ref_model'] = ref_model_path
        else:
            print('Did not find a DPO trained model!')
            return
    else:
        print('Did not pass argument!')
        return

    cfg = config['IHL_args']
    cfg['lr'] = float(cfg['lr'])
    
    print("######################")
    print("Saving to: ", config['output_dir'])
    print("######################")

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg['seed'])

    model_id = config['ref_model']

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = config['max_length']
    if cfg['forget_loss'] == "dpo":
        return
    else:
        torch_format_dataset = TextForgetDatasetQA(
             os.path.join(config['sample_save_dir'], 'samples_for_IHL'), 
            tokenizer=tokenizer, 
            model_family='', 
            max_length=max_length, 
            split='forget', 
            loss_type=cfg['forget_loss']
        )
    
    batch_size =  cfg['batch_size']
    gradient_accumulation_steps = cfg['gradient_accumulation_steps']
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg['num_epochs']*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg['lr'],
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        output_dir=config['output_dir'],
        optim="paged_adamw_32bit",
        save_strategy='no',
        save_steps=steps_per_epoch,
        save_only_model=True,
        ddp_find_unused_parameters= False,
        deepspeed='IHL_config/ds_config.json',
        weight_decay = cfg['weight_decay'],
        eval_steps = steps_per_epoch,
        eval_strategy ="no",
        seed=cfg['seed']
    )

    oracle_model = None

    print("Loading after merge and unload")
    merge_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map
    }
    
    merge_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **merge_kwargs)
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    #if model_cfg["gradient_checkpointing"] == "true":
    #    model.gradient_checkpointing_enable()

    LORA_config_dict = cfg['LORA']
    if LORA_config_dict['targets'] == 'all':
        lora_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    elif LORA_config_dict['targets'] == 'self_attn':
        lora_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif LORA_config_dict['targets'] == 'mlp':
        lora_targets = ['gate_proj', 'up_proj', 'down_proj']
    else:
        raise NotImplementedError
    
    lora_config = LoraConfig(
        r=LORA_config_dict['r'], 
        lora_alpha=LORA_config_dict['alpha'], 
        target_modules=lora_targets,
        lora_dropout=LORA_config_dict['dropout'],
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if lora_config.r != 0:
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)

    ### MODIFY MODEL WEIGHTS BASED ON IMPORTANCES
    def get_module_by_name(module, access_string):
        names = access_string.split(sep='.')
        return reduce(getattr, names, module)

    
    trainer = CustomTrainerForgetting(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None, # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback], 
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model = oracle_model,
        forget_loss = cfg['forget_loss'],
        eval_cfg = False,
        seed = cfg['seed'], # for NPO
        ref_policy = cfg['ref_policy'],
        beta = cfg['beta'],
        npo_coeff=cfg['npo_coeff'],
        grad_diff_coeff=cfg['grad_diff_coeff'],
        KL_coeff=cfg['KL_coeff'],
    )
    
    # Set tokenizer separately
    trainer.tokenizer = tokenizer
    
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    output_dir = os.path.join(config['output_dir'], f"final_checkpoint_{n}")
    trainer.train()

    merged_model = model.merge_and_unload()
    # Overides the previous checkpoint
    merged_model.save_pretrained(output_dir)

    print('Merged and Saved Model!')


if __name__ == "__main__":
    main()