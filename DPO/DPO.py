import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from DPO_alignment import H4ArgumentParser
from trl import DPOConfig, ModelConfig
from DPO_utils import MyDPOTrainer

@dataclass
class ScriptArguments:
    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    train_dir: Optional[str] = field(
        default="./data/uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    eval_dir: Optional[str] = field(
        default="/export/home/hanze/project/vllm-gen/uf_split0_offline_reward.json",  # "/export/home/data/gemma_it_2b_3w_k8_with_pairrm_rewards.json",
        metadata={"help": "the location of the evalset name or path"},
    )
    # learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    # lr_scheduler_type: Optional[str] = field(
    #     default="constant_with_warmup", metadata={"help": "the lr scheduler type"}
    # )
    # warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    # weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    # optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    # per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    # per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    # gradient_accumulation_steps: Optional[int] = field(
    #     default=16, metadata={"help": "the number of gradient accumulation steps"}
    # )
    # gradient_checkpointing: Optional[bool] = field(
    #     default=True, metadata={"help": "whether to use gradient checkpointing"}
    # )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "whether to pad with eos token"})
    # lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    # lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    # lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})

    # max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "the maximum prompt length"})
    # max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    # max_steps: Optional[int] = field(default=20, metadata={"help": "max number of training steps"})
    # num_train_epochs: Optional[int] = field(default=2, metadata={"help": "max number of training epochs"})
    # logging_steps: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})
    # save_strategy: Optional[str] = field(default="epoch", metadata={"help": "the saving strategy"})
    # save_steps: Optional[int] = field(default=50000, metadata={"help": "the saving frequency"})
    # eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    # run_name: Optional[str] = field(default="dpo_soft", metadata={"help": "the run name"})
    # loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type"})
    # output_dir: Optional[str] = field(default="./dpo_soft", metadata={"help": "the output directory"})
    # log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})
    choose_type: Optional[str] = field(default="max_random", metadata={"help": "the choose type"})

    # report_to: Optional[str] = field(
    #     default="wandb",
    #     metadata={
    #         "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
    #         '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
    #         'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
    #     },
    # )
    
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})
    # mask_prompt: Optional[bool] = field(default=False, metadata={"help": "mask prompt"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "the length penalty"})


def prepare_data(
    data_dir: str = "/home/xiongwei/data/helpful/rm/rm1003.jsonl",
    sanity_check: bool = False,
    margin_scale=1,
    choose_type="random",
    eot_token="",
    length_penalty=0,
) -> Dataset:
    ds = load_dataset("json", data_files=data_dir, split="train")
    print(ds)

    pos = []
    neg = []
    prompts = []

    margin = []
    for sample in ds:
        P = tokenizer.apply_chat_template(sample["prompt"], tokenize = False, add_generation_prompt= True)
        if choose_type == "random":
            idx0 = 0
            idx1 = 1
        elif choose_type == "max_random":
            idx0 = np.argmax(sample["rewards"])
            if idx0 == 0:
                idx1 = 1
            else:
                idx1 = 0
        elif choose_type == "max_min":
            idx0 = np.argmax(sample["rewards"])
            idx1 = np.argmin(sample["rewards"])
        elif choose_type == "max_max":
            sorted_indices = np.argsort(sample["rewards"])
            idx0 = sorted_indices[-1]
            idx1 = sorted_indices[-2]
        elif choose_type == "max_min_p":
            r = [
                sample["rewards"][i] - length_penalty * len(sample["responses"][i])
                for i in range(len(sample["rewards"]))
            ]
            idx0 = np.argmax(r)
            idx1 = np.argmin(r)
        else:
            raise NotImplementedError

        if type(idx0) == np.ndarray or type(idx0) == list:
            assert len(idx0) == len(idx1)
            for i in range(len(idx0)):
                prompts.append(P)
                pos.append(sample["responses"][idx0[i]] + eot_token)
                neg.append(sample["responses"][idx1[i]] + eot_token)
                margin.append((sample["rewards"][idx0[i]] - sample["rewards"][idx1[i]]) * margin_scale)
        else:
            if sample["rewards"][idx0] > sample["rewards"][idx1]:
                prompts.append(P)
                pos.append(sample["responses"][idx0] + eot_token)
                neg.append(sample["responses"][idx1] + eot_token)
                margin.append((sample["rewards"][idx0] - sample["rewards"][idx1]) * margin_scale)
            elif sample["rewards"][idx0] < sample["rewards"][idx1]:
                prompts.append(P)
                pos.append(sample["responses"][idx1] + eot_token)
                neg.append(sample["responses"][idx0] + eot_token)
                margin.append((-sample["rewards"][idx0] + sample["rewards"][idx1]) * margin_scale)
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


if __name__ == "__main__":

    parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse()

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
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
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
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

    # 2. Load the Stack-exchange paired dataset
    train_dataset = prepare_data(
        data_dir=script_args.train_dir,
        margin_scale=script_args.margin_scale,
        sanity_check=script_args.sanity_check,
        choose_type=script_args.choose_type,
        eot_token=script_args.eot_token,
        length_penalty=script_args.len_penalty,
    )

    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    eval_dataset = prepare_data(
        data_dir=script_args.eval_dir,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
    )
    print(training_args)

    # 5. initialize the DPO trainer
    dpo_trainer = MyDPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        beta=training_args.beta,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )
    print("begin to train")

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

    # 7. save
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)