import os
from collections import defaultdict
from dataclasses import dataclass, field
import gc

import torch
import numpy as np
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm

from datasets import load_dataset


@dataclass
class PreferenceSamplerConfig:
    samples_drawn_size: int = field(default=20, metadata={'help': 'Size of preference pairs collected.'})
    num_return_sequences: int = field(default=4, metadata={'help': 'How many samples are generated per policy before rejection sampling'})
    generation_seed: int | None = field(default=42, metadata={'help': 'Sample generation seed'})
    dataset_dir: str = field(default='ultrafeedback_iter1', metadata={'help': 'Dataset from which the prompts are sampled.'})
    ref_model_path: str = field(default='LLaMA3-SFT', metadata={'help': 'Policy being trained.'})
    ref_device: int = field(default=0, metadata={'help': 'GPU index where the policy model is loaded into.'})
    ref_gpu_utlization: float = field(default=0.75, metadata={'help': 'How much GPU should vLLM occupy between 0.0 and 1.0. Higher is better for faster inference at scale.'})
    rm_path: str = field(default='FsfairX-LLaMA3-RM-v0.1', metadata={'help': 'Reward model which simulates human feedback.'})
    # rm_gpu_utlization: float = field(default=0.75, metadata={'help': 'How much GPU should vLLM occupy between 0.0 and 1.0. Higher is better for faster inference at scale.'})
    rm_batch_size: int = field(default=2 * 4, metadata={'help': 'To speedup the reward calculating for all the samples created.'})
    rm_device: int = field(default=0, metadata={'help': 'GPU index where the reward model is loaded into.'})


    def __post_init__(self):
        self.rm_batch_size = min(2 * self.num_return_sequences, self.rm_batch_size)
        self.rm_batch_size -= self.rm_batch_size // (2 * self.num_return_sequences)


class PreferenceSampler:
    def __init__(self, config: PreferenceSamplerConfig):
        self.config = config
        dataset = load_dataset(config.dataset_dir)
        samples_drawn_size = min(config.samples_drawn_size, 20000)
        dataset = dataset['train'].take(samples_drawn_size)
        self.dataset = dataset

        self.ref_model_path = config.ref_model_path
        self.rm_path = config.rm_path

    def generate_responses(self) -> tuple[list[RequestOutput], list[RequestOutput]]:
        ref_tokenizer = AutoTokenizer.from_pretrained(self.ref_model_path)

        dataset = self.dataset.map(lambda x: {'prompt': ref_tokenizer.apply_chat_template(x['context_messages'], tokenize=False, add_generation_prompt=True)})

        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.config.ref_device}'

        ref_model = LLM(model=self.ref_model_path, enforce_eager=True, 
                        gpu_memory_utilization=self.config.ref_gpu_utlization)

        policy_1_param = SamplingParams(n=self.config.num_return_sequences, 
                            max_tokens=2048, temperature=1.0,
                            seed=self.config.generation_seed,
                            stop_token_ids=[ref_tokenizer.eos_token_id])

        policy_2_param = SamplingParams(n=self.config.num_return_sequences, 
                            max_tokens=2048, temperature=0.7,
                            seed=self.config.generation_seed,
                            stop_token_ids=[ref_tokenizer.eos_token_id])

        outputs_policy_1 = ref_model.generate(dataset['prompt'], sampling_params=policy_1_param, use_tqdm=True)
        outputs_policy_2 = ref_model.generate(dataset['prompt'], sampling_params=policy_2_param, use_tqdm=True)

        del ref_model
        gc.collect()

        del os.environ['CUDA_VISIBLE_DEVICES']

        return outputs_policy_1, outputs_policy_2


    def rejection_sampling(self, outputs_policy_1: list[RequestOutput], outputs_policy_2: list[RequestOutput]) -> dict:
        rm_tokenizer = AutoTokenizer.from_pretrained(self.rm_path)
        eot_id = '<|eot_id|>'

        rm_pipe = pipeline(
            "sentiment-analysis",
            model=self.rm_path,
            device=f'cuda:{self.config.rm_device}',
            tokenizer=rm_tokenizer,
            model_kwargs={"dtype": torch.bfloat16},
            truncation=True,
        )

        # NOTE: The scores become different if the batch size is not 1. Scores seem to become more positive in general, the relative ordering may or may not be affected.
        # So keeping it as just 1.

        pipe_kwargs = {
            # "return_all_scores": True,
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 1,
        }

        def get_reward(test_texts):
            pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
            rewards = [output[0]["score"] for output in pipe_outputs]
            return rewards

        samples = defaultdict(list)

        for output_policy_1, output_policy_2 in zip(outputs_policy_1, outputs_policy_2):
            prompt = output_policy_1.prompt

            for out1, out2 in zip(output_policy_1.outputs, output_policy_2.outputs):
                samples[prompt].append(prompt + out1.text + eot_id)
                samples[prompt].append(prompt + out2.text + eot_id)


        samples_preference_pair = {}

        pbar_samples = tqdm(samples.items())

        for idx, (prompt_key, chats) in enumerate(pbar_samples, start=1):
            pbar_samples.set_description(f'Ranking prompt {idx} samples')

            rm_rewards = np.array(get_reward(chats))
            
            best_response_idx, worst_response_idx = rm_rewards.argmax(), rm_rewards.argmin()

            samples_preference_pair[prompt_key] = {
                'best_response': chats[best_response_idx],
                'worst_response': chats[worst_response_idx],
                'best_reward': rm_rewards.max(),
                'worst_reward': rm_rewards.min()
            }

        del rm_pipe
        gc.collect()
        torch.cuda.empty_cache()

        return samples_preference_pair


    def _rejection_sampling_with_batching_fix(self, outputs_policy_1: list[RequestOutput], outputs_policy_2: list[RequestOutput]) -> dict:
        """
        NOTE: Still in progress. Still figuring out why batch scaling affects the scores. 
        """
        rm_tokenizer = AutoTokenizer.from_pretrained(self.rm_path)
        eot_id = '<|eot_id|>'

        # FIX: Set padding side to right for classification/reward modeling
        rm_tokenizer.padding_side = "right"

        # Ensure pad_token is set (Llama 3 often needs this manually mapped to eos_token if missing)
        if rm_tokenizer.pad_token is None:
            rm_tokenizer.pad_token = rm_tokenizer.eos_token

        rm_pipe = pipeline(
            "sentiment-analysis",
            model=self.rm_path,
            device=f'cuda:{self.config.rm_device}',
            tokenizer=rm_tokenizer,
            model_kwargs={"dtype": torch.bfloat16},
            truncation=True,
        )

        # NOTE: The scores become different if the batch size is not 1. Scores seem to become more positive (higher) in general, the relative ordering may or may not be affected.
        # So keeping it as just 1.

        pipe_kwargs = {
            # "return_all_scores": True,
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": self.config.rm_batch_size,
        }

        def get_reward(test_texts):
            pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
            rewards = [output[0]["score"] for output in pipe_outputs]
            return rewards

        samples = defaultdict(list)

        for output_policy_1, output_policy_2 in zip(outputs_policy_1, outputs_policy_2):
            prompt = output_policy_1.prompt

            for out1, out2 in zip(output_policy_1.outputs, output_policy_2.outputs):
                samples[prompt].append(prompt + out1.text + eot_id)
                samples[prompt].append(prompt + out2.text + eot_id)


        samples_preference_pair = {}

        pbar_samples = tqdm(samples.items())

        for idx, (prompt_key, chats) in enumerate(pbar_samples, start=1):
            pbar_samples.set_description(f'Ranking prompt {idx} samples')

            rm_rewards = np.array(get_reward(chats))
            
            best_response_idx, worst_response_idx = rm_rewards.argmax(), rm_rewards.argmin()

            samples_preference_pair[prompt_key] = {
                'best_response': chats[best_response_idx],
                'worst_response': chats[worst_response_idx]
            }

        del rm_pipe
        gc.collect()
        torch.cuda.empty_cache()

        return samples_preference_pair


if __name__ == '__main__':
    MODEL_DIR = os.path.join('/home/vbharg4@AD', 'models')
    ds_id = os.path.join(os.getcwd(), "ultrafeedback_iter1")
    ref_model_path = os.path.join(MODEL_DIR, 'LLaMA3-SFT')
    reward_model_path = os.path.join(MODEL_DIR, 'FsfairX-LLaMA3-RM-v0.1')

    config = PreferenceSamplerConfig(dataset_dir=ds_id, model_path=ref_model_path, rm_path=reward_model_path)
    pref_sampler = PreferenceSampler(config=config)
