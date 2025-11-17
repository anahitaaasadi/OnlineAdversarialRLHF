import os
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import torch
from tqdm import tqdm

from transformers import LlamaForCausalLM, AutoTokenizer, pipeline, GenerationConfig, AutoModelForCausalLM, set_seed
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, Dataset


@dataclass
class PreferenceSamplerConfig:
    rm_device: int = field(default=0, metadata={'help': 'GPU index where the model is loaded into.'})
    samples_drawn_size: int = field(default=2, metadata={'help': 'Size of preference pairs collected.'})
    num_return_sequences: int = field(default=4, metadata={'help': 'How many samples are generated per policy before rejection sampling'})
    generation_seed: int | None = field(default=42, metadata={'help': 'Sample generation seed'})
    dataset_dir: str = field(default='ultrafeedback_iter1', metadata={'help': 'Dataset from which the prompts are sampled.'})
    model_path: str = field(default='LLaMA3-SFT', metadata={'help': 'Policy being trained.'})
    rm_path: str = field(default='FsfairX-LLaMA3-RM-v0.1', metadata={'help': 'Reward model which simulates human feedback.'})


class PreferenceSampler:
    def __init__(self, config: PreferenceSamplerConfig):
        self.config = config
        dataset = load_dataset(config.dataset_dir)
        # We consider only 1000 samples as the size of d_0
        self.dataset = KeyDataset(dataset['train'].take(config.samples_drawn_size), 'context_messages')

        self.model_path = config.model_path
        self.rm_path = config.rm_path


    def generate_responses(self) -> dict:
        data_config = self.config
        ref_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        ref_model = LlamaForCausalLM.from_pretrained(self.model_path, device_map='cuda:0', torch_dtype=torch.bfloat16)

        num_return_sequences = data_config.num_return_sequences

        generate_config_policy_1 = GenerationConfig(
            do_sample=True,
            max_new_tokens=2048, # This is 2048 in Online RLHF paper
            temperature=1.0,
            top_k=0, # Online RLHF uses vLLM and there it defaults to -1. 0 in HF has the same behavior.
            num_return_sequences=num_return_sequences, # This is 4 in Online RLHF paper. \pi_t^1
            eos_token_id=[ref_model.config.eos_token_id], # Set EOS token id ('<|eot_id|>') to stop generating unnecessary EOS tokens. 
            pad_token_id=ref_model.config.eos_token_id,
            # batch_size=8 * 2
            # use_cache=False # disable KV cache to get the complete attention score matrix after the system / user prompt.
        )

        generate_config_policy_2 = GenerationConfig(
            do_sample=True,
            max_new_tokens=2048, # This is 2048 in Online RLHF paper
            temperature=0.7,
            top_k=0, # Online RLHF uses vLLM and there it defaults to -1. 0 in HF has the same behavior.
            num_return_sequences=num_return_sequences, # This is 4 in Online RLHF paper. \pi_t^2
            eos_token_id=[ref_model.config.eos_token_id], # Set EOS token id ('<|eot_id|>') to stop generating unnecessary EOS tokens. 
            pad_token_id=ref_model.config.eos_token_id,
            # batch_size=8 * 2
            # use_cache=False # disable KV cache to get the complete attention score matrix after the system / user prompt.
        )

        model_pipeline = pipeline(
            'text-generation', model=ref_model, 
            tokenizer=ref_tokenizer, device_map='cuda', 
            torch_dtype=torch.bfloat16)

        # set_seed(data_config.generation_seed)

        # idx = np.random.randint(low=0, high=len(self.dataset))
        # prompt_sample_indices = [(idx + idx_1) % len(self.dataset) for idx_1 in range(2 * num_return_sequences)]

        # responses = defaultdict({'response_1': [], 'response_2': []})
        responses = defaultdict(lambda: defaultdict(list))

        set_seed(data_config.generation_seed)
        pbar_pipeline_1 = tqdm(model_pipeline(text_inputs=self.dataset, generation_config=generate_config_policy_1))

        for idx, output in enumerate(pbar_pipeline_1, start=1):
            pbar_pipeline_1.set_description(f'Sampling prompt {idx} from policy 1')

            for sample in output:
                responses[idx]['response_1'].append(sample['generated_text'])

        set_seed(data_config.generation_seed)
        pbar_pipeline_2 = tqdm(model_pipeline(text_inputs=self.dataset, generation_config=generate_config_policy_2) )

        for idx, output in enumerate(pbar_pipeline_2, start=1):
            pbar_pipeline_2.set_description(f'Sampling prompt {idx} from policy 2')

            for sample in output:
                responses[idx]['response_2'].append(sample['generated_text'])


        return responses


    def rejection_sampling(self, responses: dict):
        data_config = self.config
        rm_tokenizer = AutoTokenizer.from_pretrained(self.rm_path)

        rm_pipe = pipeline(
            "sentiment-analysis",
            model=self.rm_path,
            device='cuda:0',
            tokenizer=rm_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            truncation=True,
        )

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

        for prompt_idx, response in responses.items():
            for policy_key, policy_responses in response.items():
                for prompt_response_pair in policy_responses:
                    prompt, response = prompt_response_pair
                    prompt_key = prompt['content']

                    samples[prompt_key].append((prompt, response))

        samples_preference_pair = {}

        pbar_samples = tqdm(samples.items())

        counter = 0

        for prompt_key, chats in pbar_samples:
            pbar_samples.set_description(f'Ranking prompt {counter + 1} samples')
            counter += 1

            rm_prompts = [chat.replace(rm_tokenizer.bos_token, "") for chat in rm_tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=False)]
            rm_rewards = np.array(get_reward(rm_prompts))
            
            best_response_idx, worst_response_idx = rm_rewards.argmax(), rm_rewards.argmin()

            samples_preference_pair[prompt_key] = {
                'best_response': chats[best_response_idx],
                'worst_response': chats[worst_response_idx]
            }

        return samples_preference_pair
    
if __name__ == '__main__':
    config = PreferenceSamplerConfig(
        dataset_dir='RLHFlow/ultrafeedback_iter1',
        model_path='RLHFlow/LLaMA3-SFT',
        rm_path='sfairXC/FsfairX-LLaMA3-RM-v0.1'
    )
    pref_sampler = PreferenceSampler(config=config)
    responses = pref_sampler.generate_responses()
    preference_samples = pref_sampler.rejection_sampling(responses=responses)