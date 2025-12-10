"""
Converts corrupted samples detected by DPO.py into the format expected by IHL_forget.py
"""
import json
import os
import sys
from pathlib import Path


def convert_corrupted_to_forget_format(corrupted_file: str, output_file: str):
    """
    Convert corrupted samples to IHL forget format.
    
    IHL expects format like:
    [
        {
            "question": "prompt text",
            "answer": "chosen response",
            "rejected": "rejected response"  # optional
        },
        ...
    ]
    """
    with open(corrupted_file, 'r') as f:
        corrupted_data = json.load(f)
    
    forget_data = []
    for i in range(len(corrupted_data['prompts'])):
        sample = {
            "question": corrupted_data['prompts'][i],
            "answer": corrupted_data['chosen'][i],
            "rejected": corrupted_data['rejected'][i]
        }
        forget_data.append(sample)
    
    # Save in IHL format
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(forget_data, f, indent=4)
    
    print(f"Converted {len(forget_data)} samples from {corrupted_file} to {output_file}")
    return len(forget_data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_forget_data.py <iteration_number>")
        sys.exit(1)
    
    iteration = sys.argv[1]
    
    corrupted_file = f"data/forget_samples/corrupted_samples_iter_{iteration}.json"
    output_file = f"data/forget_samples/forget_data_iter_{iteration}.json"
    
    if not os.path.exists(corrupted_file):
        print(f"Error: {corrupted_file} not found")
        sys.exit(1)
    
    convert_corrupted_to_forget_format(corrupted_file, output_file)
