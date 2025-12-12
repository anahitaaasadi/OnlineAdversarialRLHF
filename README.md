# Corruption-Robust Online RLHF

**Team:** Anahita Asadi, Vaibhav Bhargava

This repository implements an online RLHF (Reinforcement Learning from Human Feedback) framework designed to be robust against adversarially corrupted preference data. The pipeline combines preference sampling, corruption simulation, filtering, and alternating DPO + IHL training.

---

# Framework Overview & Usage

## Directory Structure

- **`data.py`** — Defines `PreferenceSampler` for generating prompt–response pairs and sampling preferences.
- **`requirements.txt`** — Python package dependencies.
- **`DPO.py`** — Orchestrates one step of DPO in Online DPO.
- **`uncertainty.py`** — Calculates uncertainty of LLM.
- **`IHL_config/`** — Some config files ofr IHL.
- **Other files** — `README`, caches, and supporting code.

---

## Installation

0. **Download the dataset**
Install git-xet (https://hf.co/docs/hub/git-xet)
brew install git-xet
git xet install

git clone git@hf.co:datasets/RLHFflow/ultrafeedback_iter1

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:datasets/RLHFflow/ultrafeedback_iter1

# Make sure the hf CLI is installed
curl -LsSf https://hf.co/cli/install.sh | bash

# Download the dataset
hf download RLHFflow/ultrafeedback_iter1 --repo-type=dataset

**Download the model https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct**

**Update the paths in the yaml files.**

2. **Create a Conda environment** (recommended):
    ```bash
    conda create -n DPO python=3.12.2 -y
    conda activate DPO
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

# Running the code

### Run Iterative DPO with no corruption.

```bash
./main_iter_DPO.sh
```

### Run Iterative DPO with corruption.

```bash
./main_corruption.sh
```

### Run Iterative DPO with corruption mitigation. (Best performing)

```bash
./main_mitigate.sh
```

### Run Iterative DPO with corruption mitigation.

```bash
./main_IHL.sh
```

