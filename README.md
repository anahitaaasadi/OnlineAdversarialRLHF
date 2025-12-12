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

**Download the dataset**
Install git-xet (https://hf.co/docs/hub/git-xet)
```bash
brew install git-xet
git xet install
```

```bash
git clone git@hf.co:datasets/RLHFflow/ultrafeedback_iter1
```

**Download the reward model**
```bash
git clone git@hf.co:sfairXC/FsfairX-LLaMA3-RM-v0.1
```
**Download the policy model**
```bash
git clone git@hf.co:RLHFlow/LLaMA3-SFT
```

1. **Update the paths in the yaml files.**

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

