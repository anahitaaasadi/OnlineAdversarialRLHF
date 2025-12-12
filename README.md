# Corruption-Robust Online RLHF

**Team:** Anahita Asadi, Vaibhav Bhargava

This repository implements an online RLHF (Reinforcement Learning from Human Feedback) framework designed to be robust against adversarially corrupted preference data. The pipeline combines preference sampling, corruption simulation, filtering, and alternating DPO + IHL training.

---

# Framework Overview & Usage

## Directory Structure

- **`data.py`** — Defines `PreferenceSampler` for generating prompt–response pairs and sampling preferences.
- **`requirements.txt`** — Python package dependencies.
- **`DPO.py`** - Orchestrates one step of DPO in Online DPO.
- **`uncertainty.py`** - Calculates uncertainty of LLM.
- **`IHL_config/`** — Some config files ofr IHL.
- **Other files** — `README`, caches, and supporting code.

---

## Installation

1. **Create a Conda environment** (recommended):
    ```bash
    conda create -n DPO python=3.12.2 -y
    conda activate DPO
    ```

2. **Install dependencies**:
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

