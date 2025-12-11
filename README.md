# Corruption-Robust Online RLHF

**Team:** Anahita Asadi, Vaibhav Bhargava

This repository implements an online RLHF (Reinforcement Learning from Human Feedback) framework designed to be robust against adversarially corrupted preference data. The pipeline combines preference sampling, corruption simulation, filtering, and alternating DPO + IHL training.

---

# Framework Overview & Usage

## Directory Structure

- **`data.py`** — Defines `PreferenceSampler` for generating prompt–response pairs and sampling preferences.
- **`requirements.txt`** — Python package dependencies.
- **`DPO/`** — Contains Direct Preference Optimization (DPO) utilities and trainer extensions.
- **`IHL/`** — Contains Inverse Hinge Loss (IHL) and machine unlearning components.
- **Other files** — `README`, caches, and supporting code.

---

## Installation

1. **Create a Python environment** (recommended):
    ```bash
    python3 -m venv rlhf
    source rlhf/bin/activate
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Workflow Summary

1. **Preference Sampling**  
   `PreferenceSampler` generates prompt–response pairs and assigns preference labels using a reward model.

2. **Corruption Simulation**  
   A fraction of labels are flipped to simulate adversarial corruption. Uncertainty-targeted corruption is provided in `filter.py`.

3. **Filtering**  
   Samples are split into *clean* and *corrupt-suspected* groups using margin- or agreement-based filters.

4. **Training Loop**
   - **DPO Step:** Clean samples update the policy using Direct Preference Optimization (`DPO/DPO_utils.py`).
   - **IHL Step:** Suspected corrupt samples are used for unlearning via the Inverse Hinge Loss (`IHL/IHL_Loss.py`).

5. **Checkpointing**  
   After each iteration, the updated policy is saved in `BestMods/iteration_X`.

6. **Evaluation**  
   Final models can be evaluated on GSM-8K and LC-Alpaca-2 benchmarks.

---

## Code Components

### `main.py`
- Defines the `OnlineRLHFConfig` hyperparameter class.
- Implements the full online RLHF loop: loading models, sampling preferences, simulating corruption, filtering, DPO/IHL training, checkpointing, and GPU memory cleanup.
- CLI interface allows passing configuration parameters.

### `data.py`
- Implements `PreferenceSamplerConfig` and `PreferenceSampler`.
- Handles dataset loading and model-driven response generation.

### `filter.py`
- Functions for label corruption, uncertainty-based flipping, and margin/agreement-based filtering.

### `DPO/DPO_utils.py`
- Extends the standard DPO trainer with `MyDPOTrainer`.

### `IHL/IHL_Loss.py`
- Implements inverse hinge loss and unlearning using `CustomTrainerForgetting`.

---

## How to Run

Run the full online RLHF workflow:

```bash
python main.py \
    --num_iterations 3 \
    --samples_per_iter 128 \
    --flip_rate 0.2 \
    --margin_tau 0.1 \
    --device cuda:0
```

## How to run DPO.py

Run DPO without any corruptions

```
accelerate launch --config_file ds_zero3.yaml DPO.py config_no_corruption.yaml
```

Run DPO with advesarial corruptions

```
accelerate launch --config_file ds_zero3.yaml DPO.py config_corruption.yaml
```
