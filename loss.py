# ---------- Losses ----------

import torch
from torch import nn
from torch import Tensor, tensor
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

from torchmetrics.classification import MulticlassHingeLoss
from torchmetrics.utilities.data import to_onehot
from torchmetrics.metric import Metric
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_format
from torchmetrics.functional.classification.hinge import (
    _multiclass_hinge_loss_arg_validation, 
    _multiclass_hinge_loss_tensor_validation,
    _hinge_loss_compute
)


import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from trl.trainer.utils import cap_exp
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType

class MyDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Optional[str] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            precompute_ref_log_probs=precompute_ref_log_probs,
            dataset_num_proc=dataset_num_proc,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            model_adapter_name=model_adapter_name,
            ref_adapter_name=ref_adapter_name,
            reference_free=reference_free,
            force_use_ref_model=force_use_ref_model,
        )


    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)

            delta = chosen_logratios_sorted - rejected_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios_sorted, _ = torch.sort(pi_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)

            delta = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output

            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood

            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output

            losses_chosen = F.sigmoid(self.beta * chosen_logratios)  # Decrease chosen likelihood
            losses_rejected = 1 - F.sigmoid(
                self.beta * (chosen_logratios - rejected_logratios)
            )  # Decrease rejected likelihood more

            losses = losses_chosen + losses_rejected

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # RPO loss from V3 of the paper:
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss


    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)


def _custom_multiclass_hinge_loss_update(
    preds,
    target,
    alpha,
    squared,
    multiclass_mode = "crammer-singer"
):
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.softmax(1)

    target = to_onehot(target, max(2, preds.shape[1])).bool()
    if multiclass_mode == "crammer-singer":
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    else:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]

    measures = alpha + margin
    measures = torch.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total

def multiclass_hinge_loss(
    preds,
    target,
    num_classes,
    alpha = 1.0,
    squared = False,
    multiclass_mode = "crammer-singer",
    ignore_index = None,
    validate_args = True,
):
    if validate_args:
        _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        _multiclass_hinge_loss_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
    measures, total = _custom_multiclass_hinge_loss_update(
        preds, 
        target, 
        alpha,
        squared, 
        multiclass_mode,
    )
    return _hinge_loss_compute(measures, total)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')

        # Parameters added for NPO
        self.seed = kwargs.pop('seed')
        self.npo_coeff=kwargs.pop('npo_coeff')
        self.grad_diff_coeff=kwargs.pop('grad_diff_coeff')
        self.KL_coeff=kwargs.pop('KL_coeff')
        self.ref_policy = kwargs.pop('ref_policy')
        self.beta = kwargs.pop('beta')

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        if self.loss_type == "KL" or "npo" in self.loss_type:
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    

    def compute_loss(self, model, inputs, return_outputs=False):
        
        if self.loss_type == "grad_ascent":
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

        elif self.loss_type == "grad_diff":
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss

        elif self.loss_type == "IHL":
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            #forget_loss = outputs.loss
            #forget_loss = forget_loss * -1 # need to replace

            scores = outputs.logits
            shift_logits = scores[..., :-1, :].contiguous().squeeze().view(-1, scores.size(-1)) # [BN, V]
            shift_labels = labels[..., 1:].contiguous().squeeze().view(-1) # [BN,]
            forget_loss = multiclass_hinge_loss(
                shift_logits[shift_labels != -100,:], # ignore pad tokens
                shift_labels[shift_labels != -100],
                shift_logits.size(-1),
            )

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            
            loss = forget_loss + retain_loss
        
        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(
                    retain_input_ids,labels=retain_labels, 
                    attention_mask=retain_attention_mask
                )
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            #concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        
        elif self.loss_type == "dpo":
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(
                    idk_input_ids,
                    labels=idk_labels, 
                    attention_mask=idk_attention_mask
                )
                forget_outputs_oracle = self.oracle_model(
                    forget_input_ids,
                    labels=forget_labels, 
                    attention_mask=forget_attention_mask
                )
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, labels)


            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle

            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            loss = -pi_logratios.mean()
            loss = -idk_loss_current.mean()
            outputs = forget_outputs
            
        elif self.loss_type == 'npo_grad_diff':
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(
                        input_ids,labels=labels, 
                        attention_mask=attention_mask
                    )
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
        
        return (loss, outputs) if return_outputs else loss
        
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(
            self.is_in_train, 
            args.device, 
            model.dtype, 
            self.args.dataloader_num_workers, 
            self.eval_cfg.split_list, 
            self.eval_cfg.split
        )
        
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        forget_rate = eval_cfg.split.split('_')[0]
        with torch.no_grad():
            for i, (
                folder, 
                split, 
                question_key, 
                answer_key, 
                eval_task, 
                base_answer_key, 
                perturbed_answer_key
            ) in enumerate(zip(
                eval_cfg.data_path, 
                eval_cfg.split_list, 
                eval_cfg.question_key, 
                eval_cfg.answer_key, 
                eval_cfg.eval_task, 
                eval_cfg.base_answer_key, 
                eval_cfg.perturbed_answer_key
            )):
                
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = save_filename if world_size == 1 else os.path.join(
                    curr_save_dir, 
                    f"{eval_task}_{self.accelerator.local_process_index}.json"
                )
                # print(save_filename)
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
                    eval_cfg, 
                    eval_task, 
                    self.tokenizer, 
                    folder, 
                    split, 
                    question_key, 
                    answer_key, 
                    base_answer_key, 
                    perturbed_answer_key
                )
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
                normalize_gt = False 
                # if 'eval_log' not in eval_task:
                #     normalize_gt = True

                eval_logs = get_all_evals(
                    eval_cfg, 
                    model, 
                    self.tokenizer, 
                    eval_task, 
                    eval_dataloader, 
                    base_eval_dataloader, 
                    perturb_dataloader, 
                    normalize_gt=normalize_gt
                )

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts
                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)

                            #delete old files use shutil

                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)
                                
            if self.accelerator.is_local_main_process:
                # aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    # save aggregate_stat as csv
                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), 'w') as csvfile:
                        field_names = list(aggregate_stat.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writeheader()
                        writer.writerow(aggregate_stat)

def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss