import math
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import Trainer


class SlidingWindowTrainer(Trainer):
    def calculate_class_weights(self):
        train_labels = self.train_dataset["labels"]
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        number_of_chunks = [len(x) for x in input_ids]

        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to("mps") for x in input_ids_combined]
        )

        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to("mps") for x in attention_mask_combined]
        )

        outputs = model(input_ids_combined_tensors, attention_mask_combined_tensors)
        logits = outputs.get("logits")

        logits_split = logits.split(number_of_chunks)

        pooled_logits = torch.cat(
            [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
        )

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights)
        ).to("mps")
        loss = loss_fct(
            pooled_logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if "input_ids" in input_dict:
            return input_dict["input_ids"][0].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            self.warnings_issued["estimate_tokens"] = True
        return 0

    def floating_point_ops(
        self,
        input_dict: Dict[str, Union[torch.Tensor, Any]],
        exclude_embeddings: bool = True,
    ) -> int:

        return 6 * self.estimate_tokens(input_dict)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                loss = loss.mean().detach()

                logits = outputs.get("logits")
                number_of_chunks = [len(x) for x in inputs["input_ids"]]
                logits_split = logits.split(number_of_chunks)

                pooled_logits = torch.cat(
                    [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
                )
                logits = pooled_logits

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()


# flos and floating_point_ops is set to only the first input_ids shape... not sure how this will affect the whole thing

# need to modify evaluate function now... it never ends
# theoretically you just need to modify data loader and data collator(?) try this later
# clean up
