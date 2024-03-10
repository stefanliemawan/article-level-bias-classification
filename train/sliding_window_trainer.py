import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import Trainer


class SlidingWindowTrainer(Trainer):
    def calculate_class_weights(self):
        train_labels = self.train_dataset["label"]
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label")

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

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

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

        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        labels = inputs.pop("label")

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

        return (loss, pooled_logits, labels)


# flos and floating_point_ops is set to only the first input_ids shape... not sure how this will affect the whole thing

# need to modify evaluate function now... it never ends
# theoretically you just need to modify data loader and data collator(?) try this later
# clean up