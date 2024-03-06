import os
from collections.abc import Mapping

import functions
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "distilbert-base-uncased"

# WINDOW_SIZE = 512
# STRIDE = 256
WINDOW_SIZE = 256
STRIDE = 128

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df["features"] = train_df.apply(functions.preprocess_content, axis=1)
test_df["features"] = test_df.apply(functions.preprocess_content, axis=1)
valid_df["features"] = valid_df.apply(functions.preprocess_content, axis=1)

train_dataset = Dataset.from_pandas(
    train_df[["features", "label"]], preserve_index=False
)
test_dataset = Dataset.from_pandas(test_df[["features", "label"]], preserve_index=False)
valid_dataset = Dataset.from_pandas(
    valid_df[["features", "label"]], preserve_index=False
)

dataset = DatasetDict(
    {"train": train_dataset, "test": test_dataset, "valid": valid_dataset}
)

train_labels = train_df["label"].values
class_weights = np.asarray(
    compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
).astype(np.float32)

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenised_dataset = dataset.map(
    lambda x: tokeniser(
        x["features"],
        max_length=WINDOW_SIZE,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ),
)

tokenised_dataset.set_format(
    "pt",
    columns=["input_ids", "attention_mask", "overflow_to_sample_mapping"],
    output_all_columns=True,
)

print(tokenised_dataset)

functions.print_class_distribution(tokenised_dataset)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_RANGES)
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label")
        outputs = model(**inputs)

        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS)).to(
            "mps"
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def torch_default_data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["label"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["label"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["label"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    print(first)
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            elif isinstance(v, list):
                features = torch.tensor(features)
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


# RuntimeError: stack expects each tensor to be equal size, but got [3, 256] at entry 0 and [1, 256] at entry 1
# some rows have 3 chunks, some have 5 chunks, not sure how to handle


training_args = TrainingArguments(
    output_dir="test_trainer",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    save_total_limit=2,
    save_strategy="no",
    load_best_model_at_end=False,
)

CLASS_WEIGHTS = class_weights
print(f"class_weights: {CLASS_WEIGHTS}")

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenised_dataset["train"],
    eval_dataset=tokenised_dataset["valid"],
    compute_metrics=functions.compute_metrics_classification,
    data_collator=torch_default_data_collator,
)


trainer.train()

test = trainer.evaluate(eval_dataset=tokenised_dataset["test"])
print(test)

# TODO - handle the aggregation, some functions within Trainer or data collator?
