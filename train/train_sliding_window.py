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
WINDOW_SIZE = 512
STRIDE = 0
MAX_CHUNKS = 5

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df = train_df.head(200)

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


def tokenise_dataset(x):
    features = x["features"]

    features_words = tokeniser.tokenize(features)

    if len(features_words) > WINDOW_SIZE * MAX_CHUNKS:
        features_words = features_words[: WINDOW_SIZE * MAX_CHUNKS]
        features = tokeniser.decode(
            tokeniser.convert_tokens_to_ids(features_words), skip_special_tokens=True
        )

    tokenised = tokeniser(
        features,
        max_length=WINDOW_SIZE,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return tokenised


tokenised_dataset = dataset.map(tokenise_dataset)

tokenised_dataset.set_format(
    "pt",
    columns=["input_ids", "attention_mask", "overflow_to_sample_mapping"],
    output_all_columns=True,
)


print(tokenised_dataset)

# print(tokenised_dataset["train"]["input_ids"][0].shape)
# print(tokenised_dataset["train"]["input_ids"][1].shape)
# print(tokenised_dataset["train"]["input_ids"][2].shape)

functions.print_class_distribution(tokenised_dataset)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_RANGES)
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        number_of_chunks = [len(x) for x in input_ids]
        print(number_of_chunks)

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

        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS)).to(
            "mps"
        )
        loss = loss_fct(
            pooled_logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def collate_fn_pooled_tokens(features):
    batch = {}

    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    label = torch.tensor([f["label"] for f in features])

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["label"] = label

    return batch


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
    data_collator=collate_fn_pooled_tokens,
)

# train_dataloader = trainer.get_train_dataloader()
# epoch_iterator = train_dataloader

# for step, inputs in enumerate(epoch_iterator):
#     print(step, inputs)
#     print(type(inputs))
#     break

# need to modifiy inner_training_loop because input_ids is a list of tensors...
trainer.train()

test = trainer.evaluate(eval_dataset=tokenised_dataset["test"])
print(test)
