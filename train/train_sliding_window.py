import os
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f
from custom_trainer.sliding_window_trainer import SlidingWindowTrainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import functions

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "distilbert-base-uncased"

# WINDOW_SIZE = 512
# STRIDE = 256
WINDOW_SIZE = 512
STRIDE = 0
MAX_CHUNKS = 5  # out of memory even with chunks 2, maybe something is wrong

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df["features"] = train_df.apply(functions.preprocess_content, axis=1)
test_df["features"] = test_df.apply(functions.preprocess_content, axis=1)
valid_df["features"] = valid_df.apply(functions.preprocess_content, axis=1)

train_dataset = Dataset.from_pandas(
    train_df[["features", "labels"]], preserve_index=False
)
test_dataset = Dataset.from_pandas(
    test_df[["features", "labels"]], preserve_index=False
)
valid_dataset = Dataset.from_pandas(
    valid_df[["features", "labels"]], preserve_index=False
)

dataset = DatasetDict(
    {"train": train_dataset, "test": test_dataset, "valid": valid_dataset}
)

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


functions.print_class_distribution(tokenised_dataset)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_RANGES)
)


def collate_fn_pooled_tokens(features):
    batch = {}

    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = torch.tensor([f["labels"] for f in features])

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels

    return batch


def compute_metrics_test(pred):
    labels = pred.label_ids.flatten().tolist()
    preds = pred.predictions.argmax(-1)

    print(len(labels))
    print(len(preds))

    precision = precision_score(labels, preds, average="weighted", zero_division=1)
    recall = recall_score(labels, preds, average="weighted", zero_division=1)
    f1 = f1_score(labels, preds, average="weighted", zero_division=1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


functions.train(
    tokenised_dataset,
    model,
    epoch=3,
    compute_metrics=compute_metrics_test,
    trainer_class=SlidingWindowTrainer,
    data_collator=collate_fn_pooled_tokens,
)
