import os
import platform

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import functions
from utils.sliding_window_trainer import SlidingWindowTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "bert-base-uncased"

WINDOW_SIZE = 512
STRIDE = 128
MAX_CHUNKS = 3

print(f"WINDOW_SIZE: {WINDOW_SIZE},STRIDE: {STRIDE}, MAX_CHUNKS: {MAX_CHUNKS}")
print(f"MODEL: {MODEL_NAME}")

# out of memory with 512, 0, 3

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(train_df, test_df, valid_df)


dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenise_dataset(x):
    features = x["features"]

    features_words = tokeniser.tokenize(features)

    max_len = int(WINDOW_SIZE) * int(MAX_CHUNKS)
    if len(features_words) > max_len:
        features_words = features_words[:max_len]
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
if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")


def collate_fn_pooled_tokens(features):
    batch = {}

    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = torch.tensor([f["labels"] for f in features])

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels

    return batch


# check this?, bert is significantly lower than distilbert for some reason
def compute_metrics_test(pred):
    labels = pred.label_ids.flatten().tolist()
    preds = pred.predictions.argmax(-1)

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
    epoch=4,
    compute_metrics=compute_metrics_test,
    trainer_class=SlidingWindowTrainer,
    data_collator=collate_fn_pooled_tokens,
)

# title + content, bert-base-uncased, WINDOW_SIZE: 512,STRIDE: 128, MAX_CHUNKS: 3
# {'eval_loss': 0.9980549216270447, 'eval_precision': 0.7201026448175899, 'eval_recall': 0.7185534591194969, 'eval_f1': 0.7190264075756783, 'eval_runtime': 5.7417, 'eval_samples_per_second': 110.768, 'eval_steps_per_second': 13.933, 'epoch': 4.0}
