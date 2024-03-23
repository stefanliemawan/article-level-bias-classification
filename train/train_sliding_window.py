import argparse
import os
import platform

import pandas as pd
import torch
from custom_trainer.sliding_window_trainer import SlidingWindowTrainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import functions

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("-ws", "--windowsize", help="Window Size")
parser.add_argument("-s", "--stride", help="Stride")
parser.add_argument("-mc", "--maxchunks", help="Max Chunks")
parser.add_argument("-m", "--model", help="Model")

args = parser.parse_args()


SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = args.model if args.model else "distilbert-base-uncased"

WINDOW_SIZE = int(args.windowsize) if args.windowsize else 512
STRIDE = int(args.stride) if args.stride else 0
MAX_CHUNKS = int(args.maxchunks) if args.maxchunks else 2

BATCH_SIZE = 8

print(
    f"WINDOW_SIZE: {WINDOW_SIZE},STRIDE: {STRIDE}, MAX_CHUNKS: {MAX_CHUNKS}, BATCH_SIZE: {BATCH_SIZE}"
)
print(f"MODEL: {MODEL_NAME}")

# out of memory with 512, 0, 3

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

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
    epoch=5,
    batch_size=BATCH_SIZE,
    compute_metrics=compute_metrics_test,
    trainer_class=SlidingWindowTrainer,
    data_collator=collate_fn_pooled_tokens,
)

# 512-0-2, title + "." + content, distilbert-base-uncased, mean pooling
# {'eval_loss': 0.7615396976470947, 'eval_precision': 0.7054318202676083, 'eval_recall': 0.7044025157232704, 'eval_f1': 0.7038882317040644, 'eval_runtime': 2.0838, 'eval_samples_per_second': 305.217, 'eval_steps_per_second': 38.392, 'epoch': 3.0}

# 512-0-5, title + "." + content, distilbert-base-uncased, mean pooling
# {'eval_loss': 0.7613411545753479, 'eval_precision': 0.7114559339296948, 'eval_recall': 0.7091194968553459, 'eval_f1': 0.7093012602260368, 'eval_runtime': 653.704, 'eval_samples_per_second': 0.973, 'eval_steps_per_second': 0.122, 'epoch': 3.0}

# 512-128-3, title + "." + content, distilbert-base-uncased, mean pooling
# {'eval_loss': 0.7468533515930176, 'eval_precision': 0.7139872308746571, 'eval_recall': 0.7122641509433962, 'eval_f1': 0.7116785211153052, 'eval_runtime': 797.7862, 'eval_samples_per_second': 0.797, 'eval_steps_per_second': 0.1, 'epoch': 3.0}

# 512-0-2, title + "." + content, bert-base-uncased, mean pooling
# {'eval_loss': 1.0668894052505493, 'eval_precision': 0.6903830741260983, 'eval_recall': 0.6855345911949685, 'eval_f1': 0.6867447311342364, 'eval_runtime': 3.9405, 'eval_samples_per_second': 161.401, 'eval_steps_per_second': 20.302, 'epoch': 5.0}

# 512-128-3, title + "." + content, bert-base-uncased, mean pooling
# {'eval_loss': 0.9941141605377197, 'eval_precision': 0.6383857268610628, 'eval_recall': 0.4811320754716981, 'eval_f1': 0.35640246016125765, 'eval_runtime': 4.2604, 'eval_samples_per_second': 149.281, 'eval_steps_per_second': 18.778, 'epoch': 3.0}

# 512-128-3, title + "." + content, bert-base-uncased, mean pooling
# {'eval_loss': 1.3030740022659302, 'eval_precision': 0.722740737298317, 'eval_recall': 0.7216981132075472, 'eval_f1': 0.7211264330676727, 'eval_runtime': 5.3687, 'eval_samples_per_second': 118.464, 'eval_steps_per_second': 14.901, 'epoch': 5.0}
