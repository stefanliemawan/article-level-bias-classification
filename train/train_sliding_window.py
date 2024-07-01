import os
import platform
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import functions
from utils.sliding_window_trainer import SlidingWindowTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "bert-base-cased"

WINDOW_SIZE = 512
STRIDE = 256
MAX_CHUNKS = 3

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"WINDOW_SIZE: {WINDOW_SIZE},STRIDE: {STRIDE}, MAX_CHUNKS: {MAX_CHUNKS}")
print(f"MODEL: {MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
#     train_df, test_df, valid_df
# )


dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)


tokenised_dataset = dataset.map(
    functions.tokenise_chunks,
    fn_kwargs={
        "tokeniser": tokeniser,
        "chunk_size": WINDOW_SIZE,
    },
)

print(tokenised_dataset)

functions.print_class_distribution(tokenised_dataset)

num_labels = len(pd.unique(train_df["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)

if platform.system() == "Darwin":
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = model.to(device)


def collate_fn_pooled_tokens(features):
    batch = {}

    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = torch.tensor([f["labels"] for f in features])

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels

    return batch


functions.train(
    tokenised_dataset,
    model,
    epochs=4,
    trainer_class=SlidingWindowTrainer,
    data_collator=collate_fn_pooled_tokens,
)

# vx + rescraped, title + content, bert-base-cased, WINDOW_SIZE: 512,STRIDE: 256, MAX_CHUNKS: 3
#               precision    recall  f1-score   support

#            0       0.45      0.48      0.46        27
#            1       0.39      0.52      0.45        54
#            2       0.42      0.50      0.46       103
#            3       0.91      0.81      0.86       384

#     accuracy                           0.71       568
#    macro avg       0.54      0.58      0.56       568
# weighted avg       0.75      0.71      0.73       568


# 100%|██████████| 71/71 [00:07<00:00,  9.68it/s]
# {'eval_loss': 0.8890448808670044, 'eval_precision': 0.7472220951363592, 'eval_recall': 0.7112676056338029, 'eval_f1': 0.7257911489910044, 'eval_runtime': 7.4861, 'eval_samples_per_second': 75.874, 'eval_steps_per_second': 9.484, 'epoch': 4.0}

# vx + rescraped, outlet + title + content, bert-base-cased, WINDOW_SIZE: 512,STRIDE: 256, MAX_CHUNKS: 3
#               precision    recall  f1-score   support

#            0       0.46      0.41      0.43        27
#            1       0.41      0.48      0.44        54
#            2       0.46      0.56      0.51       103
#            3       0.91      0.84      0.87       384

#     accuracy                           0.74       568
#    macro avg       0.56      0.57      0.56       568
# weighted avg       0.76      0.74      0.75       568


# {'eval_loss': 0.8257732391357422, 'eval_precision': 0.7585514237108042, 'eval_recall': 0.7359154929577465, 'eval_f1': 0.7451975162272549, 'eval_runtime': 7.4763, 'eval_samples_per_second': 75.973, 'eval_steps_per_second': 9.497, 'epoch': 4.0}
