import os
import platform

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import functions
from utils.sliding_window_trainer import SlidingWindowTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "bert-base-uncased"

WINDOW_SIZE = 512
STRIDE = 256
MAX_CHUNKS = 4

print(f"WINDOW_SIZE: {WINDOW_SIZE},STRIDE: {STRIDE}, MAX_CHUNKS: {MAX_CHUNKS}")
print(f"MODEL: {MODEL_NAME}")

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(train_df, test_df, valid_df)


dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenise_dataset(input):
    features = input["features"]

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

num_labels = len(pd.unique(train_df["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)
if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")


# check this?
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
    epoch=4,
    trainer_class=SlidingWindowTrainer,
    data_collator=collate_fn_pooled_tokens,
)

# title + content, bert-base-uncased, WINDOW_SIZE: 512,STRIDE: 256, MAX_CHUNKS: 3
# {'eval_loss': 0.9750634431838989, 'eval_precision': 0.7152073894856094, 'eval_recall': 0.7133956386292835, 'eval_f1': 0.7142062154976148, 'eval_runtime': 6.8593, 'eval_samples_per_second': 93.595, 'eval_steps_per_second': 11.809, 'epoch': 4.0}

# title + content, bert-base-uncased, WINDOW_SIZE: 512,STRIDE: 256, MAX_CHUNKS: 4
# {'eval_loss': 0.9101153612136841, 'eval_precision': 0.7298301576586326, 'eval_recall': 0.7242990654205608, 'eval_f1': 0.7255672362667162, 'eval_runtime': 7.3774, 'eval_samples_per_second': 87.023, 'eval_steps_per_second': 10.98, 'epoch': 4.0}
