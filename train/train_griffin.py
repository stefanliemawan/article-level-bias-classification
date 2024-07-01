import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "acrastt/Griffin-3B"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, legacy=False)
tokeniser.add_special_tokens({"pad_token": "[PAD]"})
# tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser)

tokenised_dataset = dataset.map(
    lambda x: tokeniser(x["features"], padding="longest", truncation=True),
    batched=True,
)

print(tokenised_dataset)
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


functions.train(tokenised_dataset, model, epochs=4)

# title + content, bert-base-uncased, with oversampling
# {'eval_loss': 0.8049562573432922, 'eval_precision': 0.6989735466648578, 'eval_recall': 0.7040498442367601, 'eval_f1': 0.6967025255248073, 'eval_runtime': 2.3663, 'eval_samples_per_second': 271.307, 'eval_steps_per_second': 34.23, 'epoch': 4.0}

# title + content, bert-base-uncased, no oversampling just weighted loss
# {'eval_loss': 1.3561333417892456, 'eval_precision': 0.7101609811080607, 'eval_recall': 0.7102803738317757, 'eval_f1': 0.709418672887263, 'eval_runtime': 2.3497, 'eval_samples_per_second': 273.223, 'eval_steps_per_second': 34.472, 'epoch': 4.0}
