import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "bert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser)

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


functions.train(tokenised_dataset, model, epoch=5)

# title + content, bert-base-uncased, slurm
# {'eval_loss': 0.995568037033081, 'eval_accuracy': 0.7009345794392523, 'eval_precision': 0.7023993347045271, 'eval_recall': 0.7009345794392523, 'eval_f1': 0.7015790759560087, 'eval_runtime': 2.3447, 'eval_samples_per_second': 273.812, 'eval_steps_per_second': 34.546, 'epoch': 4.0}

# title + content, bert-base-uncased, slurm
# {'eval_loss': 0.83188396692276, 'eval_accuracy': 0.6853582554517134, 'eval_precision': 0.6921642436829353, 'eval_recall': 0.6853582554517134, 'eval_f1': 0.6872441742852727, 'eval_runtime': 2.3496, 'eval_samples_per_second': 273.233, 'eval_steps_per_second': 34.473, 'epoch': 3.0}
