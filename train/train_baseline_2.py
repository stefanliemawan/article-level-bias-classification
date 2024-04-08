import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

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


functions.train(tokenised_dataset, model, epoch=4)

# title + content, bert-base-uncased, with oversampling
# {'eval_loss': 0.9942390322685242, 'eval_precision': 0.7049748763399751, 'eval_recall': 0.7087227414330218, 'eval_f1': 0.7049028281123628, 'eval_runtime': 2.3508, 'eval_samples_per_second': 273.098, 'eval_steps_per_second': 34.456, 'epoch': 4.0}

# title + content, bert-base-uncased, no oversampling just weighted loss
# {'eval_loss': 0.9112138152122498, 'eval_precision': 0.6956103834554174, 'eval_recall': 0.6900311526479751, 'eval_f1': 0.6920426976542601, 'eval_runtime': 2.3555, 'eval_samples_per_second': 272.548, 'eval_steps_per_second': 34.387, 'epoch': 4.0}
