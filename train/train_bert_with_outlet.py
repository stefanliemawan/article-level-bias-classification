import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "bert-base-uncased"

train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)

outlets_df = pd.read_csv("../dataset/outlets.csv", index_col=0)
train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
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

# v2, outlet + title + content, bert-base-uncased, slurm
# {'eval_loss': 0.9668617248535156, 'eval_accuracy': 0.7725856697819314, 'eval_precision': 0.7710658394071737, 'eval_recall': 0.7725856697819314, 'eval_f1': 0.7714502860953963, 'eval_runtime': 2.3516, 'eval_samples_per_second': 273.008, 'eval_steps_per_second': 34.445, 'epoch': 4.0}


# maybe take outlets_df and add that as additional features? how? pass through one embedding layer? is this worth doing tho??
