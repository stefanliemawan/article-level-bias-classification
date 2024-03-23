import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "bert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

bias_lexicon = pd.read_csv("dataset/bias_lexicon.csv")
bias_lexicon = bias_lexicon["words"].values


dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenise_dataset(x):
    features = x["features"]

    words = tokeniser.tokenize(features)

    features_words = []
    for word in words:
        features_words.append(word)
        if word in bias_lexicon:
            features_words.append("[BIAS]")

    tokenised = tokeniser(features, padding="max_length", truncation=True)

    return tokenised


tokenised_dataset = dataset.map(tokenise_dataset)

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


functions.train(tokenised_dataset, model, epoch=5)
