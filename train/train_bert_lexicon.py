import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "bert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

bias_lexicon = pd.read_csv("dataset/bias_lexicon.csv")
bias_lexicon = bias_lexicon["words"].tolist()


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(train_df, test_df, valid_df)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokeniser.add_tokens(bias_lexicon)
tokeniser.add_special_tokens({"additional_special_tokens": ["[BIAS]"]})

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_RANGES)
)
model.resize_token_embeddings(len(tokeniser))

if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")


def tokenise_dataset(x):
    features = x["features"]

    words = tokeniser.tokenize(features)

    features_words = []
    for word in words:
        if word in bias_lexicon:
            features_words.append("[BIAS]")
        else:
            features_words.append(word)

    input_ids = tokeniser.convert_tokens_to_ids(features_words)

    max_length = tokeniser.model_max_length
    if len(input_ids) < max_length:
        input_ids += [tokeniser.pad_token_id] * (max_length - len(input_ids))
    elif len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    attention_mask = [1] * len(input_ids)

    tokenised = {"input_ids": input_ids, "attention_mask": attention_mask}

    return tokenised


tokenised_dataset = dataset.map(tokenise_dataset)

functions.print_class_distribution(tokenised_dataset)


functions.train(tokenised_dataset, model, epoch=4)
