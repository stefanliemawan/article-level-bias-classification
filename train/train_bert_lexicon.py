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
bias_lexicon = bias_lexicon["words"].values

train_df["features"] = train_df["outlet_title_content"]
test_df["features"] = test_df["outlet_title_content"]
valid_df["features"] = valid_df["outlet_title_content"]

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
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

# v2 bert-base-uncased, no preprocessing, title + "." + content, [BIAS] after biased words
# {'eval_loss': 0.8706805109977722, 'eval_accuracy': 0.7327044025157232, 'eval_precision': 0.7294909633137111, 'eval_recall': 0.7327044025157232, 'eval_f1': 0.729644622536337, 'eval_runtime': 2.4436, 'eval_samples_per_second': 260.268, 'eval_steps_per_second': 32.738, 'epoch': 4.0}
