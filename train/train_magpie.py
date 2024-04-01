import nltk
import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "mediabiasgroup/magpie-babe-ft"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
dataset = functions.create_dataset(train_df, test_df, valid_df)


sentence_tokenizer = nltk.tokenize.sent_tokenize
word_tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenise_dataset(input):
    features = input["features"]
    sentences = sentence_tokenizer(features)

    tokenised = []
    for sentence in sentences:
        # Tokenize the sentence
        tokens = word_tokeniser.encode(sentence, return_tensors="pt")
        tokenised.append(*tokens)

    return {"input_ids": tokenised}


tokenised_dataset = dataset.map(tokenise_dataset)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

inputs = tokenised_dataset["train"]["input_ids"][0]
# print(inputs)

for input in inputs:
    input = torch.tensor([input])
    print(input)
    output = model(input)
    logits = output.get("logits")
    print(logits)

    pred = logits.argmax(-1)
    print(pred)
    break
# do this for all

# magpie is a sentence level classifier right? so split sequence into sentence, then 1 sentence 1 score from magpie, after that feed into MLP? might work
