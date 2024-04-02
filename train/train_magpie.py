import platform

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


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, ignore_mismatched_sizes=True
)

sentence_tokenizer = nltk.tokenize.sent_tokenize
word_tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

word_tokeniser.add_special_tokens({"additional_special_tokens": ["[BS]", "[UBS]"]})


def tokenise_dataset(input):
    features = input["features"]
    sentences = sentence_tokenizer(features)

    tokenised = torch.tensor([], dtype=torch.int16)
    for sentence in sentences:
        tokens = word_tokeniser.encode(
            sentence, add_special_tokens=False, return_tensors="pt"
        )

        if tokens.shape[1] > 512:
            tokens = tokens[:, :512]

        output = model(tokens)
        # RuntimeError: The expanded size of the tensor (793) must match the existing size (514) at non-singleton dimension 1.  Target sizes: [1, 793].  Tensor sizes: [1, 514]
        logits = output.get("logits")
        pred = logits.argmax(-1)

        if pred.item() == 1:
            sentence_bias = word_tokeniser.encode(
                "[BS]", add_special_tokens=False, return_tensors="pt"
            )
        else:
            sentence_bias = word_tokeniser.encode(
                "[UBS]", add_special_tokens=False, return_tensors="pt"
            )

        tokens = torch.cat((sentence_bias[0], tokens[0]), dim=0)

        tokenised = torch.cat((tokenised, tokens), dim=0)

    tokenised = tokenised[:512]

    return {"input_ids": tokenised}


tokenised_dataset = dataset.map(tokenise_dataset)


inputs = tokenised_dataset["train"]["input_ids"][0]
print(inputs)

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

# do this for all

# magpie is a sentence level classifier right? so split sequence into sentence, then 1 sentence 1 score from magpie, after that feed into MLP? might work
