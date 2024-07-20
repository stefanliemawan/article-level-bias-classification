import os
import platform
import sys

import nltk
import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_LENGTH = 512
MODEL_NAME = "mediabiasgroup/magpie-babe-ft"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)
train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
dataset = functions.create_dataset(train_df, test_df, valid_df)


sentence_tokenizer = nltk.tokenize.sent_tokenize
word_tokeniser = AutoTokenizer.from_pretrained("mediabiasgroup/magpie-babe-ft")

word_tokeniser.add_special_tokens({"additional_special_tokens": ["[BS]", "[UBS]"]})

magpie_model = AutoModelForSequenceClassification.from_pretrained(
    "mediabiasgroup/magpie-babe-ft", ignore_mismatched_sizes=True
)
magpie_model.resize_token_embeddings(len(word_tokeniser))


def tokenise_dataset(input):
    features = input["features"]
    labels = input["labels"]

    sentences = sentence_tokenizer(features)

    tokenised = torch.tensor([], dtype=torch.int16)
    for sentence in sentences:
        tokens = word_tokeniser.encode(
            sentence, add_special_tokens=False, return_tensors="pt"
        )

        if tokens.shape[1] > MAX_LENGTH:
            tokens = tokens[:, :MAX_LENGTH]

        output = magpie_model(tokens)

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

    attention_mask = torch.ones(
        len(tokenised), dtype=tokenised.dtype, device=tokenised.device
    )

    if len(tokenised) >= MAX_LENGTH:
        tokenised = tokenised[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
    else:

        num_zeros = MAX_LENGTH - len(tokenised)
        zeros = torch.zeros(num_zeros, dtype=tokenised.dtype, device=tokenised.device)
        tokenised = torch.cat((tokenised, zeros), dim=0)
        attention_mask = torch.cat((attention_mask, zeros), dim=0)

    return {"input_ids": tokenised, "attention_mask": attention_mask, "labels": labels}


tokenised_dataset = dataset.map(tokenise_dataset)
print(tokenised_dataset)

num_labels = len(pd.unique(train_df["labels"]))
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels
)
bert_model.resize_token_embeddings(len(word_tokeniser))


if platform.system() == "Darwin":
    bert_model = bert_model.to("mps")
elif torch.cuda.is_available():
    bert_model = bert_model.to("cuda")
else:
    bert_model = bert_model.to("cpu")


functions.train(tokenised_dataset, bert_model, epochs=4)

# {'eval_loss': 1.0878552198410034, 'eval_precision': 0.7520404499179938, 'eval_recall': 0.45482866043613707, 'eval_f1': 0.28438965491938334, 'eval_runtime': 2.2536, 'eval_samples_per_second': 284.877, 'eval_steps_per_second': 35.942, 'epoch': 4.0}
