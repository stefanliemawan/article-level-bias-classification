import os
import platform

import nltk
import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained("mediabiasgroup/magpie-babe-ft")
magpie_model = AutoModelForSequenceClassification.from_pretrained(
    "mediabiasgroup/magpie-babe-ft",
    ignore_mismatched_sizes=True,
)

if platform.system() == "Darwin":
    magpie_model = magpie_model.to("mps")
elif torch.cuda.is_available():
    magpie_model = magpie_model.to("cuda")
else:
    magpie_model = magpie_model.to("cpu")


tokenised_dataset = dataset.map(
    lambda x: tokeniser(
        x["features"], padding="max_length", truncation=True, return_tensors="pt"
    ),
    batched=True,
)
print(tokenised_dataset)

input = torch.tensor(tokenised_dataset["train"]["input_ids"][0]).to("mps")
input = input.unsqueeze(0)

output = magpie_model(input, output_hidden_states=True)
hidden_states = output.hidden_states[0]

print(hidden_states.shape)

# this works, lets do something with this hidden_states soon
