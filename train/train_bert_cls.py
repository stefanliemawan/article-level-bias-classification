import math
import platform

import numpy as np
import pandas as pd
import torch
import utils.functions as functions
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
)

# print(platform.system())
# print(platform.platform())

N_DIRECT = 256
SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

# MODEL_NAME = "distilbert-base-uncased"
MODEL_NAME = "bert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model = model.to("mps")

# so idea here is to split into chunks of word encodings, do model() model() model(), get the CLS repr, then feed into torch_model

encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8
)  # not sure if i can add this to sequential?
torch_model = nn.Sequential(
    nn.Linear(764, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
    nn.Sigmoid(),
)

# https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html


# def get_embedding(tokens):
#     tokens = [tokeniser.cls_token_id, *tokens, tokeniser.sep_token_id]
#     tokens = torch.tensor([tokens]).to("mps")

#     with torch.no_grad():
#         outputs = model(tokens)

#     embedding = outputs.last_hidden_state
#     embedding = embedding[0][0]

#     return embedding


# features = train_df["features"]

# tokenised_features = tokeniser.encode(features, add_special_tokens=False)

# # if len(tokenised_features) < 510:
# #     return tokeniser(features, padding="max_length")

# direct_tokens = tokenised_features[:N_DIRECT]
# non_direct_tokens = tokenised_features[N_DIRECT:]

# direct_embeddings = get_embedding(direct_tokens)

# n_split = math.ceil(len(non_direct_tokens) / 510)
# non_direct_tokens = np.array_split(non_direct_tokens, n_split)

# cls_embeddings = []
# for tokens in non_direct_tokens:
#     cls_embedding = get_embedding(tokens)
#     cls_embeddings.append(cls_embedding)

# print(direct_embeddings.shape)  # this is one article embeddings thooo not good

# in the paper, they divide to equal chunks, then use the CLS tokens as repr for each chunks (su)


# so no finetune, but instead do like 6 layers of transformers to get representation, then classify?
# input -> word encode -> pooling -> cls encode -> pooling -> classification
# implement this
