import os
import platform

import pandas as pd
import torch
import torch.nn.functional as F
import utils.functions as functions
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "bert-base-uncased"

print(f"MODEL: {MODEL_NAME}")
print("dataset v3")

train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)


text_test = train_df.loc[0]["features"]

sentences = text_test.split(". ")


tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")

sentence_embeddings = []
for sentence in sentences[:3]:
    tokens = tokeniser.encode(
        sentence, add_special_tokens=False, return_tensors="pt"
    ).to("mps")
    output = model(tokens)
    last_hidden_state = output.last_hidden_state

    sentence_embeddings.append(last_hidden_state)


x = sentence_embeddings[0].squeeze(0)
y = sentence_embeddings[1].squeeze(0)
z = sentence_embeddings[2].squeeze(0)

x = F.normalize(x, p=2, dim=-1)
y = F.normalize(y, p=2, dim=-1)
z = F.normalize(z, p=2, dim=-1)

x = x.cpu().detach().numpy()
y = y.cpu().detach().numpy()
z = z.cpu().detach().numpy()

print(x.shape)
print(y.shape)
print(z.shape)

similarity_matrix = cosine_similarity(x, y)
# print(similarity_matrix)
print(similarity_matrix[0][0])

similarity_matrix = cosine_similarity(x, z)
# print(similarity_matrix)
print(similarity_matrix[0][0])
# works
