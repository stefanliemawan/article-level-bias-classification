# try with GNN
# one node = one chunk?
# so one graph one document?
# or one node = one document?

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.functions as functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HIDDEN_DIM = 768
MODEL_NAME = "mediabiasgroup/magpie-babe-ft"

print(f"MODEL: {MODEL_NAME}")
print("dataset v3")


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, similarity_matrix):
        x = torch.matmul(similarity_matrix, x)
        x = self.linear(x)
        x = F.relu(x)

        return x


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.gc1 = GCNLayer(input_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, similarity_matrix):
        x = self.gc1(x, similarity_matrix)
        x = self.gc2(x, similarity_matrix)

        return F.log_softmax(x, dim=1)


train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser)

print(tokenised_dataset)

functions.print_class_distribution(tokenised_dataset)


# tfidf
def create_similarity_matrix(features):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(features)

    tfidf_vectors = tfidf_matrix.toarray()

    similarity_matrix = cosine_similarity(tfidf_vectors)

    return similarity_matrix


similarity_matrix = create_similarity_matrix(train_df["features"].values)
normalised_similarity_matrix = similarity_matrix / np.sum(
    similarity_matrix, axis=1, keepdims=True
)

print(similarity_matrix)
print(normalised_similarity_matrix)

num_labels = len(pd.unique(train_df["labels"]))
model = Model(input_dim=512, hidden_dim=HIDDEN_DIM, output_dim=num_labels)
print(model)

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    optimiser.zero_grad()
    output = model(train_df["features"].values, similarity_matrix)
    loss = criterion(output, train_df["labels"].values)
    loss.backward()
    optimiser.step()

# read more on GCN
