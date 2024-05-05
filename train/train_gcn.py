# try with GNN
# one node = one chunk?
# so one graph one document?
# or one node = one document?

import platform

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.functions as functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HIDDEN_DIM = 16
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
    def __init__(self, input_dim, hidden_dim, output_dim, train_labels):
        super(Model, self).__init__()
        if platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.gc1 = GCNLayer(input_dim, hidden_dim).to(self.device)
        self.gc2 = GCNLayer(hidden_dim, output_dim).to(self.device)

        self.calculate_class_weights(train_labels)
        self.init_loss_optimiser()

    def calculate_class_weights(self, train_labels):
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def init_loss_optimiser(self):
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights).to(self.device)
        )
        self.optimiser = torch.optim.AdamW(
            self.parameters(), lr=1e-5
        )  # transformers default in huggingface
        # self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x, similarity_matrix):
        x = self.gc1(x, similarity_matrix)
        x = self.gc2(x, similarity_matrix)

        return F.log_softmax(x, dim=1)

    def train_loop(self, input_ids, similarity_matrix, labels):
        loss = 0
        logits = self.forward(input_ids, similarity_matrix)

        loss = self.loss_function(logits, labels)

        self.optimiser.zero_grad()

        loss.backward()
        self.optimiser.step()

        loss = loss.detach().item()

        return loss

    def validation_loop(self, input_ids, similarity_matrix, labels):
        loss = 0
        with torch.no_grad():
            logits = self.forward(input_ids, similarity_matrix)

            loss = self.loss_function(logits, labels)

            loss = loss.detach().item()

        labels = labels.tolist()
        metrics = {"loss": loss, **self.compute_metrics(logits, labels)}

        return metrics

    def compute_metrics(self, logits, labels):
        preds = logits.cpu().numpy().argmax(-1)

        # report = classification_report(labels, preds)
        # print(report)

        precision = precision_score(labels, preds, average="weighted", zero_division=1)
        recall = recall_score(labels, preds, average="weighted", zero_division=1)
        f1 = f1_score(labels, preds, average="weighted", zero_division=1)

        return {"precision": precision, "recall": recall, "f1": f1}

    def fit(self, train_inputs, valid_inputs, epochs=3):

        train_input_ids, train_similarity_matrix, train_labels = train_inputs
        valid_input_ids, valid_similarity_matrix, valid_labels = valid_inputs

        train_input_ids = torch.tensor(train_input_ids, dtype=torch.float32).to(
            self.device
        )
        train_similarity_matrix = torch.tensor(
            train_similarity_matrix, dtype=torch.float32
        ).to(self.device)
        train_labels = torch.tensor(train_labels).to(self.device)

        valid_input_ids = torch.tensor(valid_input_ids, dtype=torch.float32).to(
            self.device
        )
        valid_similarity_matrix = torch.tensor(
            valid_similarity_matrix, dtype=torch.float32
        ).to(self.device)
        valid_labels = torch.tensor(valid_labels).to(self.device)

        print("Training and validating model")
        for epoch in range(epochs):
            print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

            train_loss = self.train_loop(
                train_input_ids, train_similarity_matrix, train_labels
            )

            validation_metrics = self.validation_loop(
                valid_input_ids, valid_similarity_matrix, valid_labels
            )

            print(f"Training loss: {train_loss}")
            print(f"Validation metrics: {validation_metrics}")

            print()


train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenised_dataset = dataset.map(
    lambda x: tokeniser(
        x["features"], padding=True, truncation=True, return_tensors="pt"
    ),
    batched=True,
)

print(tokenised_dataset)

functions.print_class_distribution(tokenised_dataset)


# tfidf
def create_similarity_matrix(features):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(features)

    tfidf_vectors = tfidf_matrix.toarray()

    similarity_matrix = cosine_similarity(tfidf_vectors)

    return similarity_matrix


train_similarity_matrix = create_similarity_matrix(train_df["features"].values)
train_similarity_matrix = train_similarity_matrix / np.sum(
    train_similarity_matrix, axis=1, keepdims=True
)

print(train_similarity_matrix.shape)

valid_similarity_matrix = create_similarity_matrix(valid_df["features"].values)
valid_similarity_matrix = valid_similarity_matrix / np.sum(
    valid_similarity_matrix, axis=1, keepdims=True
)

print(valid_similarity_matrix.shape)

num_labels = len(pd.unique(train_df["labels"]))

train_labels = tokenised_dataset["train"]["labels"]
valid_labels = tokenised_dataset["valid"]["labels"]

x_shape = torch.tensor(tokenised_dataset["train"]["input_ids"][0]).shape[0]
print(x_shape)

model = Model(
    input_dim=x_shape,
    hidden_dim=HIDDEN_DIM,
    output_dim=num_labels,
    train_labels=train_labels,
)
print(model)

train_input_ids = tokenised_dataset["train"]["input_ids"]
valid_input_ids = tokenised_dataset["valid"]["input_ids"]

train_inputs = [train_input_ids, train_similarity_matrix, train_labels]
valid_inputs = [valid_input_ids, valid_similarity_matrix, valid_labels]

model.fit(train_inputs, valid_inputs, epochs=3)

# read more on GCN
# maybe dont use batches so that nodes can have relationships with others?
# hm

# validation metrics are same every epoch wuuu check why
