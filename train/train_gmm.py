import os
import platform

import numpy as np
import pandas as pd
import torch
import utils.functions as functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("dataset v3")

train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

train_df = pd.concat((train_df, valid_df))

tfidf_vectorizer = TfidfVectorizer()  # You can customize this further if needed
text_features = tfidf_vectorizer.fit_transform(train_df["features"].values)

# Combine text features with bias scores
combined_features = np.column_stack(
    (text_features.toarray(), train_df["labels"].values)
)

# Step 3: Model Training
# Train Gaussian Mixture Model

num_labels = len(pd.unique(train_df["labels"]))
gmm = GaussianMixture(n_components=num_labels, random_state=42)
gmm.fit(combined_features)

# Step 4: Model Evaluation
# Evaluate clustering performance using silhouette score
labels = gmm.predict(combined_features)
silhouette_avg = silhouette_score(combined_features, labels)
print("Silhouette Score:", silhouette_avg)


precision = precision_score(
    train_df["labels"].values, labels, zero_division=1, average="weighted"
)
recall = recall_score(
    train_df["labels"].values, labels, zero_division=1, average="weighted"
)
f1 = f1_score(train_df["labels"].values, labels, zero_division=1, average="weighted")

# Aggregate scores using appropriate methods (e.g., weighted average by cluster size)
overall_precision = precision.mean()
overall_recall = recall.mean()
overall_f1 = f1.mean()

print("Overall Precision:", overall_precision)
print("Overall Recall:", overall_recall)
print("Overall F1 Score:", overall_f1)

# too long to train, try vsc? may not be worth it
