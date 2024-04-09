import os

import numpy as np
import pandas as pd
import utils.functions as functions
from belt_nlp.bert_with_pooling import BertClassifierWithPooling
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

os.environ["TOKENIZERS_PARALLELISM"] = "true"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

x_train = train_df["features"].values
y_train = train_df["labels"].values

x_test = test_df["features"].values
y_test = test_df["labels"].values

x_valid = valid_df["features"].values
y_valid = valid_df["labels"].values


class_weights = np.asarray(
    compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
).astype(np.float32)

MODEL_PARAMS = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 3,
    "chunk_size": 510,
    "stride": 510,
    "minimal_chunk_length": 510,
    "pooling_strategy": "mean",
}
model = BertClassifierWithPooling(**MODEL_PARAMS, device="cuda")

model.fit(x_train, y_train, epochs=3)

preds = model.predict_classes(x_test)
labels = y_test

precision = precision_score(labels, preds, average="weighted", zero_division=1)
recall = recall_score(labels, preds, average="weighted", zero_division=1)
f1 = f1_score(labels, preds, average="weighted", zero_division=1)

print(
    {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
)

# have to use cuda? ew
# whats belt again? i dont remember
