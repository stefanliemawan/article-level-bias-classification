import os
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f
import utils.functions as functions
from belt_nlp.bert_with_pooling import BertClassifierWithPooling
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

x_train = train_df["features"].values
y_train = train_df["label"].values

x_test = test_df["features"].values
y_test = test_df["label"].values

x_valid = valid_df["features"].values
y_valid = valid_df["label"].values


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
model = BertClassifierWithPooling(**MODEL_PARAMS)

model.fit(x_train, y_train, epochs=3)

preds = model.predict_classes(x_test)

# have to use cuda? ew
