import sys

import numpy as np
import pandas as pd
import utils.functions as functions
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from utils.chunk_model_m import ChunkModelM

CHUNK_SIZE = 156
OVERLAP = 0
NUM_TF_LAYERS = 2
HIDDEN_DIM = 768
EPOCHS = 1
DROPOUT_PROB = 0.2
TF_MODEL_NAME = "mediabiasgroup/magpie-babe-ft"


try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {TF_MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, OVERLAP {OVERLAP}, NUM_TF_LAYERS {NUM_TF_LAYERS}, HIDDEN_DIM {HIDDEN_DIM}, EPOCHS {EPOCHS}, DROPOUT {DROPOUT_PROB}"
)


train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

label_encoder = LabelEncoder()

train_metadata = label_encoder.fit_transform(train_df["outlet"].values)
test_metadata = label_encoder.fit_transform(test_df["outlet"].values)
valid_metadata = label_encoder.fit_transform(valid_df["outlet"].values)

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(TF_MODEL_NAME)


tokenised_dataset = dataset.map(
    functions.tokenise_chunks,
    fn_kwargs={"tokeniser": tokeniser, "chunk_size": CHUNK_SIZE, "overlap": OVERLAP},
)

print(tokenised_dataset)


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]
model = ChunkModelM(
    tf_model_name=TF_MODEL_NAME,
    num_tf_layers=NUM_TF_LAYERS,
    hidden_dim=HIDDEN_DIM,
    metadata_hidden_dim=int(HIDDEN_DIM / 3),
    num_classes=num_labels,
    train_labels=train_labels,
    dropout_prob=DROPOUT_PROB,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(
    tokenised_dataset["train"], train_metadata, batch_size=8
)
valid_dataloader = model.batchify(
    tokenised_dataset["valid"], valid_metadata, batch_size=8
)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"], test_metadata)
