import sys

import numpy as np
import pandas as pd
import utils.functions as functions
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from utils.chunk_model_m import ChunkModelM

CHUNK_SIZE = 156
NUM_TF_LAYERS = 2
HIDDEN_DIM = 384
METADATA_HIDDEN_DIM = 128
EPOCHS = 3
DROPOUT_PROB = 0.2
TF_MODEL_NAME = "mediabiasgroup/magpie-babe-ft"
POOLING_STRATEGY = "mean"


try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {TF_MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, POOLING_STRATEGY {POOLING_STRATEGY}, NUM_TF_LAYERS {NUM_TF_LAYERS}, HIDDEN_DIM {HIDDEN_DIM}, EPOCHS {EPOCHS}, DROPOUT {DROPOUT_PROB}"
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
    fn_kwargs={"tokeniser": tokeniser, "chunk_size": CHUNK_SIZE},
)

print(tokenised_dataset)


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]
model = ChunkModelM(
    tf_model_name=TF_MODEL_NAME,
    num_tf_layers=NUM_TF_LAYERS,
    hidden_dim=HIDDEN_DIM,
    metadata_hidden_dim=METADATA_HIDDEN_DIM,
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

# mediabiasgroup/magpie-babe-ft, dataset vx, CHUNK_SIZE 156, OVERLAP 0, NUM_TF_LAYERS 2, HIDDEN_DIM 768, EPOCHS 3, DROPOUT 0.2, title + content, warmup_steps 162, lr 1e-05
#               precision    recall  f1-score   support

#            0       0.46      0.63      0.53        27
#            1       0.40      0.46      0.43        54
#            2       0.39      0.51      0.44       104
#            3       0.93      0.80      0.86       384

#     accuracy                           0.71       569
#    macro avg       0.54      0.60      0.57       569
# weighted avg       0.76      0.71      0.73       569

# {'loss': 0.8577841520309448, 'precision': 0.7555730957707322, 'recall': 0.7100175746924429, 'f1': 0.727705275815567}
