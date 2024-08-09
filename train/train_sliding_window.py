import os
import platform
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import functions
from utils.chunk_model import ChunkModel
from utils.sliding_window_trainer import SlidingWindowTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "bert-base-cased"

CHUNK_SIZE = 256
OVERLAP = 0
EPOCHS = 4

POOLING_STRATEGY = "cls"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(
    f"WINDOW_SIZE: {CHUNK_SIZE},STRIDE: {OVERLAP}, POOLING_STRATEGY {POOLING_STRATEGY}"
)
print(f"MODEL: {MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
#     train_df, test_df, valid_df
# )

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenised_dataset = dataset.map(
    functions.tokenise_chunks,
    fn_kwargs={
        "tokeniser": tokeniser,
        "chunk_size": CHUNK_SIZE,
    },
)

print(tokenised_dataset)


class Model(ChunkModel):
    def __init__(
        self,
        tf_model_name,
        hidden_dim,
        num_classes,
        train_labels,
        dropout_prob=0,
    ):
        super(ChunkModel, self).__init__()
        if platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tf_model_name = tf_model_name
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.init_layers()
        self.calculate_class_weights(train_labels)
        self.init_loss_optimiser()

    def init_layers(self):
        self.tf_model = AutoModelForSequenceClassification.from_pretrained(
            self.tf_model_name, num_labels=self.num_classes
        )
        self.tf_model = self.tf_model.to(self.device)

    def forward(self, input_ids, attention_mask):
        tf_model_output = self.tf_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        logits = tf_model_output.logits

        return logits


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]
model = Model(
    tf_model_name=MODEL_NAME,
    hidden_dim=None,
    num_classes=num_labels,
    train_labels=train_labels,
    dropout_prob=0,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])


# vx, bert-base-cased, WINDOW_SIZE: 512, STRIDE: 0, title + content, warmup_steps: 216, learning_rate: 1e-05, CLS pooling
#               precision    recall  f1-score   support

#            0       0.55      0.44      0.49        27
#            1       0.42      0.56      0.48        54
#            2       0.40      0.57      0.47       104
#            3       0.92      0.79      0.85       384

#     accuracy                           0.71       569
#    macro avg       0.57      0.59      0.57       569
# weighted avg       0.76      0.71      0.73       569

# {'loss': 0.8862332105636597, 'precision': 0.7628740574350716, 'recall': 0.7117750439367311, 'f1': 0.7301880244092894}

# vx, bert-base-cased, WINDOW_SIZE: 156, STRIDE: 0, title + content, warmup_steps: 216, learning_rate: 1e-05, mean pooling
#               precision    recall  f1-score   support

#            0       0.41      0.56      0.47        27
#            1       0.43      0.52      0.47        54
#            2       0.41      0.53      0.46       104
#            3       0.92      0.80      0.85       384

#     accuracy                           0.71       569
#    macro avg       0.54      0.60      0.56       569
# weighted avg       0.76      0.71      0.73       569

# {'loss': 0.8757110238075256, 'precision': 0.755288078095832, 'recall': 0.7100175746924429, 'f1': 0.7274181581251454}

# vx, bert-base-cased, WINDOW_SIZE: 256,STRIDE: 0, POOLING_STRATEGY cls
#               precision    recall  f1-score   support

#            0       0.45      0.56      0.50        27
#            1       0.41      0.54      0.46        54
#            2       0.39      0.48      0.43       104
#            3       0.91      0.80      0.85       384

#     accuracy                           0.70       569
#    macro avg       0.54      0.59      0.56       569
# weighted avg       0.74      0.70      0.72       569

# {'loss': 0.9027590751647949, 'precision': 0.7432669905182095, 'recall': 0.70298769771529, 'f1': 0.7189340771476121}
