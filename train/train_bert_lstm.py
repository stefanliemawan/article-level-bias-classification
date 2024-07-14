import platform
import sys

import pandas as pd
import torch
import utils.functions as functions
from torch import nn
from transformers import AutoModel, AutoTokenizer
from utils.chunk_model import ChunkModel

CHUNK_SIZE = 156
OVERLAP = 0
HIDDEN_DIM = 256
EPOCHS = 4
DROPOUT_PROB = 0.2
TF_MODEL_NAME = "bert-base-cased"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {TF_MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, OVERLAP {OVERLAP}, HIDDEN_DIM {HIDDEN_DIM}, EPOCHS {EPOCHS}, DROPOUT {DROPOUT_PROB}"
)

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(TF_MODEL_NAME)


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
        dropout_prob=0.2,
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
        self.tf_model = AutoModel.from_pretrained(self.tf_model_name)
        self.tf_model = self.tf_model.to(self.device)

        self.bilstm_1 = nn.LSTM(
            self.tf_model.config.hidden_size, 256, bidirectional=True
        )
        self.bilstm_2 = nn.LSTM(512, 128, bidirectional=True)

        self.max_pooling = nn.MaxPool1d(kernel_size=2)

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, input_ids, attention_mask):
        tf_model_output = self.tf_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        transformer_output = tf_model_output.last_hidden_state

        bilstm_1_output, _ = self.bilstm_1(transformer_output)
        bilstm_2_output, _ = self.bilstm_2(bilstm_1_output)

        max_pooled_output = self.max_pooling(bilstm_2_output)

        flattened_max_pooled = max_pooled_output.view(max_pooled_output.size(0), -1)

        mlp = nn.Sequential(
            nn.Linear(flattened_max_pooled.shape[-1], 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        ).to(self.device)

        mlp_output = mlp(flattened_max_pooled).to(self.device)

        return mlp_output


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]
model = Model(
    tf_model_name=TF_MODEL_NAME,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_labels,
    train_labels=train_labels,
    dropout_prob=DROPOUT_PROB,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])
