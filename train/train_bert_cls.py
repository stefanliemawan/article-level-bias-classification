import math
import platform

import numpy as np
import pandas as pd
import torch
import utils.functions as functions
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
)

# print(platform.system())
# print(platform.platform())

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]
CHUNK_SIZE = 128

# MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MODEL_NAME = "bert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
train_df = train_df.head(100)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(train_df, test_df, valid_df)

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)


def tokenise_dataset(x):
    features = x["features"]

    input_ids = tokeniser.encode(features, add_special_tokens=False)

    chunk_input_ids = []

    for i in range(0, len(input_ids), CHUNK_SIZE - 2):
        chunk = (
            [tokeniser.cls_token_id]
            + input_ids[i : i + CHUNK_SIZE - 2]
            + [tokeniser.sep_token_id]
        )
        if len(chunk) < CHUNK_SIZE:
            break
            # chunk = chunk + ([0] * (CHUNK_SIZE - len(chunk)))  # pad until 128, has to handle attention mask 0 for pad 0
        chunk_input_ids.append(chunk)

    chunk_attention_masks = [[1] * len(chunk) for chunk in chunk_input_ids]

    return {"input_ids": chunk_input_ids, "attention_mask": chunk_attention_masks}


tokenised_dataset = dataset.map(tokenise_dataset)

print(tokenised_dataset)
# so idea here is to split into chunks of word encodings, do model() model() model(), get the CLS repr, then feed into torch_model


# need to handle chunks inside the class
# do we need to calculate positional embedding for the transformer layer?
class Model(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_classes):
        super(Model, self).__init__()
        if platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.bert = BertModel.from_pretrained(TRANSFORMER_MODEL_NAME)
        self.bert = self.bert.to(self.device)

        # Define transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.bert.config.hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Define Multilayer Perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def calculate_class_weights(self, train_labels):
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def handle_chunks(self, input_ids, attention_mask):
        num_of_chunks = [len(x) for x in input_ids]

        input_ids_combined = []
        for id in input_ids:
            input_ids_combined.extend(id)

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.device) for x in input_ids_combined]
        )

        attention_mask_combined = []
        for mask in attention_mask:
            attention_mask_combined.extend(mask)

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.device) for x in attention_mask_combined]
        )

        return (
            input_ids_combined_tensors,
            attention_mask_combined_tensors,
            num_of_chunks,
        )

    def forward(self, input_ids, attention_mask):
        print(input_ids)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        print(bert_output)  # nan, why, need embedding?

        # Apply transformer layers
        transformer_output = bert_output
        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)

        # Apply MLP
        outputs = self.mlp(
            transformer_output.mean(dim=1)
        )  # Average pooling over sequence length

        return outputs

    def compute_loss(self, logits, labels, num_of_chunks, return_outputs=False):
        logits_split = logits.split(num_of_chunks)

        pooled_logits = torch.cat(
            [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
        )

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights).to(self.device)
        )
        loss = loss_fct(
            pooled_logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )

        return (loss, logits) if return_outputs else loss

    def prediction_step(self, inputs, batch_size=8):
        input_ids = [f["input_ids"] for f in inputs]
        attention_mask = [f["attention_mask"] for f in inputs]
        labels = torch.tensor([f["labels"] for f in inputs])

        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = input_ids[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            with torch.no_grad():
                (
                    input_ids_combined_tensors,
                    attention_mask_combined_tensors,
                    num_of_chunks,
                ) = self.handle_chunks(batch_input_ids, batch_attention_mask)

                outputs = self.forward(
                    input_ids_combined_tensors, attention_mask_combined_tensors
                )
                print(outputs)
                # no logits in outputs, logits =

                # logits = outputs
                # loss, outputs = self.compute_loss(
                #     logits, batch_labels, num_of_chunks, return_outputs=True
                # )
                # loss = loss.mean().detach()

                # number_of_chunks = [len(x) for x in inputs["input_ids"]]
                # logits_split = logits.split(number_of_chunks)

                # pooled_logits = torch.cat(
                #     [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
                # )
                # logits = pooled_logits

                # print(logits)

            break

    def train_loop(self):
        self.train()
        total_loss = 0

        # compute_loss per batch

    def validation_loop(self): ...

    def fit(self): ...


model = Model(num_layers=3, hidden_dim=256, num_classes=len(CLASS_RANGES))
model = model.to(model.device)

model.prediction_step(tokenised_dataset["train"], batch_size=8)
# print(model)

# opt = torch.optim.SGD(model.parameters(), lr=0.01)  # put this inside class

# train_input_ids = tokenised_dataset["train"]["input_ids"]
# train_attention_mask = tokenised_dataset["train"]["attention_mask"]
# with torch.no_grad():
#     logits, num_of_chunks = model(train_input_ids, train_attention_mask)


# in the paper, they divide to equal chunks, then use the CLS tokens as repr for each chunks (su)


# so no finetune, but instead do like 6 layers of transformers to get representation, then classify?
# input -> word encode -> pooling -> cls encode -> pooling -> classification
# implement this
