import math
import platform

import numpy as np
import pandas as pd
import torch
import utils.functions as functions
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from torch import nn
from tqdm import tqdm
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
train_df = train_df.head(36)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)


def tokenise_dataset(x):
    features = x["features"]

    input_ids = tokeniser.encode(features, add_special_tokens=False)

    chunk_input_ids = []

    for i in range(0, len(input_ids), CHUNK_SIZE - 2):
        chunk = torch.tensor(
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# need to handle chunks inside the class
# do we need to calculate positional embedding for the transformer layer?
class Model(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_classes, train_labels):
        super(Model, self).__init__()
        if platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.init_layers(num_layers, hidden_dim, num_classes)
        self.calculate_class_weights(train_labels)
        self.init_loss_optimiser()

    def init_layers(self, num_layers, hidden_dim, num_classes):
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

    def init_loss_optimiser(self):
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights).to(self.device)
        )
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

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
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        transformer_output = bert_output.last_hidden_state

        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)

        mlp_output = self.mlp(
            transformer_output[:, 0, :]
        )  # Assuming you're only using [CLS] token

        # outputs = self.mlp(
        #     transformer_output.mean(dim=1)
        # )  # Average pooling over sequence length

        return mlp_output

    def batchify(self, inputs, batch_size=8):  # better way to do this?
        input_ids = [f["input_ids"] for f in inputs]
        attention_mask = [f["attention_mask"] for f in inputs]
        labels = torch.tensor([f["labels"] for f in inputs]).to(self.device)

        dataloader = []
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            dataloader.append([batch_input_ids, batch_attention_mask, batch_labels])

        return dataloader

    def train_loop(self, dataloader):
        total_loss = 0
        for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
            (
                input_ids_combined_tensors,
                attention_mask_combined_tensors,
                num_of_chunks,
            ) = self.handle_chunks(batch_input_ids, batch_attention_mask)

            logits = self.forward(
                input_ids_combined_tensors, attention_mask_combined_tensors
            )

            logits_split = logits.split(num_of_chunks)

            pooled_logits = torch.cat(
                [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
            )

            loss = self.loss_function(pooled_logits, batch_labels)

            self.optimiser.zero_grad()

            loss.backward()
            self.optimiser.step()

            total_loss += loss.detach().item()

        return total_loss / (len(dataloader))

    def validation_loop(self, dataloader):
        total_loss = 0
        with torch.no_grad():
            for (
                batch_input_ids,
                batch_attention_mask,
                batch_labels,
            ) in dataloader:  # use dataloader instead?
                (
                    input_ids_combined_tensors,
                    attention_mask_combined_tensors,
                    num_of_chunks,
                ) = self.handle_chunks(batch_input_ids, batch_attention_mask)

                logits = self.forward(
                    input_ids_combined_tensors, attention_mask_combined_tensors
                )

                logits_split = logits.split(num_of_chunks)

                pooled_logits = torch.cat(
                    [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
                )

                loss = self.loss_function(pooled_logits, batch_labels)

                total_loss += loss.detach().item()

        return total_loss / (len(dataloader))

    def fit(self, train_dataloader, valid_dataloader, epochs=3):
        train_loss_list, validation_loss_list = [], []

        print("Training and validating model")
        for epoch in tqdm(range(epochs)):
            print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

            train_loss = self.train_loop(train_dataloader)
            train_loss_list += [train_loss]

            validation_loss = self.validation_loop(valid_dataloader)
            validation_loss_list += [validation_loss]

            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")
            print()

        return train_loss_list, validation_loss_list

    def predict(self): ...


train_labels = tokenised_dataset["train"]["labels"]
model = Model(
    num_layers=3,
    hidden_dim=512,
    num_classes=len(CLASS_RANGES),
    train_labels=train_labels,
)
model = model.to(model.device)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader)

# pred = model()
# print(pred)
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
