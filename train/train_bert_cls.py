import math
import platform

import numpy as np
import pandas as pd
import torch
import utils.functions as functions
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import AutoModel, AutoTokenizer

CHUNK_SIZE = 512
NUM_TF_LAYERS = 2
HIDDEN_SIZE = 768
EPOCHS = 14
TRANSFORMER_MODEL_NAME = "bert-base-cased"

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, NUM_TF_LAYERS {NUM_TF_LAYERS}, HIDDEN_SIZE {HIDDEN_SIZE}, EPOCHS {EPOCHS}, TRANSFORMER_MODEL_NAME {TRANSFORMER_MODEL_NAME}"
)

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)


def tokenise_dataset(x):
    features = x["features"]

    input_ids = tokeniser.encode(features, add_special_tokens=False)

    chunk_input_ids = []
    chunk_attention_masks = []

    for i in range(0, len(input_ids), CHUNK_SIZE - 2):
        chunk = (
            [tokeniser.cls_token_id]
            + input_ids[i : i + CHUNK_SIZE - 2]
            + [tokeniser.sep_token_id]
        )
        attention_mask = [1] * len(chunk)

        if len(chunk) < CHUNK_SIZE:
            pad_size = CHUNK_SIZE - len(chunk)
            chunk = chunk + ([0] * pad_size)  # pad until CHUNK_SIZE
            attention_mask = attention_mask + ([0] * pad_size)

        chunk_input_ids.append(chunk)
        chunk_attention_masks.append(attention_mask)

    return {"input_ids": chunk_input_ids, "attention_mask": chunk_attention_masks}


tokenised_dataset = dataset.map(tokenise_dataset)

print(tokenised_dataset)
# so idea here is to split into chunks of word encodings, do model() model() model(), get the CLS repr, then feed into torch_model


# need to handle chunks inside the class
# do we need to calculate positional embedding for the transformer layer?
class Model(nn.Module):
    def __init__(self, num_tf_layers, hidden_dim, num_classes, train_labels):
        super(Model, self).__init__()
        if platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.init_layers(num_tf_layers, hidden_dim, num_classes)
        self.calculate_class_weights(train_labels)
        self.init_loss_optimiser()

    def init_layers(self, num_tf_layers, hidden_dim, num_classes):
        self.bert = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
        self.bert = self.bert.to(self.device)

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.bert.config.hidden_size,  # 768 for bert
                    nhead=8,
                    dim_feedforward=hidden_dim,
                )
                for _ in range(num_tf_layers)
            ]
        )

        self.dropout = nn.Dropout(0.2)

        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def init_loss_optimiser(self):
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights).to(self.device)
        )
        self.optimiser = torch.optim.AdamW(
            self.parameters(), lr=1e-5
        )  # transformers default in huggingface
        # self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

    def calculate_class_weights(self, train_labels):
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def handle_chunks(self, input_ids, attention_mask):
        num_of_chunks = [len(chunk) for chunk in input_ids]

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

        transformer_output = self.dropout(transformer_output)
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

    def train_loop(self, train_dataloader):
        loss = 0
        for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
            (
                input_ids_combined_tensors,
                attention_mask_combined_tensors,
                num_of_chunks,
            ) = self.handle_chunks(batch_input_ids, batch_attention_mask)

            self.optimiser.zero_grad()

            logits = self.forward(
                input_ids_combined_tensors, attention_mask_combined_tensors
            )

            logits_split = logits.split(num_of_chunks)

            pooled_logits = torch.cat(
                [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
            )

            batch_loss = self.loss_function(pooled_logits, batch_labels)

            batch_loss.backward()
            self.optimiser.step()

            loss += batch_loss.detach().item()

        loss = loss / (len(train_dataloader))
        return loss

    def validation_loop(self, valid_dataloader):
        loss = 0
        all_pooled_logits = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for (
                batch_input_ids,
                batch_attention_mask,
                batch_labels,
            ) in valid_dataloader:
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

                batch_loss = self.loss_function(pooled_logits, batch_labels)

                loss += batch_loss.detach().item()

                all_pooled_logits = torch.cat((all_pooled_logits, pooled_logits), dim=0)

        loss = loss / (len(valid_dataloader))

        labels = []
        for _, _, batch_labels in valid_dataloader:
            labels.extend(batch_labels.tolist())

        metrics = {"loss": loss, **self.compute_metrics(all_pooled_logits, labels)}

        return metrics

    def fit(self, train_dataloader, valid_dataloader, epochs=3):
        print("Training and validating model")
        for epoch in range(epochs):
            print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

            train_loss = self.train_loop(train_dataloader)

            validation_metrics = self.validation_loop(valid_dataloader)

            print(f"Training loss: {train_loss}")
            print(f"Validation metrics: {validation_metrics}")
            print()

    def compute_metrics(self, logits, labels):
        preds = logits.cpu().numpy().argmax(-1)

        report = classification_report(labels, preds)
        print(report)

        precision = precision_score(labels, preds, average="weighted", zero_division=1)
        recall = recall_score(labels, preds, average="weighted", zero_division=1)
        f1 = f1_score(labels, preds, average="weighted", zero_division=1)

        return {"precision": precision, "recall": recall, "f1": f1}

    def predict(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = torch.tensor(inputs["labels"]).to(self.device)
        with torch.no_grad():
            (
                input_ids_combined_tensors,
                attention_mask_combined_tensors,
                num_of_chunks,
            ) = self.handle_chunks(input_ids, attention_mask)

            logits = self.forward(
                input_ids_combined_tensors, attention_mask_combined_tensors
            )

            logits_split = logits.split(num_of_chunks)

            pooled_logits = torch.cat(
                [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
            )

            loss = self.loss_function(pooled_logits, labels)
            loss = loss.detach().item()

        labels = labels.tolist()
        metrics = {"loss": loss, **self.compute_metrics(pooled_logits, labels)}

        print(metrics)

        return metrics


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]
model = Model(
    num_tf_layers=NUM_TF_LAYERS,
    hidden_dim=HIDDEN_SIZE,
    num_classes=num_labels,
    train_labels=train_labels,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])

# TODO - apply first chunk pooling? read su
# For the transformer encoder, we add an empty chunk at the beginning of the chunk sequence that is used as the document representation to the classifier, analogous to the [CLS] token commonly used in that way for sentence representations (and used here as the chunk representation)

# title + content, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 10, TRANSFORMER_MODEL_NAME bert-base-uncased, 2 linear layer
# {'loss': 1.7769678831100464, 'precision': 0.7186231414465551, 'recall': 0.719626168224299, 'f1': 0.7175640997635683}
