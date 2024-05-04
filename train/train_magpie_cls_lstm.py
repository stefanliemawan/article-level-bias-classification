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
HIDDEN_DIM = 256
EPOCHS = 8
DROPOUT_PROB = 0.2
TF_MODEL_NAME = "mediabiasgroup/magpie-babe-ft"

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, NUM_TF_LAYERS {NUM_TF_LAYERS}, HIDDEN_DIM {HIDDEN_DIM}, EPOCHS {EPOCHS}, DROPOUT {DROPOUT_PROB}, TRANSFORMER_MODEL_NAME {TF_MODEL_NAME}"
)
print("dataset v3")

train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(TF_MODEL_NAME)


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
        self.magpie = AutoModel.from_pretrained(TF_MODEL_NAME)
        self.magpie = self.magpie.to(self.device)

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.magpie.config.hidden_size,  # 768 for magpie
                    nhead=8,
                    dim_feedforward=hidden_dim,
                )
                for _ in range(num_tf_layers)
            ]
        )

        self.lstm = nn.LSTM(
            self.magpie.config.hidden_size,
            hidden_dim,
            num_layers=2,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = nn.Dropout(DROPOUT_PROB)

    def init_loss_optimiser(self):
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights).to(self.device)
        )
        self.optimiser = torch.optim.AdamW(
            self.parameters(), lr=1e-5
        )  # transformers default in huggingface
        # self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, input_ids, attention_mask):
        magpie_output = self.magpie(input_ids=input_ids, attention_mask=attention_mask)
        transformer_output = magpie_output.last_hidden_state

        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)

        transformer_output_cls = transformer_output[:, 0, :]  # only using [CLS] token
        # transformer_output_mean = transformer_output.mean(dim=1)  # average pooling

        lstm_output, _ = self.lstm(transformer_output_cls)

        mlp_output = self.mlp(lstm_output)

        return mlp_output

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

            logits = self.forward(
                input_ids_combined_tensors, attention_mask_combined_tensors
            )

            logits_split = logits.split(num_of_chunks)

            pooled_logits = torch.cat(
                [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
            )

            batch_loss = self.loss_function(pooled_logits, batch_labels)

            self.optimiser.zero_grad()

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
    hidden_dim=HIDDEN_DIM,
    num_classes=num_labels,
    train_labels=train_labels,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])

# v2 ---
# CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_DIM 256, EPOCHS 8, DROPOUT 0.2, TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, (lstm): LSTM(768, 256, num_layers=2)
# ------------------------- Epoch 8 -------------------------
# Training loss: 0.1503344306932582
# Validation metrics: {'loss': 1.165870706180492, 'precision': 0.7041912379566014, 'recall': 0.7108433734939759, 'f1': 0.7022498295173187}
# {'loss': 1.0614279508590698, 'precision': 0.7267950648145518, 'recall': 0.7336448598130841, 'f1': 0.7256561887323222}

# CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_DIM 512, EPOCHS 10, DROPOUT 0.2, TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, (lstm): LSTM(768, 256, num_layers=2, bidirectional=True)
# ------------------------- Epoch 10 -------------------------
# Training loss: 0.07712768921450217
# Validation metrics: {'loss': 1.5255870702395957, 'precision': 0.6829998670219071, 'recall': 0.6867469879518072, 'f1': 0.6802136371274436}
# {'loss': 1.4709683656692505, 'precision': 0.7115481329417223, 'recall': 0.7165109034267912, 'f1': 0.7100893757205801}

# CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_DIM 512, EPOCHS 8, DROPOUT 0.2, TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft
# 3 layers lstm, worse
# ------------------------- Epoch 8 -------------------------
# Training loss: 0.11584379503901507
# Validation metrics: {'loss': 1.4456789765551865, 'precision': 0.6987418728300206, 'recall': 0.7048192771084337, 'f1': 0.6969043349033112}

# {'loss': 1.4155519008636475, 'precision': 0.7143933030667106, 'recall': 0.7180685358255452, 'f1': 0.7117710580545409}

# v3
