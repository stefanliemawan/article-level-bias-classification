import platform

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import utils.functions as functions
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from torch import nn
from transformers import BertModel, BertTokenizer

CHUNK_SIZE = 202
HIDDEN_SIZE = 768
NUM_FILTERS = 128
KERNEL_SIZES = [2, 3, 4]
EPOCHS = 2
TRANSFORMER_MODEL_NAME = "bert-base-uncased"

print(f"CHUNK_SIZE {CHUNK_SIZE}, HIDDEN_SIZE {HIDDEN_SIZE}, EPOCHS {EPOCHS}")

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)


dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = BertTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
tokenised_dataset = dataset.map(
    lambda x: tokeniser(
        x["features"],
        padding=False,
        truncation=False,
    ),
    batched=True,
)

max_tokens_length = max(
    len(tokens) for tokens in tokenised_dataset["train"]["input_ids"]
)

print(max_tokens_length)


padded_inputs = [
    tokens + ["[PAD]"] * (max_tokens_length - len(tokens))
    for tokens in tokenised_dataset["train"]["input_ids"]
]

# Convert tokens back to input IDs and create PyTorch tensor
input_ids = [tokeniser.convert_tokens_to_ids(tokens) for tokens in padded_inputs]
input_ids_tensor = torch.tensor(input_ids)

print(input_ids_tensor.shape)
# this works, now need to replace tokenised_dataset["train"]["input_ids"] with this, lol, maybe dont use dataset?
asd


class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_filters,
        kernel_sizes,
        num_classes,
        train_labels,
    ):
        super(Model, self).__init__()
        if platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.init_layers(vocab_size, hidden_dim, num_filters, kernel_sizes, num_classes)
        self.calculate_class_weights(train_labels)
        self.init_loss_optimiser()

    def init_layers(
        self, vocab_size, hidden_dim, num_filters, kernel_sizes, num_classes
    ):
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_dim, out_channels=num_filters, kernel_size=ks
                )
                for ks in kernel_sizes
            ]
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=2
        )  # Assuming kernel_size=2 for max pooling

        self.fc_input_size = (
            num_filters * len(kernel_sizes) // 2
        )  # // 2 for max pooling

        self.fc1 = nn.Linear(self.fc_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.5)

    def init_loss_optimiser(self):
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights).to(self.device)
        )
        self.optimiser = torch.optim.AdamW(
            self.parameters(), lr=3e-5
        )  # transformers default in huggingface
        # self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

    def calculate_class_weights(self, train_labels):
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def forward(self, input_ids, attention_mask):
        print(type(input_ids))
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # Permute dimensions for Conv1d input (expected shape: BxCxL)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)

        # Apply convolutional layers and max pooling
        conv_outputs = [F.relu(conv(embedded)) for conv in self.convs]

        pooled_outputs = [self.max_pool(conv_out) for conv_out in conv_outputs]

        # Concatenate pooled outputs along the channel dimension
        pooled_outputs = torch.cat(
            pooled_outputs, dim=1
        )  # (batch_size, num_filters * len(kernel_sizes) // 2)

        # Apply fully-connected layers (MLP)
        x = self.dropout(pooled_outputs)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Final output logits

        return x

    def batchify(self, inputs, batch_size=8):  # better way to do this?
        input_ids = torch.tensor([f["input_ids"] for f in inputs]).to(self.device)
        attention_mask = torch.tensor([f["attention_mask"] for f in inputs]).to(
            self.device
        )
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
            logits = self.forward(batch_input_ids, batch_attention_mask)

            batch_loss = self.loss_function(logits, batch_labels)

            self.optimiser.zero_grad()

            batch_loss.backward()
            self.optimiser.step()

            loss += batch_loss.detach().item()

        loss = loss / (len(train_dataloader))
        print(f"total_loss {loss}")

        return loss

    def validation_loop(self, valid_dataloader):
        loss = 0
        all_logits = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for (
                batch_input_ids,
                batch_attention_mask,
                batch_labels,
            ) in valid_dataloader:
                logits = self.forward(batch_input_ids, batch_attention_mask)

                batch_loss = self.loss_function(logits, batch_labels)

                loss += batch_loss.detach().item()

                all_logits = torch.cat((all_logits, logits), dim=0)

        loss = loss / (len(valid_dataloader))

        labels = []
        for _, _, batch_labels in valid_dataloader:
            labels.extend(batch_labels.tolist())

        metrics = {"loss": loss, **self.compute_metrics(all_logits, labels)}

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

        precision = precision_score(labels, preds, average="weighted", zero_division=1)
        recall = recall_score(labels, preds, average="weighted", zero_division=1)
        f1 = f1_score(labels, preds, average="weighted", zero_division=1)

        return {"precision": precision, "recall": recall, "f1": f1}

    def predict(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = torch.tensor(inputs["labels"]).to(self.device)

        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)

            loss = self.loss_function(logits, labels)
            loss = loss.detach().item()

        labels = labels.tolist()
        metrics = {"loss": loss, **self.compute_metrics(logits, labels)}

        print(metrics)

        return metrics


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]

model = Model(
    vocab_size=tokeniser.vocab_size,
    hidden_dim=HIDDEN_SIZE,
    num_filters=NUM_FILTERS,
    kernel_sizes=KERNEL_SIZES,
    num_classes=num_labels,
    train_labels=train_labels,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])
