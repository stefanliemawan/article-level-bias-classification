import platform
import sys

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
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

CHUNK_SIZE = 512
NUM_TF_LAYERS = 2
HIDDEN_SIZE = 768
EPOCHS = 3
DROPOUT_PROB = 0.2
TRANSFORMER_MODEL_NAME = "mediabiasgroup/magpie-babe-ft"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {TRANSFORMER_MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, NUM_TF_LAYERS {NUM_TF_LAYERS}, HIDDEN_SIZE {HIDDEN_SIZE}, EPOCHS {EPOCHS}, DROPOUT {DROPOUT_PROB}"
)


train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


# train_df, test_df, valid_df = functions.generate_title_content_features(
#     train_df, test_df, valid_df
# )
train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
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
        # CLS is <s>, SEP is </s>

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
        self.magpie = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
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

        self.dropout = nn.Dropout(DROPOUT_PROB)

        self.mlp = nn.Sequential(
            nn.Linear(self.magpie.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(hidden_dim, num_classes),
        )

    def init_loss_optimiser(self):
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights).to(self.device)
        )
        self.optimiser = torch.optim.AdamW(self.parameters(), lr=1e-5)

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
        magpie_output = self.magpie(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        transformer_output = magpie_output.last_hidden_state

        # len(magpie_output.hidden_states) --> 13
        # transformer_output = magpie_output.hidden_states[-6]

        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)

        transformer_output = self.dropout(transformer_output)

        mlp_output = self.mlp(
            transformer_output[:, 0, :]
        )  # Assuming you're only using [CLS] token

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
            if self.scheduler:
                self.scheduler.step()

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

        num_training_steps = len(train_dataloader) * epochs

        num_warmup_steps = int(0.1 * num_training_steps)
        print(f"warmup_steps: {num_warmup_steps}")

        # Create the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimiser,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        for epoch in range(epochs):
            print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

            train_loss = self.train_loop(train_dataloader)

            validation_metrics = self.validation_loop(valid_dataloader)

            print(f"Training loss: {train_loss}")
            print(f"Validation metrics: {validation_metrics}")
            print()

    def compute_metrics(self, logits, labels):
        preds = logits.cpu().numpy().argmax(-1)

        report = classification_report(labels, preds, zero_division=1)
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

# vx, no warmup steps, new rescraped, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, title + content
#               precision    recall  f1-score   support

#            0       0.56      0.19      0.28        27
#            1       0.42      0.56      0.48        54
#            2       0.42      0.49      0.45       104
#            3       0.90      0.86      0.88       384

#     accuracy                           0.73       569
#    macro avg       0.57      0.52      0.52       569
# weighted avg       0.75      0.73      0.74       569

# {'loss': 1.036009430885315, 'precision': 0.7515373221663202, 'recall': 0.7328646748681898, 'f1': 0.736117275094499}

# vx, warmup_steps: 162, new rescraped, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, title + content
# ------------------------- Epoch 3 -------------------------
#               precision    recall  f1-score   support

#            0       0.46      0.53      0.49        34
#            1       0.44      0.56      0.49        70
#            2       0.40      0.51      0.45       128
#            3       0.90      0.75      0.82       371

#     accuracy                           0.67       603
#    macro avg       0.55      0.59      0.56       603
# weighted avg       0.71      0.67      0.68       603

# Training loss: 0.732341009620818
# Validation metrics: {'loss': 0.8987960764452031, 'precision': 0.7149543365128579, 'recall': 0.6666666666666666, 'f1': 0.6844551880333128}

#               precision    recall  f1-score   support

#            0       0.46      0.63      0.53        27
#            1       0.36      0.41      0.38        54
#            2       0.41      0.55      0.47       104
#            3       0.93      0.80      0.86       384

#     accuracy                           0.71       569
#    macro avg       0.54      0.60      0.56       569
# weighted avg       0.76      0.71      0.73       569

# {'loss': 0.8415158987045288, 'precision': 0.7577533590596234, 'recall': 0.7117750439367311, 'f1': 0.7293065634451941}
# worse with 500 warmup

# vx, warmup_steps: 162, new rescraped, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, outlet + title + content
# ------------------------- Epoch 3 -------------------------
#               precision    recall  f1-score   support

#            0       0.47      0.59      0.52        34
#            1       0.44      0.53      0.48        70
#            2       0.38      0.45      0.41       128
#            3       0.89      0.78      0.83       371

#     accuracy                           0.67       603
#    macro avg       0.54      0.59      0.56       603
# weighted avg       0.71      0.67      0.68       603

# Training loss: 0.7136299732432128
# Validation metrics: {'loss': 0.9060064405202866, 'precision': 0.7071512961708725, 'recall': 0.6699834162520729, 'f1': 0.6846951204301145}

#               precision    recall  f1-score   support

#            0       0.42      0.52      0.47        27
#            1       0.35      0.43      0.38        54
#            2       0.43      0.55      0.48       104
#            3       0.93      0.82      0.87       384

#     accuracy                           0.72       569
#    macro avg       0.53      0.58      0.55       569
# weighted avg       0.76      0.72      0.73       569

# {'loss': 0.8297753930091858, 'precision': 0.760345238507249, 'recall': 0.7170474516695958, 'f1': 0.7342602984152476}
