import os
import platform
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import utils.functions as functions
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel, BertTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "bert-base-cased"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

tokeniser = BertTokenizer.from_pretrained(MODEL_NAME)


def preprocess_data(df, tokeniser, max_length=512):
    input_ids = []
    attention_masks = []
    additional_features = []
    labels = []

    for _, row in df.iterrows():
        encoded = tokeniser.encode_plus(
            row["features"],
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        additional_features.append([row["polarity"], row["subjectivity"]])
        labels.append(row["labels"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    additional_features = torch.tensor(additional_features)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, additional_features, labels


train_inputs, train_masks, train_features, train_labels = preprocess_data(
    train_df, tokeniser
)
test_inputs, test_masks, test_features, test_labels = preprocess_data(
    test_df, tokeniser
)
valid_inputs, valid_masks, valid_features, valid_labels = preprocess_data(
    valid_df, tokeniser
)


class BertWithAdditionalFeatures(BertPreTrainedModel):
    def __init__(self, config, num_class):
        super().__init__(config, num_class)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)

        self.additional_features_layer = nn.Linear(2, 512)
        self.classifier = nn.Linear(config.hidden_size * 512, num_class)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(256, num_class),
        )

    def forward(self, input_ids, attention_mask, additional_features):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]

        additional_features_output = self.additional_features_layer(additional_features)
        combined_output = torch.cat((pooled_output, additional_features_output), dim=1)

        combined_output = self.dropout(combined_output)
        logits = self.mlp(combined_output)

        return logits


# Create the datasets and dataloaders
train_dataset = TensorDataset(train_inputs, train_masks, train_features, train_labels)
val_dataset = TensorDataset(valid_inputs, valid_masks, valid_features, valid_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_features, test_labels)

train_dataloader = DataLoader(
    train_dataset, sampler=RandomSampler(train_dataset), batch_size=8
)
val_dataloader = DataLoader(
    val_dataset, sampler=SequentialSampler(val_dataset), batch_size=8
)
test_dataloader = DataLoader(
    test_dataset, sampler=SequentialSampler(test_dataset), batch_size=8
)


def calculate_class_weights(train_labels):
    class_weights = np.asarray(
        compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
    ).astype(np.float32)

    print(f"class_weights: {class_weights}")
    return class_weights


num_class = len(pd.unique(train_df["labels"]))

model = BertWithAdditionalFeatures.from_pretrained(MODEL_NAME, num_class)

optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(calculate_class_weights(train_df["labels"].tolist()))
)


def compute_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=1)
    print(report)

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Training loop with validation
def evaluate(dataloader):
    model.eval()
    total_loss = 0
    total_preds, total_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_additional_features, b_labels = batch

            outputs = model(b_input_ids, b_attention_mask, b_additional_features)

            loss = criterion(outputs, b_labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(b_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(total_labels, total_preds)

    return avg_loss, metrics


model.train()
for epoch in tqdm(range(4)):
    for batch in train_dataloader:
        b_input_ids, b_attention_mask, b_additional_features, b_labels = batch

        model.zero_grad()

        outputs = model(b_input_ids, b_attention_mask, b_additional_features)

        loss = criterion(outputs, b_labels)
        loss.backward()

        optimiser.step()

    val_loss, valid_metrics = evaluate(val_dataloader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, {valid_metrics}")

# Evaluate on the test set
test_loss, test_metrics = evaluate(test_dataloader)
print(f"Test Loss: {test_loss}, {test_metrics}")


#            0       0.39      0.32      0.35        34
#            1       0.28      0.57      0.38        70
#            2       0.35      0.30      0.33       128
#            3       0.88      0.77      0.82       371

#     accuracy                           0.62       603
#    macro avg       0.48      0.49      0.47       603
# weighted avg       0.67      0.62      0.64       603

# Epoch 4, Validation Loss: 1.2520939437182326, {'precision': 0.6725319600725181, 'recall': 0.6202321724709784, 'f1': 0.6374560262918403}
#               precision    recall  f1-score   support

#            0       0.41      0.33      0.37        27
#            1       0.30      0.59      0.40        54
#            2       0.35      0.33      0.34       104
#            3       0.90      0.80      0.85       384

#     accuracy                           0.67       569
#    macro avg       0.49      0.51      0.49       569
# weighted avg       0.72      0.67      0.69       569

# Test Loss: 1.1103522533343897, {'precision': 0.7182690938653781, 'recall': 0.6731107205623902, 'f1': 0.6888958074436385}
