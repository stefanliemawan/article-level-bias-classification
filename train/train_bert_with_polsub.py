import os
import platform
import sys

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


train_polarity_tensor = torch.tensor(train_df["polarity"].values).unsqueeze(1)
train_subjectivity_tensor = torch.tensor(train_df["subjectivity"].values).unsqueeze(1)

print(train_polarity_tensor)
print(train_subjectivity_tensor)
asd

# dataset = functions.create_dataset(train_df, test_df, valid_df)
# tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser)

# print(tokenised_dataset)

# functions.print_class_distribution(tokenised_dataset)

# num_labels = len(pd.unique(train_df["labels"]))
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME, num_labels=num_labels
# )

# if platform.system() == "Darwin":
#     model = model.to("mps")
# elif torch.cuda.is_available():
#     model = model.to("cuda")
# else:
#     model = model.to("cpu")


# functions.train(tokenised_dataset, model, epoch=4)

# =================================================================
# from GPT below, try later

# import torch.nn as nn
# from transformers import BertModel


# class BertWithMetadata(nn.Module):
#     def __init__(
#         self, bert_model_name="bert-base-uncased", metadata_size=1, num_labels=2
#     ):
#         super(BertWithMetadata, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.metadata_dense = nn.Linear(
#             metadata_size, 768
#         )  # Project metadata to the same size as BERT embeddings
#         self.classifier = nn.Linear(
#             768 * 2, num_labels
#         )  # Combine BERT embeddings and metadata embeddings

#     def forward(self, input_ids, attention_mask, metadata):
#         # Get BERT embeddings
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         cls_output = outputs.last_hidden_state[
#             :, 0, :
#         ]  # Take [CLS] token representation

#         # Process metadata
#         metadata_output = self.metadata_dense(metadata)

#         # Combine BERT and metadata embeddings
#         combined_output = torch.cat((cls_output, metadata_output), dim=1)

#         # Classification layer
#         logits = self.classifier(combined_output)

#         return logits


# from torch.utils.data import Dataset
# from transformers import Trainer, TrainingArguments


# class CustomDataset(Dataset):
#     def __init__(self, encodings, metadata, labels):
#         self.encodings = encodings
#         self.metadata = metadata
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item["metadata"] = self.metadata[idx]
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)


# # Prepare the dataset
# dataset = CustomDataset(encoded_inputs, metadata_tensor, df["label"].values)

# # Define the training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
# )

# # Initialize the custom model
# model = BertWithMetadata()

# # Define the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     eval_dataset=dataset,  # Typically you'd have a separate evaluation dataset
# )

# # Train the model
# trainer.train()
