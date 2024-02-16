import pandas as pd
from transformers import AutoTokenizer
from imblearn.over_sampling import SMOTE

from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np

DF = pd.read_csv("cleaned_dataset/scraped_merged_clean_v2.csv", index_col=0)


def train_regression(df, model_name="distilbert-base-uncased"):
    df["labels"] = df["reliability_score"]
    df["features"] = df["title"] + ". " + df["content"]

    dataset = Dataset.from_pandas(df[["features", "labels"]], preserve_index=False)
    dataset = dataset.train_test_split(test_size=0.25)
    test_valid_dataset = dataset["test"].train_test_split(test_size=0.5)

    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "test": test_valid_dataset["train"],
            "valid": test_valid_dataset["test"],
        }
    )
    tokeniser = AutoTokenizer.from_pretrained(model_name)

    tokenised_datasets = dataset.map(
        lambda x: tokeniser(x["features"], padding="max_length", truncation=True),
        batched=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        rmse = root_mean_squared_error(labels, predictions)

        return {"rmse": rmse}

    training_args = TrainingArguments(
        output_dir="test_trainer",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_datasets["train"],
        eval_dataset=tokenised_datasets["valid"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test = trainer.evaluate(eval_dataset=tokenised_datasets["test"])
    print(test)


def train_classification(df, model_name="distilbert-base-uncased"):
    class_ranges = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

    def map_to_class(score):
        for i, (start, end) in enumerate(class_ranges):
            if start <= score <= end:
                return i

    df["labels"] = df["reliability_score"].apply(map_to_class)
    df["features"] = df["title"] + ". " + df["content"]

    dataset = Dataset.from_pandas(df[["features", "labels"]], preserve_index=False)
    dataset = dataset.train_test_split(test_size=0.2)
    test_valid_dataset = dataset["test"].train_test_split(test_size=0.5)

    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "test": test_valid_dataset["train"],
            "valid": test_valid_dataset["test"],
        }
    )

    tokeniser = AutoTokenizer.from_pretrained(model_name)

    tokenised_datasets = dataset.map(
        lambda x: tokeniser(x["features"], padding="max_length", truncation=True),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_ranges)
    )

    training_args = TrainingArguments(
        output_dir="test_trainer",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
    )

    def compute_metrics(pred):
        labels = pred.label_ids.flatten().tolist()
        preds = pred.predictions.argmax(-1)

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted", zero_division=1)
        recall = recall_score(labels, preds, average="weighted", zero_division=1)
        f1 = f1_score(labels, preds, average="weighted", zero_division=1)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_datasets["train"],
        eval_dataset=tokenised_datasets["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test = trainer.evaluate(eval_dataset=tokenised_datasets["test"])
    print(test)


def train_classification_with_oversampling(df, model_name="distilbert-base-uncased"):
    class_ranges = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

    def map_to_class(score):
        for i, (start, end) in enumerate(class_ranges):
            if start <= score <= end:
                return i

    df["labels"] = df["reliability_score"].apply(map_to_class)
    df["features"] = df["title"] + ". " + df["content"]

    dataset = Dataset.from_pandas(df[["features", "labels"]], preserve_index=False)

    tokeniser = AutoTokenizer.from_pretrained(model_name)

    tokenised_datasets = dataset.map(
        lambda x: tokeniser(x["features"], padding="max_length", truncation=True),
        batched=True,
    )

    smote = SMOTE(random_state=42)

    x_train = np.asarray(tokenised_datasets["input_ids"])
    y_train = np.asarray(tokenised_datasets["labels"])

    features_oversampled, labels_oversampled = smote.fit_resample(x_train, y_train)

    tokenised_datasets = Dataset.from_dict(
        {"input_ids": features_oversampled, "labels": labels_oversampled}
    )

    class_distribution = np.bincount(labels_oversampled)

    for class_idx, count in enumerate(class_distribution):
        print(f"Class {class_idx}: {count} samples")

    # cant seem to add attention mask?
    # tokenised_datasets = tokenised_datasets.map(
    #     lambda x: tokeniser(
    #         x["input_ids"],
    #         padding="max_length",
    #         truncation=True,
    #         return_attention_mask=True,
    #     ),
    #     batched=True,
    # )

    # We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.

    tokenised_datasets = tokenised_datasets.train_test_split(test_size=0.2)
    test_valid_dataset = tokenised_datasets["test"].train_test_split(test_size=0.5)

    tokenised_datasets = DatasetDict(
        {
            "train": tokenised_datasets["train"],
            "test": test_valid_dataset["train"],
            "valid": test_valid_dataset["test"],
        }
    )

    print(tokenised_datasets)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_ranges)
    )

    training_args = TrainingArguments(
        output_dir="test_trainer",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
    )

    def compute_metrics(pred):
        labels = pred.label_ids.flatten().tolist()
        preds = pred.predictions.argmax(-1)

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted", zero_division=1)
        recall = recall_score(labels, preds, average="weighted", zero_division=1)
        f1 = f1_score(labels, preds, average="weighted", zero_division=1)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_datasets["train"],
        eval_dataset=tokenised_datasets["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test = trainer.evaluate(eval_dataset=tokenised_datasets["test"])
    print(test)


# [5274 rows x 9 columns]


# TODO - pick a dedicated, balanced, test set

# train_regression(DF, "distilbert-base-uncased")
# train_classification(DF, "distilbert-base-uncased")
train_classification_with_oversampling(DF, "distilbert-base-uncased")

# seems better with just content without title? nah?


# v2 regression, title + ". " text, distilbert-base-uncased
# {'eval_loss': 36.518577575683594, 'eval_rmse': 6.043060302734375, 'eval_runtime': 25.704, 'eval_samples_per_second': 25.638, 'eval_steps_per_second': 3.229, 'epoch': 3.0}
# {'train_runtime': 1854.9658, 'train_samples_per_second': 6.823, 'train_steps_per_second': 0.854, 'train_loss': 131.8087984335543, 'epoch': 3.0}

# v2 classification 3 classes, title + ". " text, distilbert-base-uncased
# {'eval_loss': 0.9777273535728455, 'eval_accuracy': 0.6261859582542695, 'eval_precision': 0.6238839421180123, 'eval_recall': 0.6261859582542695, 'eval_f1': 0.6178878872528192, 'eval_runtime': 22.3146, 'eval_samples_per_second': 23.617, 'eval_steps_per_second': 2.958, 'epoch': 3.0}

# v2 classification_with_oversampling 3 classes, title + ". " text, distilbert-base-uncased
# {'eval_loss': 0.6847554445266724, 'eval_accuracy': 0.6934812760055479, 'eval_precision': 0.7259663497871663, 'eval_recall': 0.6934812760055479, 'eval_f1': 0.7003987474620936, 'eval_runtime': 29.527, 'eval_samples_per_second': 24.418, 'eval_steps_per_second': 3.082, 'epoch': 1.0}

# LR RMSE: 424249227.8148557


# The reliability score indicates how much truthfulness the article
# contains. Here, the values range from 0 (least reliable, contains inaccurate/fabricated info) to 64 (most reliable, original fact reporting). (in samples max 58.67)

# TODO - combine outlet and article reliability score
