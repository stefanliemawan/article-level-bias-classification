import pandas as pd
from transformers import AutoTokenizer
from imblearn.over_sampling import SMOTE

from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np


def compute_rmse(eval_pred):
    predictions, labels = eval_pred
    rmse = root_mean_squared_error(labels, predictions)

    return {"rmse": rmse}


def create_dataset(df, class_ranges=[], regression=False):
    df["features"] = df["title"] + ". " + df["content"]

    if regression is True:
        df["labels"] = df["reliability_score"]
    else:

        def map_to_class(score):
            for i, (start, end) in enumerate(class_ranges):
                if start <= score <= end:
                    return i

        df["labels"] = df["reliability_score"].apply(map_to_class)

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

    return dataset


def compute_metrics_classification(pred):
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


def tokenise_dataset(dataset, model_name, oversampling=False):
    tokeniser = AutoTokenizer.from_pretrained(model_name)

    tokenised_dataset = dataset.map(
        lambda x: tokeniser(
            x["features"], padding="max_length", truncation=True, max_length=512
        ),
        batched=True,
    )
    # tokenised_dataset = dataset.map(
    #     lambda x: tokeniser(
    #         x["features"], padding="max_length", truncation=True
    #     ),
    #     batched=True,
    # )

    if oversampling:
        smote = SMOTE(random_state=42)

        x_train = np.asarray(tokenised_dataset["train"]["input_ids"])
        y_train = np.asarray(tokenised_dataset["train"]["labels"])

        features_oversampled, labels_oversampled = smote.fit_resample(x_train, y_train)

        class_distribution = np.bincount(labels_oversampled)

        for class_idx, count in enumerate(class_distribution):
            print(f"Class {class_idx}: {count} samples")

        tokenised_dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"input_ids": features_oversampled, "labels": labels_oversampled}
                ),
                "test": tokenised_dataset["test"],
                "valid": tokenised_dataset["valid"],
            }
        )

    print(tokenised_dataset)

    return tokenised_dataset


def train(
    tokenised_dataset,
    model,
    compute_metrics=compute_metrics_classification,
    epoch=3,
):

    training_args = TrainingArguments(
        output_dir="test_trainer",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epoch,
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset["train"],
        eval_dataset=tokenised_dataset["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test = trainer.evaluate(eval_dataset=tokenised_dataset["test"])
    print(test)


def train_regression(df, model_name="distilbert-base-uncased"):
    dataset = create_dataset(df, regression=True)
    tokenised_dataset = tokenise_dataset(dataset, model_name, oversampling=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    train(tokenised_dataset, model, compute_metrics=compute_rmse)


def train_classification(df, model_name="distilbert-base-uncased"):
    class_ranges = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]
    dataset = create_dataset(df, class_ranges)
    tokenised_dataset = tokenise_dataset(dataset, model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_ranges)
    )

    train(tokenised_dataset, model, epoch=3)


def train_classification_with_oversampling(df, model_name="distilbert-base-uncased"):
    class_ranges = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]
    dataset = create_dataset(df, class_ranges)
    tokenised_dataset = tokenise_dataset(dataset, model_name, oversampling=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_ranges)
    )

    train(
        tokenised_dataset,
        model,
        epoch=3,
    )


def train_peft(df, model_name="FacebookAI/roberta-base"):
    class_ranges = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]
    dataset = create_dataset(df, class_ranges)
    tokenised_dataset = tokenise_dataset(dataset, model_name)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train(
        tokenised_dataset,
        model,
        epoch=3,
    )


# [5274 rows x 9 columns]

df = pd.read_csv("cleaned_dataset/scraped_merged_clean_v2.csv", index_col=0)

# TODO - pick a dedicated, balanced, test set

# train_regression(df, "distilbert-base-uncased")
# train_classification(df, "distilbert-base-uncased")
# train_classification_with_oversampling(df, "distilbert-base-uncased")

# train_peft(df, "FacebookAI/roberta-base")


# v2 regression, title + ". " text, distilbert-base-uncased
# {'eval_loss': 36.518577575683594, 'eval_rmse': 6.043060302734375, 'eval_runtime': 25.704, 'eval_samples_per_second': 25.638, 'eval_steps_per_second': 3.229, 'epoch': 3.0}
# {'eval_loss': 47.13957977294922, 'eval_rmse': 6.865826606750488, 'eval_runtime': 20.4186, 'eval_samples_per_second': 25.81, 'eval_steps_per_second': 3.232, 'epoch': 3.0}

# v2 classification 3 classes, title + ". " text, distilbert-base-uncased
# {'eval_loss': 0.9777273535728455, 'eval_accuracy': 0.6261859582542695, 'eval_precision': 0.6238839421180123, 'eval_recall': 0.6261859582542695, 'eval_f1': 0.6178878872528192, 'eval_runtime': 22.3146, 'eval_samples_per_second': 23.617, 'eval_steps_per_second': 2.958, 'epoch': 3.0}

# v2 classification_with_oversampling 3 classes, title + ". " text, distilbert-base-uncased
# {'eval_loss': 1.0841343402862549, 'eval_accuracy': 0.6793168880455408, 'eval_precision': 0.6717309618347154, 'eval_recall': 0.6793168880455408, 'eval_f1': 0.6699090288605012, 'eval_runtime': 20.7939, 'eval_samples_per_second': 25.344, 'eval_steps_per_second': 3.174, 'epoch': 3.0}

# v2 classification (peft) 3 classes, title + ". " text, roberta-base
# {'eval_loss': 0.28613439202308655, 'eval_accuracy': 0.4857685009487666, 'eval_precision': 0.7502025355652453, 'eval_recall': 0.4857685009487666, 'eval_f1': 0.3176417273126035, 'eval_runtime': 37.9815, 'eval_samples_per_second': 13.875, 'eval_steps_per_second': 1.738, 'epoch': 3.0}

# LR RMSE: 424249227.8148557


# The reliability score indicates how much truthfulness the article
# contains. Here, the values range from 0 (least reliable, contains inaccurate/fabricated info) to 64 (most reliable, original fact reporting). (in samples max 58.67)

# TODO - combine outlet and article reliability score
