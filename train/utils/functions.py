import re

import numpy as np
from datasets import Dataset, DatasetDict
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import TrainingArguments, default_data_collator

from .standard_trainer import StandardTrainer

# import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")


LEMMATISER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))


def compute_metrics_classification(pred):
    labels = pred.label_ids.flatten().tolist()
    preds = pred.predictions.argmax(-1)

    report = classification_report(labels, preds, zero_division=1)
    print(f"\n{report}")

    precision = precision_score(labels, preds, average="weighted", zero_division=1)
    recall = recall_score(labels, preds, average="weighted", zero_division=1)
    f1 = f1_score(labels, preds, average="weighted", zero_division=1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_class_distribution(dataset):
    for key, dataset_dict in dataset.items():
        print(key)
        labels = dataset_dict["labels"]
        class_distribution = np.bincount(labels)

        for class_idx, count in enumerate(class_distribution):
            print(f"Class {class_idx}: {count} samples")
        print()


def preprocess_content(row):
    content = row["content"]
    content = re.sub(r"[\.\?\!\,\:\;\"]", "", content)
    # content = re.sub(r"[\?\!\,\:\;\"]", "", content)
    tokenised_content = word_tokenize(content)

    lemmatised_content = [LEMMATISER.lemmatize(token) for token in tokenised_content]

    preprocessed_content = [
        word for word in lemmatised_content if word not in STOP_WORDS
    ]
    preprocessed_content = " ".join(preprocessed_content)

    return preprocessed_content


def generate_title_content_features(train_df, test_df, valid_df):
    train_df["features"] = train_df["title"] + ". " + train_df["content"]
    test_df["features"] = test_df["title"] + ". " + test_df["content"]
    valid_df["features"] = valid_df["title"] + ". " + valid_df["content"]

    print("features: title + content")

    return train_df, test_df, valid_df


def generate_outlet_title_content_features(train_df, test_df, valid_df):
    train_df["features"] = (
        train_df["outlet"] + ". " + train_df["title"] + ". " + train_df["content"]
    )
    test_df["features"] = (
        test_df["outlet"] + ". " + test_df["title"] + ". " + test_df["content"]
    )
    valid_df["features"] = (
        valid_df["outlet"] + ". " + valid_df["title"] + ". " + valid_df["content"]
    )
    print("features: outlet + title + content")

    return train_df, test_df, valid_df


def create_dataset(train_df, test_df, valid_df):
    train_dataset = Dataset.from_pandas(
        train_df[["features", "labels"]],
        preserve_index=False,
    )
    test_dataset = Dataset.from_pandas(
        test_df[["features", "labels"]],
        preserve_index=False,
    )
    valid_dataset = Dataset.from_pandas(
        valid_df[["features", "labels"]],
        preserve_index=False,
    )

    dataset = DatasetDict(
        {"train": train_dataset, "test": test_dataset, "valid": valid_dataset}
    )

    return dataset


def tokenise_dataset(
    dataset, tokeniser, oversampling=False, seed=None, truncation=True
):
    tokenised_dataset = dataset.map(
        lambda x: tokeniser(x["features"], padding=True, truncation=truncation),
        batched=True,
    )

    if oversampling:
        smote = SMOTE(random_state=seed)

        x_train = np.asarray(tokenised_dataset["train"]["input_ids"])
        y_train = np.asarray(tokenised_dataset["train"]["labels"])

        features_oversampled, labels_oversampled = smote.fit_resample(x_train, y_train)

        tokenised_dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"input_ids": features_oversampled, "labels": labels_oversampled}
                ),
                "test": tokenised_dataset["test"],
                "valid": tokenised_dataset["valid"],
            }
        )

    # print(tokeniser.decode(tokenised_dataset["train"]["input_ids"][0]))

    return tokenised_dataset


def train(
    tokenised_dataset,
    model,
    epochs=4,
    batch_size=8,
    compute_metrics=compute_metrics_classification,
    trainer_class=StandardTrainer,
    data_collator=default_data_collator,
):

    num_training_examples = len(tokenised_dataset["train"]["input_ids"])
    total_steps = (num_training_examples // batch_size) * epochs

    # # Calculate warmup steps (10% of total steps)
    warmup_steps = int(total_steps * 0.1)
    # warmup_steps = 500
    print(f"warmup_steps: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir="test_trainer",
        logging_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
        # learning_rate=5e-5,
        # learning_rate=1e-5,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        # weight_decay=0.01,
    )

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset["train"],
        eval_dataset=tokenised_dataset["valid"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.calculate_class_weights()

    trainer.train()

    test = trainer.evaluate(eval_dataset=tokenised_dataset["test"])
    print(test)
