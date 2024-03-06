import re

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
from transformers import Trainer, TrainingArguments, default_data_collator

LEMMATISER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))


CLASS_WEIGHTS = []
# CLASS_WEIGHTS = [2.57981651, 0.72812015, 0.80711825]  # train only
# CLASS_WEIGHTS = [2.53631554, 0.7314468, 0.80738019] # all samples
# [2.5656934  0.72812015 0.80851066] actual computed train with 42 seed


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label")
        outputs = model(**inputs)

        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS)).to(
            "mps"
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_rmse(eval_pred):
    predictions, labels = eval_pred
    rmse = root_mean_squared_error(labels, predictions)

    return {"rmse": rmse}


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


def print_class_distribution(dataset):
    for key, dataset_dict in dataset.items():
        print(key)
        labels = dataset_dict["label"]
        class_distribution = np.bincount(labels)

        for class_idx, count in enumerate(class_distribution):
            print(f"Class {class_idx}: {count} samples")
        print()


def preprocess_content(row):
    content = row["content"]
    # content = re.sub(r"[\.\?\!\,\:\;\"]", "", content)
    content = re.sub(r"[\?\!\,\:\;\"]", "", content)
    tokenised_content = word_tokenize(content)

    lemmatised_content = [LEMMATISER.lemmatize(token) for token in tokenised_content]

    preprocessed_content = [
        word for word in lemmatised_content if word not in STOP_WORDS
    ]
    preprocessed_content = " ".join(preprocessed_content)

    return preprocessed_content


def create_dataset(df, class_ranges=[], regression=False, seed=None):
    df["features"] = df["title"] + ". " + df["content"]
    # df["features"] = df["content"]

    if regression is True:
        df["label"] = df["reliability_score"]
    else:

        def map_to_class(score):
            for i, (start, end) in enumerate(class_ranges):
                if start <= score <= end:
                    return i

        df["label"] = df["reliability_score"].apply(map_to_class)

    dataset = Dataset.from_pandas(df[["features", "label"]], preserve_index=False)
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)

    test_valid_dataset = dataset["test"].train_test_split(test_size=0.5, seed=seed)

    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "test": test_valid_dataset["train"],
            "valid": test_valid_dataset["test"],
        }
    )

    return dataset


def tokenise_dataset(
    dataset, tokeniser, oversampling=False, seed=None, truncation=True
):
    tokenised_dataset = dataset.map(
        lambda x: tokeniser(x["features"], padding="max_length", truncation=truncation),
        batched=True,
    )

    if oversampling:
        smote = SMOTE(random_state=seed)

        x_train = np.asarray(tokenised_dataset["train"]["input_ids"])
        y_train = np.asarray(tokenised_dataset["train"]["label"])

        features_oversampled, labels_oversampled = smote.fit_resample(x_train, y_train)

        tokenised_dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"input_ids": features_oversampled, "label": labels_oversampled}
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
    compute_metrics=compute_metrics_classification,
    epoch=3,
    batch_size=8,
    class_weights=[],
):

    training_args = TrainingArguments(
        output_dir="test_trainer",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
    )

    if len(class_weights) == 0:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenised_dataset["train"],
            eval_dataset=tokenised_dataset["valid"],
            compute_metrics=compute_metrics,
        )
    else:

        global CLASS_WEIGHTS
        CLASS_WEIGHTS = class_weights
        print(f"class_weights: {CLASS_WEIGHTS}")

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenised_dataset["train"],
            eval_dataset=tokenised_dataset["valid"],
            compute_metrics=compute_metrics,
            data_collator=default_data_collator,
        )

    trainer.train()

    test = trainer.evaluate(eval_dataset=tokenised_dataset["test"])
    print(test)
