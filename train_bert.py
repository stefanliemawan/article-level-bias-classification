import pandas as pd
from transformers import AutoTokenizer
from imblearn.over_sampling import SMOTE
import torch
import re
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    TrainingArguments,
    Trainer,
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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

LEMMATISER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

SEED = 42
# CLASS_WEIGHTS = [2.53631554, 0.7314468, 0.80738019]
CLASS_WEIGHTS = [2.57981651, 0.72812015, 0.80711825]  # train only


# y = df["labels"]
# x = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
# print(x)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
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
        labels = dataset_dict["labels"]
        class_distribution = np.bincount(labels)

        for class_idx, count in enumerate(class_distribution):
            print(f"Class {class_idx}: {count} samples")
        print()


def preprocess_content(row):
    content = row["content"]
    # content = re.sub(r"[\.\?\!\,\:\;\"]", "", content)
    tokenised_content = word_tokenize(content)

    lemmatised_content = [LEMMATISER.lemmatize(token) for token in tokenised_content]

    preprocessed_content = [
        word for word in lemmatised_content if word not in STOP_WORDS
    ]
    preprocessed_content = " ".join(preprocessed_content)

    return preprocessed_content


def create_dataset(df, class_ranges=[], regression=False):
    df["features"] = df["title"] + ". " + df["content"]
    # df["features"] = df["content"]

    if regression is True:
        df["labels"] = df["reliability_score"]
    else:

        def map_to_class(score):
            for i, (start, end) in enumerate(class_ranges):
                if start <= score <= end:
                    return i

        df["labels"] = df["reliability_score"].apply(map_to_class)

    dataset = Dataset.from_pandas(df[["features", "labels"]], preserve_index=False)
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED)

    test_valid_dataset = dataset["test"].train_test_split(test_size=0.5, seed=SEED)

    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "test": test_valid_dataset["train"],
            "valid": test_valid_dataset["test"],
        }
    )

    return dataset


def tokenise_dataset(dataset, model_name, oversampling=False):
    tokeniser = AutoTokenizer.from_pretrained(model_name)

    tokenised_dataset = dataset.map(
        lambda x: tokeniser(x["features"], padding="max_length", truncation=True),
        batched=True,
    )

    if oversampling:
        smote = SMOTE(random_state=SEED)

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

    print(tokeniser.decode(tokenised_dataset["train"]["input_ids"][0]))

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
    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenised_dataset["train"],
    #     eval_dataset=tokenised_dataset["valid"],
    #     compute_metrics=compute_metrics,
    # )

    trainer.train()

    test = trainer.evaluate(eval_dataset=tokenised_dataset["test"])
    print(test)


def train_regression(df, model_name="distilbert-base-uncased"):
    dataset = create_dataset(df, regression=True)
    tokenised_dataset = tokenise_dataset(dataset, model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    train(tokenised_dataset, model, compute_metrics=compute_rmse)


def train_classification(df, model_name="distilbert-base-uncased"):
    df["content"] = df.apply(preprocess_content, axis=1)

    class_ranges = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]
    dataset = create_dataset(df, class_ranges)
    tokenised_dataset = tokenise_dataset(dataset, model_name)

    print_class_distribution(tokenised_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_ranges)
    )

    train(tokenised_dataset, model, epoch=3)


def train_classification_with_oversampling(df, model_name="distilbert-base-uncased"):
    df["content"] = df.apply(preprocess_content, axis=1)

    class_ranges = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]
    dataset = create_dataset(df, class_ranges)
    tokenised_dataset = tokenise_dataset(dataset, model_name, oversampling=True)

    print_class_distribution(tokenised_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_ranges)
    )

    train(tokenised_dataset, model, epoch=3)


def train_peft(df, model_name="FacebookAI/roberta-base"):
    df["content"] = df.apply(preprocess_content, axis=1)

    class_ranges = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]
    dataset = create_dataset(df, class_ranges)
    tokenised_dataset = tokenise_dataset(dataset, model_name)

    print_class_distribution(tokenised_dataset)

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
train_classification(df, "distilbert-base-uncased")
# train_classification_with_oversampling(df, "distilbert-base-uncased")


# v2 regression, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 39.483707427978516, 'eval_rmse': 6.283606052398682, 'eval_runtime': 23.8558, 'eval_samples_per_second': 22.091, 'eval_steps_per_second': 2.767, 'epoch': 3.0}

# v2 classification 3 classes, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 0.825508713722229, 'eval_accuracy': 0.6717267552182163, 'eval_precision': 0.6753208471113097, 'eval_recall': 0.6717267552182163, 'eval_f1': 0.667779770031748, 'eval_runtime': 20.8621, 'eval_samples_per_second': 25.261, 'eval_steps_per_second': 3.164, 'epoch': 3.0}

# v2 classification 3 classes, content only, distilbert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 0.8318021893501282, 'eval_accuracy': 0.681214421252372, 'eval_precision': 0.6873372214160806, 'eval_recall': 0.681214421252372, 'eval_f1': 0.6802339141881031, 'eval_runtime': 21.7416, 'eval_samples_per_second': 24.239, 'eval_steps_per_second': 3.036, 'epoch': 3.0}

# v2 classification 3 classes, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone), weighted loss
# {'eval_loss': 0.9086113572120667, 'eval_accuracy': 0.683111954459203, 'eval_precision': 0.6884837573495253, 'eval_recall': 0.683111954459203, 'eval_f1': 0.6854369304488173, 'eval_runtime': 21.0092, 'eval_samples_per_second': 25.084, 'eval_steps_per_second': 3.141, 'epoch': 3.0}

# v2 classification_with_oversampling 3 classes, title + ". " content, distilbert-base-uncased,preprocessed text (lemma and stop words gone)
# {'eval_loss': 0.8091003894805908, 'eval_accuracy': 0.7058823529411765, 'eval_precision': 0.7039783724648657, 'eval_recall': 0.7058823529411765, 'eval_f1': 0.6958079783637592, 'eval_runtime': 21.6961, 'eval_samples_per_second': 24.29, 'eval_steps_per_second': 3.042, 'epoch': 3.0}

# LR RMSE: 424249227.8148557


# The reliability score indicates how much truthfulness the article
# contains. Here, the values range from 0 (least reliable, contains inaccurate/fabricated info) to 64 (most reliable, original fact reporting). (in samples max 58.67)

# TODO - combine outlet and article reliability score
