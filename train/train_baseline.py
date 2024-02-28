import pandas as pd
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import functions

SEED = 42

CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]


def train_regression(df, model_name="distilbert-base-uncased", epoch=3):
    df["content"] = df.apply(functions.preprocess_content, axis=1)

    dataset = functions.create_dataset(df, regression=True, seed=SEED)

    tokeniser = AutoTokenizer.from_pretrained(model_name)
    tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser, seed=SEED)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    functions.train(
        tokenised_dataset, model, compute_metrics=functions.compute_rmse, epoch=epoch
    )


def train_classification(df, model_name="distilbert-base-uncased", epoch=3):
    df["content"] = df.apply(functions.preprocess_content, axis=1)

    dataset = functions.create_dataset(df, CLASS_RANGES, seed=SEED)

    train_labels = dataset["train"]["labels"]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = np.asarray(class_weights).astype(np.float32)

    tokeniser = AutoTokenizer.from_pretrained(model_name)
    tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser, seed=SEED)

    functions.print_class_distribution(tokenised_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CLASS_RANGES)
    )

    functions.train(tokenised_dataset, model, epoch=epoch, class_weights=class_weights)


def train_classification_with_oversampling(
    df, model_name="distilbert-base-uncased", epoch=3
):
    df["content"] = df.apply(functions.preprocess_content, axis=1)

    dataset = functions.create_dataset(df, CLASS_RANGES, seed=SEED)

    tokeniser = AutoTokenizer.from_pretrained(model_name)
    tokenised_dataset = functions.tokenise_dataset(
        dataset, tokeniser, oversampling=True, seed=SEED
    )

    functions.print_class_distribution(tokenised_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CLASS_RANGES)
    )

    functions.train(tokenised_dataset, model, epoch=epoch)


def train_peft(df, model_name="FacebookAI/roberta-base", epoch=3):
    df["content"] = df.apply(functions.preprocess_content, axis=1)

    dataset = functions.create_dataset(df, CLASS_RANGES, seed=SEED)

    tokeniser = AutoTokenizer.from_pretrained(model_name)
    tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser, seed=SEED)

    functions.print_class_distribution(tokenised_dataset)

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

    functions.train(
        tokenised_dataset,
        model,
        epoch=epoch,
    )


# [5274 rows x 9 columns]

df = pd.read_csv("../cleaned_dataset/scraped_merged_clean_v2.csv", index_col=0)

# TODO - pick a dedicated, balanced, test set

# train_regression(df, "distilbert-base-uncased")
train_classification(df, "distilbert-base-uncased", epoch=3)
# train_classification_with_oversampling(df, "distilbert-base-uncased")


# v2 regression, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 41.74140930175781, 'eval_rmse': 6.46075963973999, 'eval_runtime': 20.5366, 'eval_samples_per_second': 25.661, 'eval_steps_per_second': 3.214, 'epoch': 3.0}

# v2 classification 3 classes, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 0.825508713722229, 'eval_accuracy': 0.6717267552182163, 'eval_precision': 0.6753208471113097, 'eval_recall': 0.6717267552182163, 'eval_f1': 0.667779770031748, 'eval_runtime': 20.8621, 'eval_samples_per_second': 25.261, 'eval_steps_per_second': 3.164, 'epoch': 3.0}

# v2 classification 3 classes, content only, distilbert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 0.8318021893501282, 'eval_accuracy': 0.681214421252372, 'eval_precision': 0.6873372214160806, 'eval_recall': 0.681214421252372, 'eval_f1': 0.6802339141881031, 'eval_runtime': 21.7416, 'eval_samples_per_second': 24.239, 'eval_steps_per_second': 3.036, 'epoch': 3.0}

# v2 classification 3 classes, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone), weighted loss
# {'eval_loss': 0.9086113572120667, 'eval_accuracy': 0.683111954459203, 'eval_precision': 0.6884837573495253, 'eval_recall': 0.683111954459203, 'eval_f1': 0.6854369304488173, 'eval_runtime': 21.0092, 'eval_samples_per_second': 25.084, 'eval_steps_per_second': 3.141, 'epoch': 3.0}

# v2 classification_with_oversampling 3 classes, title + ". " content, distilbert-base-uncased,preprocessed text (lemma and stop words gone)
# {'eval_loss': 0.8091003894805908, 'eval_accuracy': 0.7058823529411765, 'eval_precision': 0.7039783724648657, 'eval_recall': 0.7058823529411765, 'eval_f1': 0.6958079783637592, 'eval_runtime': 21.6961, 'eval_samples_per_second': 24.29, 'eval_steps_per_second': 3.042, 'epoch': 3.0}

# v2 classification 3 classes, title + ". " content, bert-base-uncased, preprocessed text (lemma and stop words gone), weighted loss
# {'eval_loss': 0.8252025842666626, 'eval_accuracy': 0.6698292220113852, 'eval_precision': 0.6749151966305013, 'eval_recall': 0.6698292220113852, 'eval_f1': 0.6719404783580116, 'eval_runtime': 39.2251, 'eval_samples_per_second': 13.435, 'eval_steps_per_second': 1.683, 'epoch': 3.0}

# LR RMSE: 424249227.8148557


# The reliability score indicates how much truthfulness the article
# contains. Here, the values range from 0 (least reliable, contains inaccurate/fabricated info) to 64 (most reliable, original fact reporting). (in samples max 58.67)

# TODO - combine outlet and article reliability score
