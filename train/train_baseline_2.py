import functions
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "google/bigbird-pegasus-large-pubmed"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df["features"] = train_df.apply(functions.preprocess_content, axis=1)
test_df["features"] = test_df.apply(functions.preprocess_content, axis=1)
valid_df["features"] = valid_df.apply(functions.preprocess_content, axis=1)


train_dataset = Dataset.from_pandas(
    train_df[["features", "label"]], preserve_index=False
)
test_dataset = Dataset.from_pandas(test_df[["features", "label"]], preserve_index=False)
valid_dataset = Dataset.from_pandas(
    valid_df[["features", "label"]], preserve_index=False
)

dataset = DatasetDict(
    {"train": train_dataset, "test": test_dataset, "valid": valid_dataset}
)

train_labels = train_dataset["label"]
class_weights = np.asarray(
    compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
).astype(np.float32)

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser, seed=SEED)

functions.print_class_distribution(tokenised_dataset)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_RANGES)
)

functions.train(tokenised_dataset, model, epoch=3, class_weights=class_weights)


# sample_weights = np.asarray(compute_sample_weight("balanced", train_labels)).astype(
#     np.float32
# )
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
# functions.train(
#     tokenised_dataset,
#     model,
#     epoch=3,
#     compute_metrics=functions.compute_rmse,
#     class_weights=sample_weights,
# )


# v2 train-test-valid classification 3 classes, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone), weighted loss
# {'eval_loss': 0.8578395247459412, 'eval_accuracy': 0.6776729559748428, 'eval_precision': 0.6795493125424996, 'eval_recall': 0.6776729559748428, 'eval_f1': 0.6776522593187786, 'eval_runtime': 25.6869, 'eval_samples_per_second': 24.76, 'eval_steps_per_second': 3.114, 'epoch': 3.0}

# v2 train-test-valid regression, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 39.180782318115234, 'eval_rmse': 6.25945520401001, 'eval_runtime': 24.9721, 'eval_samples_per_second': 25.468, 'eval_steps_per_second': 3.204, 'epoch': 3.0}


# TODO - apply weight loss for regression?
