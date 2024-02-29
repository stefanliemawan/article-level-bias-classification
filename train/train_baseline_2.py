import pandas as pd
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import functions

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "distilbert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df["content"] = train_df.apply(functions.preprocess_content, axis=1)
test_df["content"] = test_df.apply(functions.preprocess_content, axis=1)
valid_df["content"] = valid_df.apply(functions.preprocess_content, axis=1)


train_dataset = Dataset.from_pandas(
    train_df[["features", "labels"]], preserve_index=False
)
test_dataset = Dataset.from_pandas(
    test_df[["features", "labels"]], preserve_index=False
)
valid_dataset = Dataset.from_pandas(
    valid_df[["features", "labels"]], preserve_index=False
)

dataset = DatasetDict(
    {"train": train_dataset, "test": test_dataset, "valid": valid_dataset}
)

train_labels = train_dataset["labels"]
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)
class_weights = np.asarray(class_weights).astype(np.float32)


tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser, seed=SEED)

functions.print_class_distribution(tokenised_dataset)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_RANGES)
)

functions.train(tokenised_dataset, model, epoch=3)


# v2 classification 3 classes, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone), weighted loss
# {'eval_loss': 0.7332867383956909, 'eval_accuracy': 0.7185534591194969, 'eval_precision': 0.7185534591194969, 'eval_recall': 0.7185534591194969, 'eval_f1': 0.7185534591194969, 'eval_runtime': 27.3762, 'eval_samples_per_second': 23.232, 'eval_steps_per_second': 2.922, 'epoch': 3.0}
