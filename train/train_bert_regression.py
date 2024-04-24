import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "bert-base-uncased"
# MODEL_NAME = "bert-large-cased"

print(f"MODEL: {MODEL_NAME}")
print("dataset v3")

train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)

train_df["labels"] = train_df["reliability_score"]
test_df["labels"] = test_df["reliability_score"]
valid_df["labels"] = valid_df["reliability_score"]

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser)

print(tokenised_dataset)

num_labels = len(pd.unique(train_df["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")


CLASS_RANGES = [(-200, 29.32), (29.33, 43.98), (43.98, 200)]


def map_to_class(score):
    for i, (start, end) in enumerate(CLASS_RANGES):
        if start <= score <= end:
            return i

    return 1


def compute_metrics(pred):
    labels = pred.label_ids.flatten().tolist()
    preds = pred.predictions.argmax(-1)

    labels = [map_to_class(l) for l in labels]
    preds = [map_to_class(p) for p in preds]

    precision = precision_score(labels, preds, average="weighted", zero_division=1)
    recall = recall_score(labels, preds, average="weighted", zero_division=1)
    f1 = f1_score(labels, preds, average="weighted", zero_division=1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


training_args = TrainingArguments(
    output_dir="test_trainer",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
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
    data_collator=default_data_collator,
)

trainer.train()

test = trainer.evaluate(eval_dataset=tokenised_dataset["test"])
print(test)

# {'eval_loss': 76.12373352050781, 'eval_precision': 0.8963292930759182, 'eval_recall': 0.11746987951807229, 'eval_f1': 0.02469717143506641, 'eval_runtime': 47.8064, 'eval_samples_per_second': 13.889, 'eval_steps_per_second': 1.736, 'epoch': 2.0}
# stuck at this
