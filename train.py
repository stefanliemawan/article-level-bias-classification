import pandas as pd
from transformers import AutoTokenizer

from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import root_mean_squared_error
import numpy as np


def train_model(dataset, model_name):
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
        num_train_epochs=4,
        save_total_limit=2,
        save_strategy="no",
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_datasets["train"],
        eval_dataset=tokenised_datasets["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()


df = pd.read_csv("cleaned_dataset/scraped_merged_clean_v2.csv", index_col=0)
# [5274 rows x 9 columns]

df["labels"] = df["reliability_score"]
df["features"] = df["title"] + ". " + df["content"]

dataset = Dataset.from_pandas(df[["features", "labels"]], preserve_index=False)
dataset = dataset.train_test_split(test_size=0.2)


train_model(dataset, "distilbert-base-uncased")

# seems better with just content without title? nah?

# title + ". " text, distilbert-base-uncased
# {'eval_loss': 45.62968063354492, 'eval_rmse': 6.754974365234375, 'eval_runtime': 46.21, 'eval_samples_per_second': 22.831, 'eval_steps_per_second': 2.857, 'epoch': 3.0}
# {'train_runtime': 1854.9658, 'train_samples_per_second': 6.823, 'train_steps_per_second': 0.854, 'train_loss': 131.8087984335543, 'epoch': 3.0}

# title + ". " text, bert-base-uncased
# {'eval_loss': 76.54418182373047, 'eval_rmse': 8.748952865600586, 'eval_runtime': 78.1991, 'eval_samples_per_second': 13.491, 'eval_steps_per_second': 1.688, 'epoch': 3.0}
# {"loss": 75.8268, "learning_rate": 1e-05, "epoch": 4.0}

# LR RMSE: 424249227.8148557
