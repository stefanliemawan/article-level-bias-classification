import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "SpanBERT/spanbert-base-cased"
MAX_LENGTH = 512

print(f"MODEL: {MODEL_NAME}, MAX_LENGTH: {MAX_LENGTH}")
print("dataset v3")

train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser)

tokenised_dataset = dataset.map(
    lambda x: tokeniser(
        x["features"], padding=True, truncation=True, max_length=MAX_LENGTH
    ),
    batched=True,
)

print(tokenised_dataset)

functions.print_class_distribution(tokenised_dataset)

num_labels = len(pd.unique(train_df["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)

if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")

functions.train(tokenised_dataset, model, epochs=4)

# v3, title + content, MODEL: SpanBERT/spanbert-base-cased, MAX_LENGTH: 512
# {'eval_loss': 0.8917548060417175, 'eval_precision': 0.6020342677906867, 'eval_recall': 0.5778816199376947, 'eval_f1': 0.5736767677493624, 'eval_runtime': 2.3225, 'eval_samples_per_second': 276.422, 'eval_steps_per_second': 34.876, 'epoch': 4.0}
