import os
import platform
import sys

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MODEL_NAME = "google/bigbird-pegasus-large-arxiv"
MODEL_NAME = "google/bigbird-roberta-base"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)

# train_df, test_df, valid_df = functions.generate_title_content_features(
#     train_df, test_df, valid_df
# )
train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenised_dataset = dataset.map(
    lambda x: tokeniser(
        x["features"], padding="max_length", truncation=True, max_length=4096
    ),
    batched=True,
)

print(tokenised_dataset)

functions.print_class_distribution(tokenised_dataset)

num_labels = len(pd.unique(train_df["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, max_length=4096
)
if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")

functions.train(tokenised_dataset, model, epochs=4)

#               precision    recall  f1-score   support

#            0       0.41      0.44      0.43        27
#            1       0.39      0.52      0.45        54
#            2       0.43      0.53      0.48       104
#            3       0.92      0.82      0.86       384

#     accuracy                           0.72       569
#    macro avg       0.54      0.58      0.55       569
# weighted avg       0.75      0.72      0.73       569


# 100%|██████████| 72/72 [00:19<00:00,  3.65it/s]
# {'eval_loss': 0.8981084823608398, 'eval_precision': 0.7538596748874499, 'eval_recall': 0.7170474516695958, 'eval_f1': 0.7318008367517874, 'eval_runtime': 20.0418, 'eval_samples_per_second': 28.391, 'eval_steps_per_second': 3.592, 'epoch': 4.0}
