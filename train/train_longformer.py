import os
import platform
import sys

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "allenai/longformer-base-4096"


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

# title + content, vx
#               precision    recall  f1-score   support

#            0       0.48      0.48      0.48        27
#            1       0.44      0.50      0.47        54
#            2       0.47      0.59      0.52       104
#            3       0.92      0.84      0.88       384

#     accuracy                           0.74       569
#    macro avg       0.58      0.60      0.59       569
# weighted avg       0.77      0.74      0.75       569


# {'eval_loss': 0.7925838828086853, 'eval_precision': 0.7708420391217801, 'eval_recall': 0.7434094903339191, 'eval_f1': 0.7544174735254525, 'eval_runtime': 24.3887, 'eval_samples_per_second': 23.331, 'eval_steps_per_second': 2.952, 'epoch': 4.0}

# outlet + title + content, vx
#               precision    recall  f1-score   support

#            0       0.60      0.44      0.51        27
#            1       0.45      0.56      0.50        54
#            2       0.44      0.53      0.48       104
#            3       0.91      0.85      0.88       384

#     accuracy                           0.74       569
#    macro avg       0.60      0.59      0.59       569
# weighted avg       0.77      0.74      0.75       569


# 100%|██████████| 72/72 [00:23<00:00,  3.00it/s]
# {'eval_loss': 0.8032352924346924, 'eval_precision': 0.7676585977630122, 'eval_recall': 0.7434094903339191, 'eval_f1': 0.7529084481326138, 'eval_runtime': 24.3199, 'eval_samples_per_second': 23.396, 'eval_steps_per_second': 2.961, 'epoch': 4.0}
