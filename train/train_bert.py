import os
import platform
import sys

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "bert-base-cased"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
#     train_df, test_df, valid_df
# )


# train_df, test_df, valid_df = (
#     functions.generate_outlet_title_content_polarity_subjectivity_features(
#         train_df, test_df, valid_df
#     )
# )

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser)

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


functions.train(tokenised_dataset, model, epoch=3)

# new split, vx, 4 classes, not done, this is valid on epoch 4, laptop heating. test later, but i think 4 classes is better.
# {'eval_loss': 1.358709454536438, 'eval_precision': 0.732512403908757, 'eval_recall': 0.7028753993610224, 'eval_f1': 0.7107864589046988, 'eval_runtime': 49.3262, 'eval_samples_per_second': 12.691, 'eval_steps_per_second': 1.602, 'epoch': 4.0}
#
#               precision    recall  f1-score   support

#            0       0.60      0.25      0.35        24
#            1       0.47      0.53      0.50        51
#            2       0.37      0.56      0.45        99
#            3       0.89      0.79      0.84       370

#     accuracy                           0.70       544
#    macro avg       0.58      0.53      0.53       544
# weighted avg       0.74      0.70      0.71       544
# {'eval_loss': 1.9810607433319092, 'eval_precision': 0.742856365034052, 'eval_recall': 0.7003676470588235, 'eval_f1': 0.7132017243769818, 'eval_runtime': 37.6228, 'eval_samples_per_second': 14.459, 'eval_steps_per_second': 1.807, 'epoch': 6.0}
# 3 epoch
#               precision    recall  f1-score   support
#            0       0.35      0.25      0.29        24
#            1       0.38      0.35      0.37        51
#            2       0.41      0.63      0.50        99
#            3       0.91      0.81      0.86       370

#     accuracy                           0.71       544
#    macro avg       0.51      0.51      0.50       544
# weighted avg       0.75      0.71      0.72       544

# {'eval_loss': 0.9740434288978577, 'eval_precision': 0.7463925146847605, 'eval_recall': 0.7095588235294118, 'eval_f1': 0.7214331813328282, 'eval_runtime': 40.637, 'eval_samples_per_second': 13.387, 'eval_steps_per_second': 1.673, 'epoch': 3.0}


# with outlet information
#               precision    recall  f1-score   support

#            0       0.67      0.42      0.51        24
#            1       0.46      0.61      0.52        51
#            2       0.46      0.62      0.52        99
#            3       0.95      0.84      0.89       370

#     accuracy                           0.76       544
#    macro avg       0.63      0.62      0.61       544
# weighted avg       0.80      0.76      0.77       544

# {'eval_loss': 0.8534926772117615, 'eval_precision': 0.7997825187003461, 'eval_recall': 0.7573529411764706, 'eval_f1': 0.7717662709748838, 'eval_runtime': 37.8122, 'eval_samples_per_second': 14.387, 'eval_steps_per_second': 1.798, 'epoch': 4.0}
