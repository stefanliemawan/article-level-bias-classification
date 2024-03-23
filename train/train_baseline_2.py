import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

# MODEL_NAME = "distilbert-base-uncased"
MODEL_NAME = "bert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenised_dataset = functions.tokenise_dataset(dataset, tokeniser, seed=SEED)

print(tokenised_dataset)

functions.print_class_distribution(tokenised_dataset)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_RANGES)
)
if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")


functions.train(tokenised_dataset, model, epoch=5)

# v2 train-test-valid classification 3 classes, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone), weighted loss
# {'eval_loss': 0.8578395247459412, 'eval_accuracy': 0.6776729559748428, 'eval_precision': 0.6795493125424996, 'eval_recall': 0.6776729559748428, 'eval_f1': 0.6776522593187786, 'eval_runtime': 25.6869, 'eval_samples_per_second': 24.76, 'eval_steps_per_second': 3.114, 'epoch': 3.0}

# v2 train-test-valid regression, title + ". " content, distilbert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 39.180782318115234, 'eval_rmse': 6.25945520401001, 'eval_runtime': 24.9721, 'eval_samples_per_second': 25.468, 'eval_steps_per_second': 3.204, 'epoch': 3.0}

# v2 train-test-valid regression, title + ". " content, bert-base-uncased, preprocessed text (lemma and stop words gone)
# {'eval_loss': 0.7542935609817505, 'eval_accuracy': 0.699685534591195, 'eval_precision': 0.70062888135781, 'eval_recall': 0.699685534591195, 'eval_f1': 0.6993842803109104, 'eval_runtime': 2.333, 'eval_samples_per_second': 272.605, 'eval_steps_per_second': 34.29, 'epoch': 3.0}

# v2 train-test-valid regression, title + ". " content, bert-base-uncased, no preprocessing
# {'eval_loss': 0.8713238835334778, 'eval_accuracy': 0.7075471698113207, 'eval_precision': 0.7069202112045856, 'eval_recall': 0.7075471698113207, 'eval_f1': 0.7055632394331082, 'eval_runtime': 2.3262, 'eval_samples_per_second': 273.408, 'eval_steps_per_second': 34.391, 'epoch': 3.0}
