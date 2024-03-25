import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "bert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(train_df, test_df, valid_df)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokeniser.add_tokens(train_df["outlet"].tolist())
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


functions.train(tokenised_dataset, model, epoch=4)

# outlet + title + content, bert-base-uncased
# {'eval_loss': 0.7819182276725769, 'eval_accuracy': 0.6792452830188679, 'eval_precision': 0.6840998543100941, 'eval_recall': 0.6792452830188679, 'eval_f1': 0.6808352544547148, 'eval_runtime': 2.3317, 'eval_samples_per_second': 272.765, 'eval_steps_per_second': 34.31, 'epoch': 4.0}
