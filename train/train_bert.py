import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "bert-base-cased"

DATASET_VERSION = "v4"

print(f"MODEL: {MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

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


functions.train(tokenised_dataset, model, epoch=6)

# v2, title + content, bert-base-uncased, with oversampling
# {'eval_loss': 0.8049562573432922, 'eval_precision': 0.6989735466648578, 'eval_recall': 0.7040498442367601, 'eval_f1': 0.6967025255248073, 'eval_runtime': 2.3663, 'eval_samples_per_second': 271.307, 'eval_steps_per_second': 34.23, 'epoch': 4.0}

# v2, title + content, bert-base-uncased, no oversampling just weighted loss
# {'eval_loss': 1.3561333417892456, 'eval_precision': 0.7101609811080607, 'eval_recall': 0.7102803738317757, 'eval_f1': 0.709418672887263, 'eval_runtime': 2.3497, 'eval_samples_per_second': 273.223, 'eval_steps_per_second': 34.472, 'epoch': 4.0}

# v3, title + content, bert-base-uncased, no oversampling just weighted loss
# {'eval_loss': 1.2678630352020264, 'eval_precision': 0.7163276764016158, 'eval_recall': 0.7118380062305296, 'eval_f1': 0.7134907269760798, 'eval_runtime': 45.7962, 'eval_samples_per_second': 14.019, 'eval_steps_per_second': 1.769, 'epoch': 4.0}

# v4_ranked, title + content, bert-base-cased, no oversampling just weighted loss, old

#               precision    recall  f1-score   support

#            0       0.49      0.58      0.53        74
#            1       0.69      0.63      0.66       292
#            2       0.76      0.79      0.77       276

#     accuracy                           0.69       642
#    macro avg       0.65      0.67      0.65       642
# weighted avg       0.69      0.69      0.69       642

# {'eval_loss': 0.9702945351600647, 'eval_precision': 0.6942916688679829, 'eval_recall': 0.6915887850467289, 'eval_f1': 0.6918596226482016, 'eval_runtime': 47.7002, 'eval_samples_per_second': 13.459, 'eval_steps_per_second': 1.698, 'epoch': 4.0}


# v4, title + content, bert-base-cased, no oversampling just weighted loss
#               precision    recall  f1-score   support

#            0       0.56      0.66      0.61        74
#            1       0.70      0.66      0.68       292
#            2       0.77      0.78      0.77       276

#     accuracy                           0.71       642
#    macro avg       0.68      0.70      0.69       642
# weighted avg       0.71      0.71      0.71       642

# {'eval_loss': 0.886698305606842, 'eval_precision': 0.7142597176941626, 'eval_recall': 0.7118380062305296, 'eval_f1': 0.71233323335984, 'eval_runtime': 44.0597, 'eval_samples_per_second': 14.571, 'eval_steps_per_second': 1.838, 'epoch': 4.0}

# v4, title + content, bert-base-cased, no oversampling just weighted loss, old
#               precision    recall  f1-score   support

#            0       0.62      0.54      0.58        74
#            1       0.70      0.73      0.71       292
#            2       0.78      0.77      0.78       276

#     accuracy                           0.73       642
#    macro avg       0.70      0.68      0.69       642
# weighted avg       0.73      0.73      0.73       642

# {'eval_loss': 1.7175806760787964, 'eval_precision': 0.7253978396507152, 'eval_recall': 0.7258566978193146, 'eval_f1': 0.7250427756243668, 'eval_runtime': 48.3937, 'eval_samples_per_second': 13.266, 'eval_steps_per_second': 1.674, 'epoch': 6.0}
