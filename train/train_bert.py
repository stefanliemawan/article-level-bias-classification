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

# train_df, test_df, valid_df = functions.generate_title_content_features(
#     train_df, test_df, valid_df
# )

train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
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


functions.train(tokenised_dataset, model, epochs=4)

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

# =========================================================================================================================

# vx + rescraped, 4 classes, learning rate 1e-5
#               precision    recall  f1-score   support

#            0       0.43      0.33      0.38        27
#            1       0.38      0.37      0.37        54
#            2       0.38      0.49      0.42       104
#            3       0.87      0.82      0.85       384

#     accuracy                           0.69       569
#    macro avg       0.51      0.50      0.50       569
# weighted avg       0.71      0.69      0.70       569

# {'eval_loss': 0.9382869005203247, 'eval_precision': 0.7149649953016143, 'eval_recall': 0.6924428822495606, 'eval_f1': 0.7013658906789624, 'eval_runtime': 61.2367, 'eval_samples_per_second': 9.292, 'eval_steps_per_second': 1.176, 'epoch': 4.0}

# vx + rescraped, 4 classes, learning rate 2e-5
#               precision    recall  f1-score   support

#            0       0.44      0.41      0.42        27
#            1       0.35      0.43      0.38        54
#            2       0.37      0.50      0.43       104
#            3       0.91      0.80      0.85       384

#     accuracy                           0.69       569
#    macro avg       0.52      0.53      0.52       569
# weighted avg       0.73      0.69      0.71       569

# {'eval_loss': 0.9411536455154419, 'eval_precision': 0.7334918612086225, 'eval_recall': 0.6906854130052724, 'eval_f1': 0.7078053581833883, 'eval_runtime': 43.9151, 'eval_samples_per_second': 12.957, 'eval_steps_per_second': 1.64, 'epoch': 3.0}


# vx + rescraped, 4 classes, learning rate 2e-5, 500 warmup steps, 0.01 weight decay
#               precision    recall  f1-score   support

#            0       0.47      0.30      0.36        27
#            1       0.34      0.46      0.39        54
#            2       0.41      0.56      0.47       104
#            3       0.91      0.80      0.85       384

#     accuracy                           0.70       569
#    macro avg       0.53      0.53      0.52       569
# weighted avg       0.74      0.70      0.71       569

# {'eval_loss': 1.0056016445159912, 'eval_precision': 0.7449721489160396, 'eval_recall': 0.6977152899824253, 'eval_f1': 0.7146010560984263, 'eval_runtime': 45.1213, 'eval_samples_per_second': 12.61, 'eval_steps_per_second': 1.596, 'epoch': 4.0}


# vx + rescraped, 4 classes, learning rate 2e-5, 500 warmup steps, no weight decay
#               precision    recall  f1-score   support

#            0       0.43      0.44      0.44        27
#            1       0.29      0.39      0.33        54
#            2       0.44      0.48      0.46       104
#            3       0.90      0.83      0.86       384

#     accuracy                           0.71       569
#    macro avg       0.51      0.54      0.52       569
# weighted avg       0.74      0.71      0.72       569

# {'eval_loss': 0.956117570400238, 'eval_precision': 0.7359465879430142, 'eval_recall': 0.7065026362038664, 'eval_f1': 0.7193567444490316, 'eval_runtime': 47.4066, 'eval_samples_per_second': 12.003, 'eval_steps_per_second': 1.519, 'epoch': 4.0}

# vx + rescraped, 4 classes, with outlet, learning rate 2e-5, 500 warmup steps, no weight decay
#               precision    recall  f1-score   support

#            0       0.65      0.56      0.60        27
#            1       0.45      0.50      0.47        54
#            2       0.44      0.62      0.52       104
#            3       0.94      0.84      0.88       384

#     accuracy                           0.75       569
#    macro avg       0.62      0.63      0.62       569
# weighted avg       0.79      0.75      0.76       569

# {'eval_loss': 0.8186598420143127, 'eval_precision': 0.7883162926291302, 'eval_recall': 0.7504393673110721, 'eval_f1': 0.7645458957067559, 'eval_runtime': 51.7116, 'eval_samples_per_second': 11.003, 'eval_steps_per_second': 1.392, 'epoch': 4.0}

# vx + rescraped, learning rate 1e-5, with outlet
#               precision    recall  f1-score   support

#            0       0.41      0.41      0.41        27
#            1       0.36      0.46      0.41        54
#            2       0.43      0.51      0.47       104
#            3       0.91      0.83      0.87       384

#     accuracy                           0.72       569
#    macro avg       0.53      0.55      0.54       569
# weighted avg       0.75      0.72      0.73       569

# {'eval_loss': 0.8919610977172852, 'eval_precision': 0.7456408683529249, 'eval_recall': 0.7152899824253075, 'eval_f1': 0.7280234470927682, 'eval_runtime': 47.3076, 'eval_samples_per_second': 12.028, 'eval_steps_per_second': 1.522, 'epoch': 4.0}
