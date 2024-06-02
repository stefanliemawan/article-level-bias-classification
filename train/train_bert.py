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

# ====================================================================================================================================================================================================

# seems to be generalising better now.

# new v4, title + content, bert-base-cased, weighted loss
# {'loss': 1.048, 'grad_norm': 16.598970413208008, 'learning_rate': 4.166666666666667e-05, 'epoch': 1.0}
#  17%|██████████████████████████████████████▎                                                                                                                                                                                               | 496/2976 [17:09<1:19:21,  1.92s/it]
#               precision    recall  f1-score   support███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [00:48<00:00,  1.69it/s]

#            0       0.26      0.69      0.38        78
#            1       0.53      0.12      0.20       310
#            2       0.61      0.84      0.70       276

#     accuracy                           0.49       664
#    macro avg       0.46      0.55      0.43       664
# weighted avg       0.53      0.49      0.43       664

# {'eval_loss': 0.9119434356689453, 'eval_precision': 0.5285385538389101, 'eval_recall': 0.4879518072289157, 'eval_f1': 0.42975619472187593, 'eval_runtime': 49.1925, 'eval_samples_per_second': 13.498, 'eval_steps_per_second': 1.687, 'epoch': 1.0}
# {'loss': 0.9574, 'grad_norm': 20.267196655273438, 'learning_rate': 3.3333333333333335e-05, 'epoch': 2.0}
#  33%|████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                         | 992/2976 [35:16<1:06:51,  2.02s/it]
#               precision    recall  f1-score   support███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [00:50<00:00,  1.58it/s]

#            0       0.28      0.64      0.39        78
#            1       0.56      0.13      0.21       310
#            2       0.59      0.87      0.70       276

#     accuracy                           0.50       664
#    macro avg       0.47      0.55      0.43       664
# weighted avg       0.54      0.50      0.43       664

# {'eval_loss': 0.9240626096725464, 'eval_precision': 0.5355551198894325, 'eval_recall': 0.49849397590361444, 'eval_f1': 0.4347578576533078, 'eval_runtime': 51.136, 'eval_samples_per_second': 12.985, 'eval_steps_per_second': 1.623, 'epoch': 2.0}
# {'loss': 0.8594, 'grad_norm': 34.72406005859375, 'learning_rate': 2.5e-05, 'epoch': 3.0}
#  50%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                   | 1488/2976 [55:10<59:23,  2.40s/it]
#               precision    recall  f1-score   support███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [00:53<00:00,  1.53it/s]

#            0       0.27      0.77      0.40        78
#            1       0.53      0.29      0.37       310
#            2       0.69      0.68      0.69       276

#     accuracy                           0.51       664
#    macro avg       0.50      0.58      0.49       664
# weighted avg       0.57      0.51      0.51       664

# {'eval_loss': 0.908385694026947, 'eval_precision': 0.5656866411129918, 'eval_recall': 0.5075301204819277, 'eval_f1': 0.5058888127836014, 'eval_runtime': 54.8315, 'eval_samples_per_second': 12.11, 'eval_steps_per_second': 1.514, 'epoch': 3.0}
# {'loss': 0.748, 'grad_norm': 21.253252029418945, 'learning_rate': 1.6666666666666667e-05, 'epoch': 4.0}
#  67%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                            | 1984/2976 [1:16:21<41:21,  2.50s/it]
#               precision    recall  f1-score   support███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [00:58<00:00,  1.15it/s]

#            0       0.48      0.46      0.47        78
#            1       0.62      0.56      0.59       310
#            2       0.67      0.74      0.70       276

#     accuracy                           0.62       664
#    macro avg       0.59      0.59      0.59       664
# weighted avg       0.62      0.62      0.62       664

# {'eval_loss': 0.8203396797180176, 'eval_precision': 0.620722119602863, 'eval_recall': 0.6234939759036144, 'eval_f1': 0.6208332603801083, 'eval_runtime': 59.3114, 'eval_samples_per_second': 11.195, 'eval_steps_per_second': 1.399, 'epoch': 4.0}
# {'loss': 0.6332, 'grad_norm': 23.149702072143555, 'learning_rate': 8.333333333333334e-06, 'epoch': 5.0}
#  83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                      | 2480/2976 [1:38:30<17:34,  2.13s/it]
#               precision    recall  f1-score   support███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [01:06<00:00,  1.21it/s]

#            0       0.54      0.50      0.52        78
#            1       0.63      0.71      0.67       310
#            2       0.76      0.66      0.71       276

#     accuracy                           0.67       664
#    macro avg       0.64      0.62      0.63       664
# weighted avg       0.67      0.67      0.67       664

# {'eval_loss': 0.8912252187728882, 'eval_precision': 0.6719588581599124, 'eval_recall': 0.6656626506024096, 'eval_f1': 0.666018865033661, 'eval_runtime': 67.4588, 'eval_samples_per_second': 9.843, 'eval_steps_per_second': 1.23, 'epoch': 5.0}
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍| 2968/2976 [1:58:36<00:18,  2.35s/it]

# {'loss': 0.5415, 'grad_norm': 24.761274337768555, 'learning_rate': 0.0, 'epoch': 6.0}
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2976/2976 [1:58:53<00:00,  2.01s/it]
#               precision    recall  f1-score   support███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [00:52<00:00,  1.58it/s]

#            0       0.54      0.46      0.50        78
#            1       0.62      0.67      0.64       310
#            2       0.71      0.68      0.70       276

#     accuracy                           0.65       664
#    macro avg       0.62      0.60      0.61       664
# weighted avg       0.65      0.65      0.65       664

# {'eval_loss': 0.9813154935836792, 'eval_precision': 0.6495918251951616, 'eval_recall': 0.6490963855421686, 'eval_f1': 0.6484202885369127, 'eval_runtime': 53.0371, 'eval_samples_per_second': 12.52, 'eval_steps_per_second': 1.565, 'epoch': 6.0}
# {'train_runtime': 7186.1792, 'train_samples_per_second': 3.312, 'train_steps_per_second': 0.414, 'train_loss': 0.7979135000577537, 'epoch': 6.0}
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2976/2976 [1:59:46<00:00,  2.41s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:51<00:00,  1.88it/s]
#               precision    recall  f1-score   support

#            0       0.50      0.49      0.49        74
#            1       0.65      0.63      0.64       292
#            2       0.73      0.76      0.75       276

#     accuracy                           0.67       642
#    macro avg       0.63      0.63      0.63       642
# weighted avg       0.67      0.67      0.67       642

# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:51<00:00,  1.58it/s]
# {'eval_loss': 0.8995118141174316, 'eval_precision': 0.6695772745529229, 'eval_recall': 0.6713395638629284, 'eval_f1': 0.6702896266746192, 'eval_runtime': 51.7642, 'eval_samples_per_second': 12.402, 'eval_steps_per_second': 1.565, 'epoch': 6.0}

# -----
# vx
# {'eval_loss': 1.9800511598587036, 'eval_precision': 0.7056521727814955, 'eval_recall': 0.705607476635514, 'eval_f1': 0.7032725263999192, 'eval_runtime': 48.2933, 'eval_samples_per_second': 13.294, 'eval_steps_per_second': 1.677, 'epoch': 6.0}

# new split, vx, 3 classes
# {'eval_loss': 1.9766238927841187, 'eval_precision': 0.758306434771169, 'eval_recall': 0.7576219512195121, 'eval_f1': 0.7538789014487921, 'eval_runtime': 46.6118, 'eval_samples_per_second': 14.074, 'eval_steps_per_second': 1.759, 'epoch': 6.0}

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
