import sys

import pandas as pd
import utils.functions as functions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

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


train_df = pd.concat((train_df, valid_df))

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train_df["features"].values)
x_test = vectorizer.transform(test_df["features"].values)

y_train = train_df["labels"].values
y_test = test_df["labels"].values

print(x_train.shape)
print(x_test.shape)

clf = LogisticRegression()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

y_test_classi = y_test  # for below

report = classification_report(y_test, y_pred)
print(report)


precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)

print(
    {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
)

y_train = train_df["reliability_score"].values
y_test = test_df["reliability_score"].values

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)

# vx, new split, 4 classes
#               precision    recall  f1-score   support

#            0       0.44      0.33      0.38        24
#            1       0.32      0.31      0.32        51
#            2       0.38      0.43      0.41        99
#            3       0.86      0.85      0.85       370

#     accuracy                           0.70       544
#    macro avg       0.50      0.48      0.49       544
# weighted avg       0.70      0.70      0.70       544

# {'precision': 0.7043289262820513, 'recall': 0.6985294117647058, 'f1': 0.700754718733791}
# Root Mean Squared Error: 11.768355756468427
# R-squared Score: -1.060256189378769

# vx, with outlet
#               precision    recall  f1-score   support

#            0       0.45      0.38      0.41        24
#            1       0.34      0.35      0.35        51
#            2       0.37      0.39      0.38        99
#            3       0.86      0.85      0.86       370

#     accuracy                           0.70       544
#    macro avg       0.51      0.49      0.50       544
# weighted avg       0.71      0.70      0.70       544

# {'precision': 0.7056256461617989, 'recall': 0.7003676470588235, 'f1': 0.7027264297753896}
# Root Mean Squared Error: 11.173182888150889
# R-squared Score: -0.857134959596002

# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================

# BELOW is to compare regression vs classi
# {'precision': 0.683636963290562, 'recall': 0.6822429906542056, 'f1': 0.6828527804303766} - classification
# {'precision': 0.5252956817235379, 'recall': 0.5046728971962616, 'f1': 0.5104488636987526} - regression, then map class, then classification

# ----------------------------------------------------------------------------------------------------------------
# code below

# CLASS_RANGES = [(-200, 29.32), (29.33, 43.98), (43.98, 200)]


# def map_to_class(score):
#     for i, (start, end) in enumerate(CLASS_RANGES):
#         if start <= score <= end:
#             return i


# y_pred_classi = [map_to_class(pred) for pred in y_pred]

# precision = precision_score(
#     y_test_classi, y_pred_classi, average="weighted", zero_division=1
# )
# recall = recall_score(y_test_classi, y_pred_classi, average="weighted", zero_division=1)
# f1 = f1_score(y_test_classi, y_pred_classi, average="weighted", zero_division=1)

# print(
#     {
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#     }
# )
