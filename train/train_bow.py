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

train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

# train_df["content"] = train_df.apply(functions.preprocess_content, axis=1)
# test_df["content"] = test_df.apply(functions.preprocess_content, axis=1)
# valid_df["content"] = valid_df.apply(functions.preprocess_content, axis=1)

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


# V2
#               precision    recall  f1-score   support

#            0       0.48      0.53      0.50        74
#            1       0.66      0.65      0.65       292
#            2       0.76      0.75      0.75       276

#     accuracy                           0.68       642
#    macro avg       0.63      0.64      0.63       642
# weighted avg       0.68      0.68      0.68       642

# {'precision': 0.6788933546977032, 'recall': 0.67601246105919, 'f1': 0.6772696229141721}
# Root Mean Squared Error: 17.968150650094042
# R-squared Score: -3.0529706496787528

# V3
#               precision    recall  f1-score   support

#            0       0.49      0.50      0.50        74
#            1       0.66      0.67      0.66       292
#            2       0.76      0.74      0.75       276

#     accuracy                           0.68       642
#    macro avg       0.64      0.64      0.64       642
# weighted avg       0.68      0.68      0.68       642

# {'precision': 0.683636963290562, 'recall': 0.6822429906542056, 'f1': 0.6828527804303766}
# Root Mean Squared Error: 16.379482558072976
# R-squared Score: -2.367961002518596

# ================================================================================================================================================

# BELOW is to compare regression vs classi
# {'precision': 0.683636963290562, 'recall': 0.6822429906542056, 'f1': 0.6828527804303766} - classification
# {'precision': 0.5252956817235379, 'recall': 0.5046728971962616, 'f1': 0.5104488636987526} - regression, then map class, then classification

CLASS_RANGES = [(-200, 29.32), (29.33, 43.98), (43.98, 200)]


def map_to_class(score):
    for i, (start, end) in enumerate(CLASS_RANGES):
        if start <= score <= end:
            return i


y_pred_classi = [map_to_class(pred) for pred in y_pred]

precision = precision_score(
    y_test_classi, y_pred_classi, average="weighted", zero_division=1
)
recall = recall_score(y_test_classi, y_pred_classi, average="weighted", zero_division=1)
f1 = f1_score(y_test_classi, y_pred_classi, average="weighted", zero_division=1)

print(
    {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
)
