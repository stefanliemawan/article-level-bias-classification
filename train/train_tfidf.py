import sys

import pandas as pd
import utils.functions as functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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

tfidf_vectorizer = TfidfVectorizer()

x_train = tfidf_vectorizer.fit_transform(train_df["features"].values)
x_test = tfidf_vectorizer.transform(test_df["features"].values)

y_train = train_df["labels"].values
y_test = test_df["labels"].values

print(x_train.shape)
print(x_test.shape)

clf = LogisticRegression()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

report = classification_report(y_test, y_pred)
print(report)


accuracy = accuracy_score(y_test, y_pred)
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


# vx + rescraped, new split 4 classes
#               precision    recall  f1-score   support

#            0       1.00      0.11      0.20        27
#            1       0.47      0.17      0.25        54
#            2       0.36      0.31      0.33       104
#            3       0.79      0.94      0.86       384

#     accuracy                           0.71       569
#    macro avg       0.65      0.38      0.41       569
# weighted avg       0.69      0.71      0.67       569

# {'precision': 0.6904947579415414, 'recall': 0.7117750439367311, 'f1': 0.6725642034140528}
# Root Mean Squared Error: 6.566735323176123
# R-squared Score: 0.369010342588014

# vx + rescraped, with outlet (maybe not the right way to include outlet)
#               precision    recall  f1-score   support

#            0       1.00      0.11      0.20        27
#            1       0.50      0.17      0.25        54
#            2       0.36      0.32      0.34       104
#            3       0.79      0.94      0.86       384

#     accuracy                           0.72       569
#    macro avg       0.66      0.38      0.41       569
# weighted avg       0.70      0.72      0.68       569

# {'precision': 0.6957633399277345, 'recall': 0.7152899824253075, 'f1': 0.6760595491600301}
# Root Mean Squared Error: 6.461067706981272
# R-squared Score: 0.38915390648029846