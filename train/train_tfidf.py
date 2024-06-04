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

tfidf_vectorizer = TfidfVectorizer(max_features=1000)

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


# vx, new split 4 classes
#               precision    recall  f1-score   support

#            0       1.00      0.04      0.08        24
#            1       0.56      0.29      0.38        51
#            2       0.44      0.41      0.42        99
#            3       0.82      0.93      0.87       370

#     accuracy                           0.74       544
#    macro avg       0.70      0.42      0.44       544
# weighted avg       0.73      0.74      0.71       544

# {'accuracy': 0.7389705882352942, 'precision': 0.7316219412792847, 'recall': 0.7389705882352942, 'f1': 0.709459642701604}
# Root Mean Squared Error: 7.044471273220665
# R-squared Score: 0.2617788981933262

# vx, with outlet
#               precision    recall  f1-score   support

#            0       1.00      0.04      0.08        24
#            1       0.54      0.27      0.36        51
#            2       0.43      0.44      0.44        99
#            3       0.82      0.92      0.87       370

#     accuracy                           0.74       544
#    macro avg       0.70      0.42      0.44       544
# weighted avg       0.73      0.74      0.71       544

# {'accuracy': 0.7371323529411765, 'precision': 0.7336086093108126, 'recall': 0.7371323529411765, 'f1': 0.7099331665186448}
# Root Mean Squared Error: 7.001971395443109
# R-squared Score: 0.2706595262754652
