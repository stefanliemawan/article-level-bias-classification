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
        "accuracy": accuracy,
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


# V3 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#               precision    recall  f1-score   support

#            0       0.62      0.18      0.27        74
#            1       0.66      0.76      0.70       292
#            2       0.76      0.79      0.78       276

#     accuracy                           0.70       642
#    macro avg       0.68      0.57      0.58       642
# weighted avg       0.70      0.70      0.69       642

# {'accuracy': 0.7040498442367601, 'precision': 0.6990965451838657, 'recall': 0.7040498442367601, 'f1': 0.6856967728656046}
# Root Mean Squared Error: 7.213715142373027
# R-squared Score: 0.34674209184813376

# V4 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#               precision    recall  f1-score   support

#            0       0.62      0.18      0.27        74
#            1       0.66      0.76      0.71       292
#            2       0.76      0.78      0.77       276

#     accuracy                           0.70       642
#    macro avg       0.68      0.57      0.58       642
# weighted avg       0.70      0.70      0.68       642

# {'accuracy': 0.7024922118380063, 'precision': 0.6979456384008158, 'recall': 0.7024922118380063, 'f1': 0.6842438697072533}
# Root Mean Squared Error: 7.171388583275486
# R-squared Score: 0.3543855987213742


# vx, new split 4 classes
#               precision    recall  f1-score   support

#            0       0.50      0.04      0.07        26
#            1       0.55      0.31      0.40        55
#            2       0.44      0.39      0.42       109
#            3       0.81      0.93      0.87       405

#     accuracy                           0.74       595
#    macro avg       0.58      0.42      0.44       595
# weighted avg       0.71      0.74      0.71       595

# {'accuracy': 0.7378151260504202, 'precision': 0.7074338920376668, 'recall': 0.7378151260504202, 'f1': 0.707936758773465}
# Root Mean Squared Error: 7.175571322301571
# R-squared Score: 0.22236351244414454
