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

DATASET_VERSION = "v4"

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

#            0       0.60      0.16      0.26        74
#            1       0.66      0.77      0.71       292
#            2       0.77      0.79      0.78       276

#     accuracy                           0.71       642
#    macro avg       0.68      0.57      0.58       642
# weighted avg       0.70      0.71      0.69       642

# {'accuracy': 0.7087227414330218, 'precision': 0.7020962286836469, 'recall': 0.7087227414330218, 'f1': 0.6891113551980523}
# Root Mean Squared Error: 7.229617918388458
# R-squared Score: 0.3438586774243867
