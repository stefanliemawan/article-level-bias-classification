import pandas as pd
import utils.functions as functions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

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

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


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

# {'precision': 0.6788933546977032, 'recall': 0.67601246105919, 'f1': 0.6772696229141721}
# SAD
