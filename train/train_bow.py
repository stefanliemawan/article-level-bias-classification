import pandas as pd
import utils.functions as functions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

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

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Evaluate the model performance
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

# {'precision': 0.6788933546977032, 'recall': 0.67601246105919, 'f1': 0.6772696229141721}
# SAD


y_train = train_df["reliability_score"].values
y_test = test_df["reliability_score"].values

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
