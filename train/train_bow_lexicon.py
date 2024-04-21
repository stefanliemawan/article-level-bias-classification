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

train_df = pd.read_csv("dataset/v2/train.csv", index_col=0)
test_df = pd.read_csv("dataset/v2/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/v2/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

bias_lexicon = pd.read_csv("dataset/bias_lexicon.csv")
bias_lexicon = bias_lexicon["words"].tolist()

train_df = pd.concat((train_df, valid_df))


def filter_words(content):
    words = content.split(" ")

    filtered_words = [word for word in words if word in bias_lexicon]

    if len(filtered_words) == 0:
        features = "unbiased"
    else:
        features = " ".join(filtered_words)

    return features


train_df["features"] = train_df["features"].apply(filter_words)
test_df["features"] = test_df["features"].apply(filter_words)

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

#               precision    recall  f1-score   support

#            0       0.20      0.05      0.09        74
#            1       0.51      0.51      0.51       292
#            2       0.54      0.64      0.59       276

#     accuracy                           0.52       642
#    macro avg       0.42      0.40      0.39       642
# weighted avg       0.49      0.52      0.49       642

# {'precision': 0.4870291702067403, 'recall': 0.5155763239875389, 'f1': 0.4944493554545354}

# worse, maybe keep the other words and do something with the bias lexicon?

y_train = train_df["reliability_score"].values
y_test = test_df["reliability_score"].values

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


# Mean Squared Error: 121.65277205779424
# R-squared Score: -0.5271746548871781
