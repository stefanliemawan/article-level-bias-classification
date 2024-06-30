import sys

import numpy as np
import pandas as pd
import utils.functions as functions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

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


# vectorizer = TfidfVectorizer(max_features=1000)

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train_df["features"].values)
x_test = vectorizer.transform(test_df["features"].values)
x_valid = vectorizer.transform(valid_df["features"].values)

num_labels = len(pd.unique(train_df["labels"]))

y_train = train_df["labels"].values
y_test = test_df["labels"].values
y_valid = valid_df["labels"].values


print(x_train.shape)
print(x_test.shape)


class_weights = np.asarray(
    compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
).astype(np.float32)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(class_weights_dict)

y_train = to_categorical(y_train, num_classes=num_labels)
y_valid = to_categorical(y_valid, num_classes=num_labels)

model = Sequential()
model.add(
    Dense(
        128,
        input_dim=x_train.shape[1],
        activation="relu",
    )
)
model.add(Dropout(0.2))
model.add(Dense(num_labels, activation="softmax"))

# Compile and train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
    x_train,
    y_train,
    epochs=4,
    batch_size=6,
    validation_data=(x_valid, y_valid),
    class_weight=class_weights_dict,
)

# Predictions and evaluation
predictions = model.predict(x_test)
y_pred = predictions.argmax(axis=1)
print(classification_report(y_test, y_pred))


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

# 6 epoch, bow
#               precision    recall  f1-score   support

#            0       0.89      0.33      0.48        24
#            1       0.35      0.43      0.39        51
#            2       0.39      0.43      0.41        99
#            3       0.87      0.86      0.86       370

#     accuracy                           0.72       544
#    macro avg       0.63      0.51      0.54       544
# weighted avg       0.74      0.72      0.72       544

# {'precision': 0.7366000516980076, 'recall': 0.7169117647058824, 'f1': 0.7206226905839207}

# 6 epoch, tdiff
#               precision    recall  f1-score   support

#            0       1.00      0.08      0.15        24
#            1       0.40      0.53      0.46        51
#            2       0.51      0.45      0.48        99
#            3       0.86      0.90      0.88       370

#     accuracy                           0.75       544
#    macro avg       0.69      0.49      0.49       544
# weighted avg       0.76      0.75      0.74       544

# {'precision': 0.7619584635673662, 'recall': 0.75, 'f1': 0.7374589872512924}
