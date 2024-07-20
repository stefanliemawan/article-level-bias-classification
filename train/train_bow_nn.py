import sys

import numpy as np
import pandas as pd
import tensorflow as tf
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
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
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

vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()

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


# def scheduler(epoch, lr):
#     warmup_epochs = 3
#     if epoch < warmup_epochs:
#         return lr + (0.001 - lr) / warmup_epochs
#     return lr


# lr_scheduler = LearningRateScheduler(scheduler)
optimiser = tf.keras.optimizers.AdamW(learning_rate=2e-5)

model.compile(
    loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=8,
    validation_data=(x_valid, y_valid),
    class_weight=class_weights_dict,
    # callbacks=[lr_scheduler],
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

# 10 epoch, bow, title + content
#               precision    recall  f1-score   support

#            0       0.38      0.44      0.41        27
#            1       0.33      0.35      0.34        54
#            2       0.41      0.42      0.42       104
#            3       0.88      0.85      0.86       384

#     accuracy                           0.71       569
#    macro avg       0.50      0.52      0.51       569
# weighted avg       0.72      0.71      0.71       569

# {'precision': 0.7162298537356339, 'recall': 0.7065026362038664, 'f1': 0.711064316646461}

# 10 epoch, tfidf, title + content
#               precision    recall  f1-score   support

#            0       0.26      0.63      0.37        27
#            1       0.40      0.07      0.12        54
#            2       0.36      0.48      0.41       104
#            3       0.88      0.82      0.85       384

#     accuracy                           0.68       569
#    macro avg       0.48      0.50      0.44       569
# weighted avg       0.71      0.68      0.68       569

# {'precision': 0.7118441600972082, 'recall': 0.6766256590509666, 'f1': 0.6776529851708551}

# use outlet and title as features + bow of content?
