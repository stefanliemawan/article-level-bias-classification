import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import utils.functions as functions
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

GLOVE_VERSION = "glove.6B.300d.txt"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"dataset {DATASET_VERSION}")
print(f"GLOVE_VERSION {GLOVE_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df["content"])

sequences = tokenizer.texts_to_sequences(train_df["content"])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df["content"])

train_sequences = tokenizer.texts_to_sequences(train_df["content"])
max_sequence_length = max(len(seq) for seq in train_sequences)
x_train = pad_sequences(train_sequences, maxlen=max_sequence_length)

test_sequences = tokenizer.texts_to_sequences(test_df["content"])
x_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

valid_sequences = tokenizer.texts_to_sequences(valid_df["content"])
x_valid = pad_sequences(valid_sequences, maxlen=max_sequence_length)

word_index = tokenizer.word_index


def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    return embeddings_index


glove_path = f"../embeddings/{GLOVE_VERSION}"
embeddings_index = load_glove_embeddings(glove_path)

embedding_dim = 300  # Depends on the GloVe file used (e.g., 50d, 100d, 200d)
vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index


embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


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

num_linear_layers = 2

model = Sequential()
model.add(Input(shape=(max_sequence_length,)))
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=512,  # Define your max sequence length
        trainable=False,
    ),
)
model.add(Flatten())
for i in range(num_linear_layers):
    model.add(
        Dense(
            256,
            activation="relu",
        )
    )
    model.add(Dropout(0.2))
model.add(Dense(num_labels, activation="softmax"))

optimiser = tf.keras.optimizers.AdamW(learning_rate=2e-5)

model.compile(
    loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"]
)
model.summary()


model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_valid, y_valid),
    class_weight=class_weights_dict,
    verbose=2,
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

# 10 epoch, batch_size 32, 256 * 2 linear layer
#               precision    recall  f1-score   support

#            0       0.67      0.15      0.24        27
#            1       0.31      0.22      0.26        54
#            2       0.29      0.13      0.18       104
#            3       0.76      0.93      0.84       384

#     accuracy                           0.68       569
#    macro avg       0.50      0.36      0.38       569
# weighted avg       0.62      0.68      0.63       569

# {'precision': 0.6231156910798017, 'recall': 0.6836555360281195, 'f1': 0.633536355461211}

# 4 epoch, batch_size 8, 128 * 1 linear layer
#               precision    recall  f1-score   support

#            0       1.00      0.11      0.20        27
#            1       0.26      0.61      0.37        54
#            2       0.30      0.14      0.19       104
#            3       0.82      0.83      0.82       384

#     accuracy                           0.65       569
#    macro avg       0.59      0.42      0.40       569
# weighted avg       0.68      0.65      0.63       569

# {'precision': 0.6774174858533383, 'recall': 0.648506151142355, 'f1': 0.6344369489994209}
