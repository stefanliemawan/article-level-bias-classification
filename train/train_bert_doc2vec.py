import os

import functions
import gensim
import nltk
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]
MODEL_NAME = "distilbert-base-uncased"
DOC2VEC_MODEL_NAME = "doc2vec_512_100.model"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df["features"] = train_df["title"] + ". " + train_df["content"]

doc2vec_model = Doc2Vec.load(f"../models/{DOC2VEC_MODEL_NAME}")

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)


def generate_doc2vec_embedding(doc2vec_model, contents):
    doc2vec_vectors = []

    for content in contents:

        tokens = tokeniser(content, return_tensors="pt", padding=True, truncation=True)
        tokenised_text = tokeniser.convert_ids_to_tokens(tokens["input_ids"].squeeze())
        tokenised_text_str = " ".join(tokenised_text)

        doc2vec_vector = doc2vec_model.infer_vector(tokenised_text_str.split())

        doc2vec_vectors.append(doc2vec_vector)

    doc2vec_array = np.array(doc2vec_vectors)

    doc2vec_tensor = torch.tensor(doc2vec_array, dtype=torch.int64)

    return doc2vec_tensor


# def generate_doc2vec_embedding(doc2vec_model, content):
#     return torch.tensor(
#         [doc2vec_model.infer_vector(nltk.word_tokenize(x.lower())) for x in content]
#     ).to(torch.int64)


train_doc2vec_embedding = generate_doc2vec_embedding(
    doc2vec_model, train_df["content"].values.tolist()
)
test_doc2vec_embedding = generate_doc2vec_embedding(
    doc2vec_model, test_df["content"].values.tolist()
)
valid_doc2vec_embedding = generate_doc2vec_embedding(
    doc2vec_model, valid_df["content"].values.tolist()
)

train_labels = torch.tensor(train_df["labels"].values)
test_labels = torch.tensor(test_df["labels"].values)
valid_labels = torch.tensor(valid_df["labels"].values)

print(train_doc2vec_embedding.shape)
print(train_labels.shape)


dataset = DatasetDict(
    {
        "train": Dataset.from_dict(
            {"input_ids": train_doc2vec_embedding, "labels": train_labels}
        ),
        "test": Dataset.from_dict(
            {"input_ids": test_doc2vec_embedding, "labels": test_labels}
        ),
        "valid": Dataset.from_dict(
            {"input_ids": valid_doc2vec_embedding, "labels": valid_labels}
        ),
    }
)

train_labels = np.asarray(train_labels)
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)
class_weights = np.asarray(class_weights).astype(np.float32)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASS_RANGES)
)
functions.train(dataset, model, epoch=3, class_weights=class_weights)

# v2 train-test-valid classification 3 classes, title + ". " content, doc2vec_512_100.model, distilbert-base-uncased, weighted loss
# {'eval_loss': 1.0887564420700073, 'eval_accuracy': 0.45754716981132076, 'eval_precision': 0.7518022427910289, 'eval_recall': 0.45754716981132076, 'eval_f1': 0.2872626244122855, 'eval_runtime': 25.0162, 'eval_samples_per_second': 25.424, 'eval_steps_per_second': 3.198, 'epoch': 3.0}
# TODO - try pretrained?
# TODO - concat tokeniser from bert embeddings
# TODO - try as regression?
