from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
from tqdm import tqdm

from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt


from transformers import AutoTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import functions

import nltk
from nltk.corpus import stopwords


tqdm.pandas(desc="progress-bar")


SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "distilbert-base-uncased"

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

# train_df["content"] = train_df.apply(functions.preprocess_content, axis=1)
# test_df["content"] = test_df.apply(functions.preprocess_content, axis=1)
# valid_df["content"] = valid_df.apply(functions.preprocess_content, axis=1)


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


train_tagged = train_df.apply(
    lambda r: TaggedDocument(words=tokenize_text(r["content"]), tags=[r.Product]),
    axis=1,
)
test_tagged = test_df.apply(
    lambda r: TaggedDocument(words=tokenize_text(r["content"]), tags=[r.Product]),
    axis=1,
)
test_tagged = valid_df.apply(
    lambda r: TaggedDocument(words=tokenize_text(r["content"]), tags=[r.Product]),
    axis=1,
)

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
