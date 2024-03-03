import re

import gensim
import nltk
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")


EPOCH = 500
VECTOR_SIZE = 512


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


train_df = pd.read_csv("dataset/train.csv", index_col=0)

train_tagged = train_df.apply(
    lambda r: TaggedDocument(words=tokenize_text(r["content"]), tags=[r.product]),
    axis=1,
)
print(train_tagged)

# documents (iterable of list of TaggedDocument, optional) – Input corpus, can be simply a list of elements, but for larger corpora,consider an iterable that streams the documents directly from disk/network. If you don’t supply documents (or corpus_file), the model is left uninitialized – use if you plan to initialize it in some other way.
# dm ({1,0}, optional) – Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
# vector_size (int, optional) – Dimensionality of the feature vectors.
# min_count (int, optional) – Ignores all words with total frequency lower than this.
# sample (float, optional) – The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
# negative (int, optional) – If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.

model = Doc2Vec(
    dm=1,
    vector_size=VECTOR_SIZE,
    hs=0,
    min_count=2,
    sample=0,
    negative=5,
)

model.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(EPOCH):
    print(f"Training epoch {epoch}...")
    model.train(
        utils.shuffle([x for x in tqdm(train_tagged.values)]),
        total_examples=len(train_tagged.values),
        epochs=1,
    )
    model.alpha -= 0.002
    model.min_alpha = model.alpha


model.save(f"../models/doc2vec_{VECTOR_SIZE}_{EPOCH}.model")
