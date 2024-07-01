import sys

import pandas as pd
import utils.functions as functions
from transformers import AutoTokenizer
from utils.chunk_model import ChunkModel

CHUNK_SIZE = 512
NUM_TF_LAYERS = 2
HIDDEN_DIM = 768
EPOCHS = 14
TF_MODEL_NAME = "bert-base-cased"
DROPOUT_PROB = 0.2

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {TF_MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, NUM_TF_LAYERS {NUM_TF_LAYERS}, HIDDEN_SIZE {HIDDEN_DIM}, EPOCHS {EPOCHS}, DROPOUT {DROPOUT_PROB}"
)


train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(TF_MODEL_NAME)


tokenised_dataset = dataset.map(
    functions.tokenise_chunks,
    fn_kwargs={
        "tokeniser": tokeniser,
        "chunk_size": CHUNK_SIZE,
    },
)

print(tokenised_dataset)


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]
model = ChunkModel(
    tf_model_name=TF_MODEL_NAME,
    num_tf_layers=NUM_TF_LAYERS,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_labels,
    train_labels=train_labels,
    dropout_prob=DROPOUT_PROB,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])

# TODO - apply first chunk pooling? read su
# For the transformer encoder, we add an empty chunk at the beginning of the chunk sequence that is used as the document representation to the classifier, analogous to the [CLS] token commonly used in that way for sentence representations (and used here as the chunk representation)

# title + content, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 10, TRANSFORMER_MODEL_NAME bert-base-uncased, 2 linear layer
# {'loss': 1.7769678831100464, 'precision': 0.7186231414465551, 'recall': 0.719626168224299, 'f1': 0.7175640997635683}
