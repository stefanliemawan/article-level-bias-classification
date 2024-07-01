import sys

import pandas as pd
import utils.functions as functions
from transformers import AutoTokenizer
from utils.chunk_model import ChunkModel

CHUNK_SIZE = 512
NUM_TF_LAYERS = 2
HIDDEN_DIM = 768
EPOCHS = 3
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


# train_df, test_df, valid_df = functions.generate_title_content_features(
#     train_df, test_df, valid_df
# )
train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
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

# vx + rescraped, title + content, bert-base-cased, warmup_steps: 162, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2
#               precision    recall  f1-score   support

#            0       0.44      0.67      0.53        27
#            1       0.41      0.48      0.44        54
#            2       0.39      0.50      0.44       104
#            3       0.91      0.79      0.84       384

#     accuracy                           0.70       569
#    macro avg       0.54      0.61      0.56       569
# weighted avg       0.74      0.70      0.72       569

# {'loss': 0.9091169238090515, 'precision': 0.7440445027139698, 'recall': 0.6994727592267135, 'f1': 0.7163546531981585}

# vx + rescraped, outlet + title + content, bert-base-cased, warmup_steps: 162, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2

#               precision    recall  f1-score   support

#            0       0.37      0.59      0.46        27
#            1       0.32      0.43      0.37        54
#            2       0.40      0.47      0.43       104
#            3       0.92      0.79      0.85       384

#     accuracy                           0.69       569
#    macro avg       0.50      0.57      0.53       569
# weighted avg       0.74      0.69      0.71       569

# {'loss': 0.9438098669052124, 'precision': 0.7384057291425898, 'recall': 0.687170474516696, 'f1': 0.7071647651827099}
