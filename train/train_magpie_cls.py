import sys

import pandas as pd
import utils.functions as functions
from transformers import AutoTokenizer
from utils.chunk_model import ChunkModel

CHUNK_SIZE = 156
NUM_TF_LAYERS = 2
HIDDEN_DIM = 768
EPOCHS = 3
DROPOUT_PROB = 0.2
# TF_MODEL_NAME = "mediabiasgroup/magpie-babe-ft"
TF_MODEL_NAME = "mediabiasgroup/da-roberta-pt"

POOLING_STRATEGY = "mean"

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"MODEL: {TF_MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, POOLING_STRATEGY {POOLING_STRATEGY}, NUM_TF_LAYERS {NUM_TF_LAYERS}, HIDDEN_DIM {HIDDEN_DIM}, EPOCHS {EPOCHS}, DROPOUT {DROPOUT_PROB}"
)


train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
#     train_df, test_df, valid_df
# )

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(TF_MODEL_NAME)


tokenised_dataset = dataset.map(
    functions.tokenise_chunks,
    fn_kwargs={"tokeniser": tokeniser, "chunk_size": CHUNK_SIZE},
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
    pooling_strategy=POOLING_STRATEGY,
    dropout_prob=DROPOUT_PROB,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])

# 132,926,212 parameters according to GPT

# vx + rescraped, title + content, no warmup steps, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, CLS pooling
#               precision    recall  f1-score   support

#            0       0.56      0.19      0.28        27
#            1       0.42      0.56      0.48        54
#            2       0.42      0.49      0.45       104
#            3       0.90      0.86      0.88       384

#     accuracy                           0.73       569
#    macro avg       0.57      0.52      0.52       569
# weighted avg       0.75      0.73      0.74       569

# {'loss': 1.036009430885315, 'precision': 0.7515373221663202, 'recall': 0.7328646748681898, 'f1': 0.736117275094499}

# vx + rescraped, title + content, warmup_steps: 162, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, CLS pooling
#               precision    recall  f1-score   support

#            0       0.46      0.63      0.53        27
#            1       0.36      0.41      0.38        54
#            2       0.41      0.55      0.47       104
#            3       0.93      0.80      0.86       384

#     accuracy                           0.71       569
#    macro avg       0.54      0.60      0.56       569
# weighted avg       0.76      0.71      0.73       569

# {'loss': 0.8415158987045288, 'precision': 0.7577533590596234, 'recall': 0.7117750439367311, 'f1': 0.7293065634451941}
# worse with 500 warmup

# vx + rescraped, title + content, learning_rate 2e-5, warmup_steps: 162, CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, CLS pooling
#               precision    recall  f1-score   support

#            0       0.45      0.48      0.46        27
#            1       0.30      0.41      0.35        54
#            2       0.41      0.54      0.46       104
#            3       0.94      0.80      0.86       384

#     accuracy                           0.70       569
#    macro avg       0.52      0.56      0.53       569
# weighted avg       0.76      0.70      0.72       569

# {'loss': 0.8991358280181885, 'precision': 0.7558342374893745, 'recall': 0.7012302284710018, 'f1': 0.7225577730868136}


# vx + rescraped, warmup_steps: 162, outlet + title + content , CHUNK_SIZE 512, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, CLS pooling
#               precision    recall  f1-score   support

#            0       0.42      0.52      0.47        27
#            1       0.35      0.43      0.38        54
#            2       0.43      0.55      0.48       104
#            3       0.93      0.82      0.87       384

#     accuracy                           0.72       569
#    macro avg       0.53      0.58      0.55       569
# weighted avg       0.76      0.72      0.73       569

# {'loss': 0.8297753930091858, 'precision': 0.760345238507249, 'recall': 0.7170474516695958, 'f1': 0.7342602984152476}

# vx + rescraped, warmup_steps: 162, title + content , CHUNK_SIZE 156, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, DROPOUT 0.2,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, mean pooling
#               precision    recall  f1-score   support
#               precision    recall  f1-score   support

#            0       0.50      0.70      0.58        27
#            1       0.42      0.46      0.44        54
#            2       0.46      0.57      0.51       104
#            3       0.93      0.83      0.87       384

#     accuracy                           0.74       569
#    macro avg       0.58      0.64      0.60       569
# weighted avg       0.77      0.74      0.75       569

# {'loss': 0.8169947266578674, 'precision': 0.7732142516918309, 'recall': 0.7398945518453427, 'f1': 0.7526923090194392}

# vx + rescraped, warmup_steps: 162, title + content , CHUNK_SIZE 156, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, NO DROPOUT,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, mean pooling
#               precision    recall  f1-score   support

#            0       0.50      0.63      0.56        27
#            1       0.46      0.50      0.48        54
#            2       0.47      0.59      0.52       104
#            3       0.92      0.83      0.87       384

#     accuracy                           0.74       569
#    macro avg       0.59      0.64      0.61       569
# weighted avg       0.77      0.74      0.76       569

# {'loss': 0.8172813057899475, 'precision': 0.7743184031806093, 'recall': 0.7434094903339191, 'f1': 0.7554627857699413}

# vx + rescraped, warmup_steps: 162, outlet + title + content , CHUNK_SIZE 156, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, NO DROPOUT, TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, mean pooling
#               precision    recall  f1-score   support

#            0       0.42      0.59      0.49        27
#            1       0.35      0.41      0.38        54
#            2       0.42      0.56      0.48       104
#            3       0.93      0.80      0.86       384

#     accuracy                           0.71       569
#    macro avg       0.53      0.59      0.55       569
# weighted avg       0.76      0.71      0.73       569

# {'loss': 0.8227108120918274, 'precision': 0.7591502615412127, 'recall': 0.7117750439367311, 'f1': 0.7298321588425787}

# worse with 78 chunk size, even worse with 102
# vx + rescraped, warmup_steps: 162, title + content , CHUNK_SIZE 78, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, NO DROPOUT, TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, mean pooling
#               precision    recall  f1-score   support

#            0       0.49      0.63      0.55        27
#            1       0.39      0.43      0.41        54
#            2       0.46      0.55      0.50       104
#            3       0.91      0.84      0.88       384

#     accuracy                           0.74       569
#    macro avg       0.56      0.61      0.58       569
# weighted avg       0.76      0.74      0.75       569

# {'loss': 0.850915253162384, 'precision': 0.7620966298697005, 'recall': 0.7363796133567663, 'f1': 0.7469557599081315}


# vx + rescraped, warmup_steps: 162, title + content , CHUNK_SIZE 156, NUM_TF_LAYERS 2, HIDDEN_SIZE 768, EPOCHS 3, NO DROPOUT,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, cls pooling
#               precision    recall  f1-score   support

#            0       0.47      0.63      0.54        27
#            1       0.38      0.43      0.40        54
#            2       0.44      0.57      0.49       104
#            3       0.93      0.81      0.87       384

#     accuracy                           0.72       569
#    macro avg       0.55      0.61      0.57       569
# weighted avg       0.76      0.72      0.74       569

# {'loss': 0.8463671803474426, 'precision': 0.7628749748774759, 'recall': 0.7223198594024605, 'f1': 0.7378859890489406}

# vx + rescraped, warmup_steps: 162, title + content , CHUNK_SIZE 156, NUM_TF_LAYERS 1, HIDDEN_SIZE 768, EPOCHS 5, NO DROPOUT,TRANSFORMER_MODEL_NAME mediabiasgroup/magpie-babe-ft, cls pooling
#               precision    recall  f1-score   support

#            0       0.50      0.48      0.49        27
#            1       0.41      0.54      0.46        54
#            2       0.46      0.62      0.53       104
#            3       0.94      0.81      0.87       384

#     accuracy                           0.74       569
#    macro avg       0.58      0.61      0.59       569
# weighted avg       0.78      0.74      0.75       569

# {'loss': 0.8838070034980774, 'precision': 0.782877512068557, 'recall': 0.7363796133567663, 'f1': 0.7532728105721062}
