import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

# do majority votes based on outlet and record the f1

outlet_majority_label = (
    train_df.groupby("outlet")["labels"].apply(lambda x: x.mode()[0]).reset_index()
)

# print(outlet_majority_label)


def predict_majority_label(outlet):
    if outlet in outlet_majority_label["outlet"].values:
        return outlet_majority_label.loc[
            outlet_majority_label["outlet"] == outlet, "labels"
        ].values[0]
    else:
        return None  # Handle case when outlet is not found in the DataFrame


def calculate_metrics(df):
    preds = df["outlet"].apply(predict_majority_label).values
    labels = df["labels"]

    precision = precision_score(labels, preds, average="weighted", zero_division=1)
    recall = recall_score(labels, preds, average="weighted", zero_division=1)
    f1 = f1_score(labels, preds, average="weighted", zero_division=1)

    print(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


print("TRAIN")
calculate_metrics(train_df)
print("TEST")
calculate_metrics(test_df)
print("VALID")
calculate_metrics(valid_df)

# TRAIN
# {
#     "precision": 0.6777043682896989,
#     "recall": 0.6780942777917822,
#     "f1": 0.6758973993196211,
# }
# TEST
# {
#     "precision": 0.8015031675263043,
#     "recall": 0.8006230529595015,
#     "f1": 0.8008682506283301,
# }
# VALID
# {
#     "precision": 0.7379677025015344,
#     "recall": 0.7364457831325302,
#     "f1": 0.7369006270201327,
# }
