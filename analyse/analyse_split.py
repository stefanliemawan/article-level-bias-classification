import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from transformers import BertTokenizer

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"
print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


train_outlets = set(train_df["outlet"].unique())
test_outlets = set(test_df["outlet"].unique())
valid_outlets = set(valid_df["outlet"].unique())

unique_test_outlets = test_outlets - train_outlets - valid_outlets
unique_valid_outlets = valid_outlets - train_outlets - test_outlets

print("Outlets that exist only in the test set:")
print(unique_test_outlets)
print("Outlets that exist only in the valid set:")
print(unique_valid_outlets)
