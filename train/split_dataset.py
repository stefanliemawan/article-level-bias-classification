import numpy as np
import pandas as pd
from click import group
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]


df = pd.read_csv(
    "../cleaned_dataset/scraped_merged_clean_v2.csv",
    index_col=0,
)


def map_to_class(score):
    for i, (start, end) in enumerate(CLASS_RANGES):
        if start <= score <= end:
            return i


def preprocess_outlet(outlet):
    if "www." in outlet:
        outlet = outlet[5:]
    if ".org" in outlet or ".com" in outlet:
        outlet = outlet[:-4]

    return outlet


df["outlet"] = df["outlet"].apply(preprocess_outlet)

df["labels"] = df["reliability_score"].apply(map_to_class)

grouped_df = df.groupby(["labels", "outlet"])


train_data, test_data, valid_data = [], [], []

for group_name, group_df in grouped_df:
    if len(group_df) <= 5:
        train_data.append(group_df)
        # chance = np.random.random()
        # if chance < (1 / 3):
        #     train_data.append(group_df)
        # elif chance < (2 / 3):
        #     test_data.append(group_df)
        # else:
        #     test_data.append(group_df)
    else:
        train_group, test_group = train_test_split(
            group_df, test_size=0.25, random_state=SEED
        )
        test_group, valid_group = train_test_split(
            test_group, test_size=0.4, random_state=SEED
        )

        train_data.append(train_group)
        test_data.append(test_group)
        valid_data.append(valid_group)


# Concatenate data subsets to form train, test, and validation sets

train_set = pd.concat(train_data)
test_set = pd.concat(test_data)
valid_set = pd.concat(valid_data)

# Shuffle the datasets to ensure randomness
train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)
valid_set = valid_set.sample(frac=1, random_state=42).reset_index(drop=True)

train_set.to_csv("dataset/train.csv")
test_set.to_csv("dataset/test.csv")
valid_set.to_csv("dataset/valid.csv")

print("TRAIN")
print(train_set.shape)
print(train_set["labels"].value_counts())
# print(train_set["outlet"].value_counts(normalize=True))

print("TEST")
print(test_set.shape)
print(test_set["labels"].value_counts())
# print(test_set["outlet"].value_counts(normalize=True))

print("VALID")
print(valid_set.shape)
print(valid_set["labels"].value_counts())
# print(valid_set["outlet"].value_counts(normalize=True))
