import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from transformers import BertTokenizer

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"
print(f"dataset {DATASET_VERSION}")

df = pd.read_csv(
    f"../dataset/scraped_merged_clean_{DATASET_VERSION}.csv",
    index_col=0,
)

# outlet_df = pd.read_csv("../dataset/BAT/ad_fontes/outlets_classes_scores.csv", index_col=0)


# v2_edited
# MEAN 1216.7633225867628
# MEDIAN 900.0
# MAX 15796
# MIN 7


def plot_tokens_count(df):
    tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
    df["tokens_count"] = df.apply(lambda x: len(tokeniser.encode(x["content"])), axis=1)

    print("MEAN_TOKENS_COUNT", np.mean(df["tokens_count"].values))
    print("MEDIAN_TOKENS_COUNT", np.median(df["tokens_count"].values))
    print("MAX_TOKENS_COUNT", np.max(df["tokens_count"].values))
    print("MIN_TOKENS_COUNT", np.min(df["tokens_count"].values))

    plt.clf()
    plt.hist(df["tokens_count"], bins=20)
    # plt.xlim(0, 6000)

    plt.xlabel("Tokens Count")
    plt.ylabel("Frequency")
    plt.title("Tokens Count")
    plt.grid(True)
    # plt.show()
    plt.savefig(f"figures/tokens_count_all_{DATASET_VERSION}.png")

    df = df.sort_values(by=["tokens_count"], ascending=False)
    print(df.head())


def plot_reliability_score(df):
    plt.clf()

    plt.hist(df["reliability_score"], bins=10)
    print("MAX_RELIABILITY", np.max(df["reliability_score"].values))
    print("MIN_RELIABILITY", np.min(df["reliability_score"].values))

    plt.title("Reliability Score Distribution")
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/reliability_score.png")


# def plot_outlet_reliability_score():
#     print(outlet_df)


# plot_tokens_count(df)
# plot_tokens_count(df)
plot_reliability_score(df)
# plot_outlet_reliability_score()
