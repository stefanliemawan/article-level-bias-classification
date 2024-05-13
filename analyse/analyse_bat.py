import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from transformers import BertTokenizer

v1_df = pd.read_csv(
    "../dataset/scraped_merged_clean_v1.csv",
    index_col=0,
)

v2_edited_df = pd.read_csv(
    "../dataset/scraped_merged_clean_v2_edited.csv",
    index_col=0,
)


v3_df = pd.read_csv(
    "../dataset/scraped_merged_clean_v3.csv",
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

    print("MEAN", np.mean(df["tokens_count"].values))
    print("MEDIAN", np.median(df["tokens_count"].values))
    print("MAX", np.max(df["tokens_count"].values))
    print("MIN", np.min(df["tokens_count"].values))

    plt.clf()
    plt.hist(df["tokens_count"], bins=20)
    # plt.xlim(0, 6000)

    plt.xlabel("Tokens Count")
    plt.ylabel("Frequency")
    plt.title("Tokens Count")
    plt.grid(True)
    # plt.show()
    # plt.savefig("figures/tokens_count_all_v2_edited.png")
    plt.savefig("figures/tokens_count_all_v1.png")


def plot_reliability_score(df):
    plt.clf()

    plt.hist(df["reliability_score"], bins=10)

    plt.title("Reliability Score Distribution")
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/reliability_score.png")


# def plot_outlet_reliability_score():
#     print(outlet_df)


# plot_tokens_count(v2_edited_df)
plot_tokens_count(v2_edited_df)
# plot_reliability_score()
# plot_outlet_reliability_score()


max_content = v2_edited_df["content"].apply(len)
print(max_content)

row = v2_edited_df.loc[max_content.idxmax()]

print(row)
