import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from transformers import BertTokenizer

df = pd.read_csv(
    "../cleaned_dataset/scraped_merged_clean_v2.csv",
    index_col=0,
)

# outlet_df = pd.read_csv("../dataset/BAT/ad_fontes/outlets_classes_scores.csv", index_col=0)


def plot_tokens_count():
    tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
    df["tokens_count"] = df.apply(lambda x: len(tokeniser.encode(x["content"])), axis=1)
    print(np.mean(df["tokens_count"].values))  # 1293.1905935899867
    print(np.median(df["tokens_count"].values))  # 895

    plt.clf()
    plt.hist(df["tokens_count"], bins=20)
    plt.ylim(0, 2)

    plt.xlabel("Tokens Count")
    plt.ylabel("Frequency")
    plt.title("Tokens Count")
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/tokens_count.png")


def plot_reliability_score():
    plt.clf()

    plt.hist(df["reliability_score"], bins=10)

    plt.title("Reliability Score Distribution")
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/reliability_score.png")


# def plot_outlet_reliability_score():
#     print(outlet_df)


plot_tokens_count()
# plot_reliability_score()
# plot_outlet_reliability_score()
