import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats


df = pd.read_csv(
    "cleaned_dataset/scraped_merged_clean_v2.csv",
    index_col=0,
)

outlet_df = pd.read_csv("dataset/BAT/ad_fontes/outlets_classes_scores.csv", index_col=)


def plot_word_count():
    df["word_count"] = df["content"].str.count(" ") + 1
    # print(df.iloc[1815]["content"])
    # print(df["word_count"].sort_values())
    # print(df.loc[1815]["content"]) # repeating content
    # print(df.loc[1802]["content"].count(" ") + 1)

    plt.clf()
    plt.hist(df["word_count"], bins=20)
    plt.ylim(0, 2)

    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.title("Word Count")
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/word_count.png")


def plot_reliability_score():
    plt.clf()

    plt.hist(df["reliability_score"], bins=10)

    plt.title("Reliability Score Distribution")
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/reliability_score.png")


def plot_outlet_reliability_score():
    print(outlet_df)


# plot_word_count()
# plot_reliability_score()
plot_outlet_reliability_score()
