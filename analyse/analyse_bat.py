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

# vx
# MEAN_TOKENS_COUNT 1184.8609899487958
# MEDIAN_TOKENS_COUNT 885.0
# MAX_TOKENS_COUNT 15530
# MIN_TOKENS_COUNT 7

# MAX_RELIABILITY 58.67
# MIN_RELIABILITY 1.0

# Number of rows with tokens count below 100: 106
# Number of rows with tokens count below 512: 1206

# Number of rows with reliability_score below 10: 29


def count_tokens(df):
    tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
    df["tokens_count"] = df.apply(lambda x: len(tokeniser.encode(x["content"])), axis=1)

    df.to_csv(f"../dataset/scraped_merged_clean_{DATASET_VERSION}.csv")


def plot_tokens_count(df):
    print("MEAN_TOKENS_COUNT", np.mean(df["tokens_count"].values))
    print("MEDIAN_TOKENS_COUNT", np.median(df["tokens_count"].values))
    print("MAX_TOKENS_COUNT", np.max(df["tokens_count"].values))
    print("MIN_TOKENS_COUNT", np.min(df["tokens_count"].values))

    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.hist(df["tokens_count"], bins=20, edgecolor="black")
    # plt.xlim(0, 6000)

    plt.xlabel("Tokens Count")
    plt.ylabel("Frequency")
    plt.title("Tokens Count Distribution")

    plt.savefig(f"figures/tokens_count_{DATASET_VERSION}_hist.png")

    print(df.sort_values(by=["tokens_count"], ascending=False).head(20))
    print(df.sort_values(by=["tokens_count"], ascending=True).head(20))

    count_below_100 = df[df["tokens_count"] < 100].shape[0]
    print(f"Number of rows with tokens_count below 100: {count_below_100}")

    count_below_512 = df[df["tokens_count"] < 512].shape[0]
    print(f"Number of rows with tokens_count below 512: {count_below_512}")


def plot_reliability_score(df):
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.hist(df["reliability_score"], bins=20, edgecolor="black")
    print("MAX_RELIABILITY", np.max(df["reliability_score"].values))
    print("MIN_RELIABILITY", np.min(df["reliability_score"].values))

    plt.xlabel("Reliability Score")
    plt.ylabel("Frequency")
    plt.title("Reliability Score Distribution")

    plt.savefig("figures/reliability_score_hist.png")

    count_below_10 = df[df["reliability_score"] < 10].shape[0]
    print(f"Number of rows with reliability_score below 10: {count_below_10}")
    count_below_20 = df[df["reliability_score"] < 20].shape[0]
    print(f"Number of rows with reliability_score below 20: {count_below_20}")


def plot_dates(df):
    plt.clf()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(by="date")

    plt.figure(figsize=(10, 6))
    plt.hist(df["date"], bins=20, edgecolor="black")
    plt.xlabel("Date")
    plt.ylabel("Frequency")
    plt.title("Distribution of Articles By Date")
    # plt.xticks(rotation=45)

    plt.savefig("figures/dates_hist.png")


def plot_correlation_tokens_reliability(df):
    correlation, p_value = stats.pearsonr(df["reliability_score"], df["tokens_count"])
    print(f"Pearson correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="reliability_score", y="tokens_count", data=df)
    sns.regplot(
        x="reliability_score", y="tokens_count", data=df, scatter=False, color="red"
    )

    plt.title(
        f"Reliability Score vs Tokens Count\nPearson correlation: {correlation:.2f}"
    )
    plt.xlabel("Reliability Score")
    plt.ylabel("Tokens Count")

    plt.savefig("figures/correlation_tokens_reliability_score.png")


def plot_correlation_bias_reliability(df):
    correlation, p_value = stats.pearsonr(df["bias_score"], df["reliability_score"])
    print(f"Pearson correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="bias_score", y="reliability_score", data=df)
    sns.regplot(
        x="bias_score", y="reliability_score", data=df, scatter=False, color="red"
    )

    plt.title(
        f"Bias Score vs Reliability Score \nPearson correlation: {correlation:.2f}"
    )
    plt.xlabel("bias_score")
    plt.ylabel("Reliability Score")

    plt.savefig("figures/correlation_bias_reliability_score.png")


# def plot_outlet_reliability_score():
#     print(outlet_df)


# count_tokens(df)
plot_tokens_count(df)
plot_reliability_score(df)
plot_dates(df)
# plot_correlation_tokens_reliability(df)
# plot_correlation_bias_reliability(df)
