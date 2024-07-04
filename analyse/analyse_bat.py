import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"
print(f"dataset {DATASET_VERSION}")

df = pd.read_csv(
    f"../dataset/scraped_clean_{DATASET_VERSION}.csv",
    index_col=0,
)
train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)

# outlet_df = pd.read_csv("../dataset/BAT/ad_fontes/outlets_classes_scores.csv", index_col=0)


# v2
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
    tokeniser = BertTokenizer.from_pretrained("bert-base-cased")
    df["tokens_count"] = df.apply(lambda x: len(tokeniser.encode(x["content"])), axis=1)

    df.to_csv(f"../dataset/scraped_clean_{DATASET_VERSION}.csv")


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
    print(f"Number of rows with tokens_count below 128: {count_below_100}")

    count_below_512 = df[df["tokens_count"] < 512].shape[0]
    print(f"Number of rows with tokens_count below 512: {count_below_512}")

    count_above_10k = df[df["tokens_count"] > 10000].shape[0]
    print(f"Number of rows with tokens_count below 10k: {count_above_10k}")


def plot_tokens_count_split(df):

    range1 = df[(df["tokens_count"] >= 0) & (df["tokens_count"] < 512)]
    range2 = df[(df["tokens_count"] >= 512) & (df["tokens_count"] < 2048)]
    range3 = df[(df["tokens_count"] >= 2048) & (df["tokens_count"] < 4096)]
    range4 = df[df["tokens_count"] >= 4096]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    range_dict = [
        {
            "ax_position": [0, 0],
            "color": "skyblue",
            "df": range1,
            "text_position": [0.4, 0.95],
            "title": "Tokens Count 0-512",
        },
        {
            "ax_position": [0, 1],
            "color": "orange",
            "df": range2,
            "text_position": [0.6, 0.95],
            "title": "Tokens Count 512-2048",
        },
        {
            "ax_position": [1, 0],
            "color": "green",
            "df": range3,
            "text_position": [0.4, 0.95],
            "title": "Tokens Count 2048-4096",
        },
        {
            "ax_position": [1, 1],
            "color": "red",
            "df": range4,
            "text_position": [0.6, 0.95],
            "title": "Tokens Count > 4096",
        },
    ]

    for range in range_dict:
        ax_position = range["ax_position"]
        text_pos_x, text_pos_y = range["text_position"]
        df_range = range["df"]

        axes[*ax_position].hist(
            df_range["tokens_count"], bins=15, color=range["color"], edgecolor="black"
        )
        axes[*ax_position].set_title(range["title"])
        axes[*ax_position].set_xlabel("Tokens Count")
        axes[*ax_position].set_ylabel("Frequency")
        axes[*ax_position].text(
            text_pos_x,
            text_pos_y,
            f"Count: {len(df_range)}",
            transform=axes[*ax_position].transAxes,
            ha="center",
        )

    plt.suptitle("Articles Tokens Count Distribution Across Ranges")
    plt.tight_layout()

    plt.savefig(f"figures/tokens_count_{DATASET_VERSION}_split_hist.png")


def plot_tokens_count_per_class(df):
    plt.clf()
    # plt.figure(figsize=(10, 6))

    custom_sort_order = [
        "Problematic",
        "Questionable",
        "Generally Reliable",
        "Reliable",
    ]

    # Convert 'class' column to categorical with custom sort order
    df["class"] = pd.Categorical(
        df["class"], categories=custom_sort_order, ordered=True
    )

    average_token_count = df.groupby("class")["tokens_count"].mean()

    # Plotting the histogram of the average token counts
    average_token_count.plot(
        kind="bar",
        color=["darkred", "darkorange", "darkgreen", "darkblue"],
        edgecolor="black",
        alpha=0.7,
        width=0.8,
    )
    plt.xlabel("Class")
    plt.ylabel("Average Tokens Count")
    plt.title("Average Tokens Count by Class")

    plt.xticks(rotation=60, ha="right")

    plt.tight_layout()

    plt.savefig(f"figures/tokens_count_{DATASET_VERSION}_per_class_hist.png")


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

    print(df.sort_values(by=["reliability_score"], ascending=False).head(10))
    print(df.sort_values(by=["reliability_score"], ascending=True).head(10))

    count_below_10 = df[df["reliability_score"] < 10].shape[0]
    print(f"Number of rows with reliability_score below 10: {count_below_10}")
    count_below_20 = df[df["reliability_score"] < 20].shape[0]
    print(f"Number of rows with reliability_score below 20: {count_below_20}")


def plot_dates(df):
    plt.clf()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(by="date")

    before_2019 = df[df["date"] < "2019-01-01"]
    before_2019.to_csv(f"../dataset/{DATASET_VERSION}_pre2019.csv")
    after_2019 = df[df["date"] >= "2019-01-01"]

    print("Articles before 2019:", before_2019.shape[0])
    print("Articles after 2019:", after_2019.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.hist(before_2019["date"], bins=30, color="skyblue", edgecolor="black")
    ax1.set_title("Articles Before 2019")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Frequency")

    ax2.hist(after_2019["date"], bins=30, color="orange", edgecolor="black")
    ax2.set_title("Articles After 2019")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()

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
    plt.xlabel("Bias Score")
    plt.ylabel("Reliability Score")

    plt.savefig("figures/correlation_bias_reliability_score.png")


def plot_outlet_reliability_score(df):
    average_scores = df.groupby("outlet")["reliability_score"].mean()

    # Plot the average reliability scores
    plt.figure(figsize=(50, 6))
    average_scores.plot(kind="bar", color="skyblue")
    plt.title("Average Reliability Score by Outlet")
    plt.xlabel("Outlet")
    plt.ylabel("Average Reliability Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("figures/outlet_reliability_score.png")


def plot_class_weights(train_df):
    train_labels = train_df["labels"]
    class_weight = np.asarray(
        compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
    )
    class_counts = Counter(train_labels)

    classes = ["Problematic", "Questionable", "Generally Reliable", "Reliable"]
    counts = [class_counts[i] for i in range(len(classes))]

    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_weight)

    plt.title("Class Weights Visualization")
    plt.xlabel("Classes")
    plt.ylabel("Class Weights")
    plt.ylim(top=5.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # for i, weight in enumerate(class_weight):
    #     plt.text(i, weight + 0.05, f"{weight:.2f}", ha="center", va="bottom")

    for i, (weight, count) in enumerate(zip(class_weight, counts)):
        plt.text(
            i,
            weight + 0.05,
            f"Count: {count}\nWeight: {weight:.2f}",
            ha="center",
            va="bottom",
        )

    # plt.tight_layout()

    plt.savefig("figures/class_weight.png")


# if "tokens_count" not in df:
#     count_tokens(df)

# plot_tokens_count(df)
# plot_tokens_count_split(df)
# plot_tokens_count_per_class(df)
# plot_reliability_score(df)
# plot_dates(df)
# plot_correlation_tokens_reliability(df)
# plot_correlation_bias_reliability(df)
# plot_outlet_reliability_score(df)
plot_class_weights(train_df)
