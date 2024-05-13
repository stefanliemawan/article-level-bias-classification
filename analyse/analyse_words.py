import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

DATASET_VERSION = "v4"

print(f"dataset {DATASET_VERSION}")


df = pd.read_csv(
    "../dataset/scraped_merged_clean_v4.csv",
    index_col=0,
)

text = " ".join(df["content"])


def load_word_list():
    with open("../word_list/words_alpha.txt") as word_file:
        word_list = set(word_file.read().split())

    return word_list


def generate_word_cloud(text):
    # Generate word cloud
    wordcloud = WordCloud(width=5000, height=3000, background_color="white").generate(
        text
    )

    # Display the word cloud
    # plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"figures/wordcloud_{DATASET_VERSION}.png")


def generate_word_occurrences(text):

    words = re.findall(r"(\w+)", text)
    # words = text.split()

    word_counts = Counter(words)

    word_occurrences_df = pd.DataFrame(
        list(word_counts.items()), columns=["word", "count"]
    )
    word_occurrences_df = word_occurrences_df.sort_values(by="count", ascending=False)

    word_occurrences_df.to_csv(
        f"figures/word_occurrences_{DATASET_VERSION}.csv", index=False
    )

    word_list = load_word_list()
    word_list = list(word_list)

    word_not_in_dict_df = word_occurrences_df[
        ~word_occurrences_df["word"].str.lower().isin(word_list)
    ]

    print(word_not_in_dict_df)

    word_not_in_dict_df.to_csv(
        f"figures/word_not_in_dict_{DATASET_VERSION}.csv", index=False
    )


# generate_word_cloud(text)
generate_word_occurrences(text)
