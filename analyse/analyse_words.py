import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

DATASET_VERSION = "v4"

print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)


text = " ".join(train_df["content"])


def generate_word_cloud(text):
    # Generate word cloud
    wordcloud = WordCloud(width=5000, height=3000, background_color="white").generate(
        text
    )

    # Display the word cloud
    # plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"figures/{DATASET_VERSION}_wordcloud.png")


generate_word_cloud(text)


def word_occurrences(text):
    # Split the text into words
    words = re.findall(r"\w+", text)

    # Count the occurrences of each word
    word_counts = Counter(words)

    return word_counts


word_counts = word_occurrences(text)

word_occurrences_df = pd.DataFrame(list(word_counts.items()), columns=["word", "count"])
word_occurrences_df = word_occurrences_df.sort_values(by="count", ascending=False)

word_occurrences_df.to_csv(
    f"figures/{DATASET_VERSION}_word_occurrences_csv", index=False
)
