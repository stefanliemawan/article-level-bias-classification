import sys

import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

tqdm.pandas()

df = pd.read_csv(
    f"../dataset/scraped_clean_{DATASET_VERSION}.csv",
    index_col=0,
)


def get_polarity(content):
    blob = TextBlob(content)
    return blob.sentiment.polarity


def get_subjectivity(content):
    blob = TextBlob(content)
    return blob.sentiment.subjectivity


df["polarity"] = df["content"].progress_apply(get_polarity)
df["subjectivity"] = df["content"].progress_apply(get_subjectivity)

df.to_csv(f"../dataset/scraped_clean_{DATASET_VERSION}.csv")
