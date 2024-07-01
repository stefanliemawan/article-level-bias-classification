import pandas as pd
from summarizer import Summarizer
from tqdm import tqdm

SEED = 42
MODEL_NAME = "bert-base-uncased"
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]


df = pd.read_csv("../cleaned_dataset/scraped_clean_v2.csv", index_col=0)
summariser = Summarizer()


def summarise_text(row):
    content = row["content"]
    # word_count = content.count(" ")
    # min_length = min(word_count, 512)
    summarised = summariser(content)

    return summarised


tqdm.pandas()
df["content"] = df.progress_apply(summarise_text, axis=1)

df.to_csv("scraped_clean_v2+.csv", index=False)

# according to tqdm, this takes 7 hours, find quicker way
