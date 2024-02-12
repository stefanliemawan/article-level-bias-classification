import re
import pandas as pd


def merge_df():
    df = pd.read_csv("scrape/scraped_uniform_test_4.csv", index_col=0)

    scraped_7_df = pd.read_csv("scrape/scraped_7.csv", index_col=0)

    df[df.isnull()] = scraped_7_df

    df.to_csv("scraped_merged.csv")


def clean_df():
    df = pd.read_csv("scraped_merged.csv", index_col=0)
    df = df[df["content"].notna()]

    # invalid content but notna
    df.drop(df[df["outlet"].str.startswith("hillreporter")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("quillette")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("www.newsday")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("www.oann")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("www.financialbuzz")].index, inplace=True)
    df.drop(
        df[df["outlet"].str.startswith("www.libertynation")].index, inplace=True
    )  # theres a solution for this one, update if rescraped

    df.reset_index(inplace=True)
    df.to_csv("scraped_merged_clean_v1.csv")

    # [5358 rows x 8 columns]


def delete_noise(row):
    content = row["content"].replace("\n", "")

    sentences = content.split(".")

    for index, sentence in enumerate(sentences.copy()):
        if "Link Copied" in sentence:
            sentences[index] = sentence[11:]
        if "Already a subscriber?" in sentence:
            sentences.pop(index)
        if re.match(r"[Cc]opyright.*\d{4}", sentence):
            return ". ".join(sentences[:index])

    content = ". ".join(sentences)

    return content


def clean_content():
    df = pd.read_csv("scraped_merged_clean_v1.csv", index_col=0)

    df["content"] = df.apply(delete_noise, axis=1)

    df.to_csv("scraped_merged_clean_v2.csv")


# merge_df()
clean_df()
clean_content()
