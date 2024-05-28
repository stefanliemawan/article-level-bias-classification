import re

import pandas as pd


def merge_df():
    df = pd.read_csv("cleaned_dataset/scrape/scraped_uniform_test_4.csv", index_col=0)

    scraped_7_df = pd.read_csv("scrape/scraped_7.csv", index_col=0)

    df[df.isnull()] = scraped_7_df

    df.to_csv("cleaned_dataset/scraped_merged.csv")


def clean_df():
    df = pd.read_csv("cleaned_dataset/scraped_merged.csv", index_col=0)
    df.dropna(subset=["content"], inplace=True)

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
    df.to_csv("cleaned_dataset/scraped_merged_clean_v1.csv")


# merge_df()
clean_df()
