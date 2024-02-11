import pandas as pd


def merge_df():
    df = pd.read_csv("scrape/scraped_uniform_test_4.csv", index_col=0)

    scraped_7_df = pd.read_csv("scrape/scraped_7.csv", index_col=0)

    df[df.isnull()] = scraped_7_df

    df.to_csv("scrape/scraped_merged.csv")


def clean_df():
    df = pd.read_csv("scrape/scraped_merged.csv", index_col=0)
    df = df[df["content"].notna()]

    # invalid content but notna
    df.drop(df[df["outlet"].str.startswith("hillreporter")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("quillette")].index, inplace=True)

    df.to_csv("scrape/scraped_merged_clean.csv")

    # [5358 rows x 8 columns]


# merge_df()
# clean_df()
