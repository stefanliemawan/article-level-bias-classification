import pandas as pd


def merge_df():
    df = pd.read_csv("scrape/scraped_uniform_test_4.csv", index_col=0)

    scraped_7_df = pd.read_csv("scrape/scraped_7.csv", index_col=0)

    df[df.isnull()] = scraped_7_df

    df.to_csv("scrape/scraped_merged_uniform4_scraped7.csv")


df = pd.read_csv("scrape/scraped_merged_uniform4_scraped7.csv", index_col=0)
df = df[df["content"].notna()]

print(df)
# [5407 rows x 8 columns]