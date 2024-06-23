import outlet
import pandas as pd
from tqdm import tqdm

import scrape

tqdm.pandas()

NOT_FOUND_OUTLETS = [
    "www.ozy.com",
    "pjmedia.com",
    "www.rawstory.com",
    "www.foxbusiness.com",
    "abcnews.go.com",
]

def rescrape_1():
    outlets = ["www.libertynation.com"]

    df = pd.read_csv("../dataset/scraped.csv", index_col=0)
    df = df[df["content"].isnull() | df["outlet"].isin(outlets)]

    print(df)

    df["content"] = df.progress_apply(scrape.scrape_content, axis=1)
    print(f"{df["content"].count()} out of {len(df)} rows of articles text are scraped")

    df.to_csv("rescraped_1.csv")

def rescrape_2():
    df = pd.read_csv("rescraped_1.csv")
    outlets = ["www.cato.org"]

    df = df[df["outlet"].isin(outlets)]
    print(df)

    df["content"] = df.progress_apply(scrape.scrape_content, axis=1)
    print(f"{df["content"].count()} out of {len(df)} rows of articles text are scraped")
    
def clean_content(content):
    try:
        return content.replace("\n", "")
    except:
        return

def format_df():
    df = pd.read_csv("../dataset/rescraped_2.csv", index_col=0)
    df["content"] = df["content"].progress_apply(clean_content)

    print(df["content"].isnull().sum(), "rows are empty out of", len(df))

    df.to_csv("../dataset/rescraped_2.csv")




format_df()
# rescraped_1
# 757 rows are empty out of 956
# rescraped_2
# 692 rows are empty out of 955

# test_url = (
#     "https://www.nationalreview.com/news/j-d-vance-launches-senate-bid-joins-crowded-ohio-gop-primary-field/"
# )
# outlet.uniform_scrape(test_url)

