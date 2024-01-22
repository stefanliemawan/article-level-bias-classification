import pandas as pd
import functions

df = pd.read_csv("../dataset/BAT/ad_fontes/articles_sorted_by_outlet_occurences.csv")
# print(df.head())


# TODO - only print per outlet
def scrape_content(row):
    url = row["article_url"]
    content = None

    try:
        print(f"scraping {url}...")
        content = functions.scrape_outlet(url)
    except KeyError:
        # print(f"KeyError for {url}")
        pass
    except Exception as exception:
        print(f"{url} failed, {exception}")

    return content


df["content"] = df.apply(scrape_content, axis=1)
print(f"{df["content"].count()} rows of articles text are scraped")

df.to_csv("scraped_7.csv", index=False)
# 525 rows of articles text are scraped

# functions.scrape_stream("")

# content not tested: pjmedia, politico, politicususa, politifact, popsugar, prageru, qz, rawstory, reason, newsandguts, defensenews, stream

# rt: sometimes max retries

# newsnation: This site is currently unavailable to visitors from the European Economic Area while we work to ensure your data is protected in accordance with applicable EU laws.
# oann: page not found
# ozy: page not found
# quillette: paywall locked
# newsday: not found
# baltimoresun: paywall
# ajc: unavailable in europe (GDPR)
