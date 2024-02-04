import pandas as pd
import functions
import outlet

df = pd.read_csv("../dataset/BAT/ad_fontes/articles_sorted_by_outlet_occurences.csv")
# print(df.head())


# TODO - only print per outlet
def scrape_content(row):
    url = row["article_url"]
    content = None

    try:
        print(f"\nscraping {row.name} {url}...\n")
        # content = functions.scrape_outlet(url)
        content = outlet.uniform_scrape(url)
    except KeyError:
        # print(f"KeyError for {url}")
        pass
    except Exception as exception:
        print(f"\n{url} failed, {exception}\n")

    return content


df = (
    df.groupby("outlet").first().sort_values(by=["outlet_story_count"], ascending=False)
)
df["content"] = df.apply(scrape_content, axis=1)
print(f"{df["content"].count()} out of {len(df)} rows of articles text are scraped")

df.to_csv("scraped_per_outlet_8.csv", index=False)

# {'article.*ody': 96, 'article__text': 3, 'article-content': 15, 'full-article': 4, 'tds-content': 1, 'td-post-content': 2, 'entry-content': 31, 'story[-]?text': 2, 'Afg.*': 2, 'ssrcss': 1, '^wsw$': 0, 'story-body.*': 1, 'story_column': 1, '.*-page-content': 2, 'story-transcript': 1, 'body-description': 4, 'body-text': 2, 'body-content': 2, 'post-body': 8, 'single-post': 28, '^article$': 5, '.*-article': 12, '.*-content': 43, '.*_content': 7, 'article': 3, 'content': 3, 'body': 1, '<article>': 33, 'is this even used': 29}
# 282 out of 320 rows of articles text are scraped

# df["content"] = df.apply(scrape_content, axis=1)
# print(f"{df["content"].count()} out of {len(df)} rows of articles text are scraped")
# df.to_csv("scraped_uniform_test_2.csv", index=False)

# {'article.*ody': 2065, 'single-post': 1317, 'full-article': 91, 'tds-content': 0, 'body-description': 97, 'body-text': 29, 'body-content': 25, 'post-body': 152, 'story.*text': 267, 'Afg.*': 38, 'ssrcss': 25, 'wsw': 28, 'story-transcript': 25, '.*-content': 1192, '.*_content': 115, 'content': 122, 'body': 89, 'article': 590}
# 5754 out of 6345 rows of articles text are scraped


# test_url = "https://www.propublica.org/article/tax-funded-forest-institute-in-oregon-misled-public-may-have-broken-state-law-audit-finds"
# outlet.uniform_scrape(test_url)


# {'article.*ody': 46, 'article__story': 0, 'mainArticleDiv': 0, 'story-transcript': 0, 'full-article': 1, 'tds-content': 1, 'body-description': 8, 'body-text': 1, 'body-content': 0, 'post-body': 7, 'single-post': 71, 'story.*text': 12, 'Afg.*': 1, 'ssrcss': 1, 'wsw': 1, '.*-content': 39, '.*_content': 12, 'body': 112, 'article': 58}
# 276 out of 320 rows of articles text are scraped


# rt: sometimes max retries

# newsnation: This site is currently unavailable to visitors from the European Economic Area while we work to ensure your data is protected in accordance with applicable EU laws.
# oann: page not found
# ozy: page not found
# quillette: paywall locked
# newsday: not found
# baltimoresun: paywall
# ajc: unavailable in europe (GDPR)
# judicialwatch: access limited
# sfchronicle: access denied
# hillreporter: 404
