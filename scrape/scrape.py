import pandas as pd
import outlet

df = pd.read_csv("../articles_sorted_by_outlet_occurences.csv")
# print(df.head())


# TODO - only print per outlet
def scrape_content(row):
    url = row["article_url"]
    content = None

    try:
        print(f"\nscraping {row.name} {url}...\n")
        content = outlet.uniform_scrape(url)
    except KeyError:
        # print(f"KeyError for {url}")
        pass
    except Exception as exception:
        print(f"\n{url} failed, {exception}\n")

    return content


# df = (
#     df.groupby("outlet").first().sort_values(by=["outlet_story_count"], ascending=False)
# )
# df["content"] = df.apply(scrape_content, axis=1)
# print(f"{df["content"].count()} out of {len(df)} rows of articles text are scraped")

# df.to_csv("scraped_per_outlet_10.csv", index=False)

# scraped_per_outlet_10 (I think)
# {'article-content': 32, 'article__text': 4, 'article__blocks': 1, 'full-article': 4, 'article-output': 1, 'tds-content': 1, 'td-post-content': 3, 'entry-content': 38, 'story[-]?text': 2, 'Afg.*': 2, 'ssrcss': 1, '^wsw$': 0, 'story-two': 1, 'article.*ody': 68, '.*rticle_content': 1, 'story-body.*': 1, 'story_column': 1, '.*-page-content': 2, 'story-transcript': 1, 'body-description': 4, 'body-text': 2, 'body-content': 2, 'post-body': 8, 'the_content_wrapper': 1, 'mvp-content-main': 1, 'single-post': 26, '^body$': 3, '^article$': 2, '^content$': 26, '.*-content': 29, '.*_content': 5, '.*-article': 1, 'article': 2, 'content': 3, 'body': 1, '<article>': 35, 'is this even used': 21}
# 282 out of 320 rows of articles text are scraped


# df["content"] = df.apply(scrape_content, axis=1)
# print(f"{df["content"].count()} out of {len(df)} rows of articles text are scraped")
# df.to_csv("scraped_uniform_4.csv", index=False)

# scraped_uniform_4
# {'article-content': 741, 'c-blog-post__content': 25, 'cbn-text-formatted': 24, 'RichTextStoryBody': 30, 'a-content': 187, 'article-restofcontent': 17, 'node__content': 46, 'content-core': 8, 'article__text': 33, 'article__blocks': 24, 'full-article': 42, 'article-output': 24, 'tds-content': 25, 'td-post-content': 63, 'entry-content': 817, 'story[-]?text': 50, 'Afg.*': 41, 'ssrcss': 25, '^wsw$': 0, 'story-two': 25, 'article.*ody': 1364, 'mvp-content-main': 16, '.*rticle_content': 24, 'story-body.*': 17, 'story_column': 20, '.*-page-content': 25, 'story-transcript': 25, 'the_content_wrapper': 16, 'body-description': 84, 'body-text': 29, 'body-content': 40, 'post-body': 174, '<article>': 3280, '^single-post$': 63, '^body$': 30, '^article$': 3, '^content$': 36, '.*-content': 203, '.*_content': 32, '.*-article': 0, 'article': 0, 'content': 26, 'body': 2, 'is this even used': 462}
# 5594 out of 6345 rows of articles text are scraped

# test_url = (
#     "https://www.libertynation.com/meghan-and-harry-racism-and-media-behind-megxit/"
# )
# outlet.uniform_scrape(test_url)

# TODO - new compatibility (not yet applied and scraped)
# libertynation


# rt: sometimes max retries

# abcnews: page unavailable
# ajc: unavailable in europe (GDPR)
# axios: enable javascript to continue
# baltimoresun: paywall
# billoreilly: not available in the country
# bizjournals: Request unsuccessful. Incapsula incident ID: 770001240051159111-56456275709533957
# bloomberg: are you a robot
# bringmethenews: need js and disable adblocker
# cato: no soup
# cfo: default to home
# chron: access denied
# cnsnews: not found, default to home
# dissenter: 404
# financialbuzz: not found
# ft: locked by paywall
# hillreporter: 404
# houstonchronicle: access denied
# judicialwatch: access limited
# laconiadailysun: unavailable 451
# meidastouch: 404
# nationalreview: javascript website
# newsday: not found
# newsnation: site unavailable for european
# news.yahoo: no soup
# nysun: page not found
# oann: page not found
# ozy: page not found
# post-gazette: timeout
# quillette: paywall locked
# realclearpolitics: js and adblocker
# reuters: no soup
# seattlepi: access denied
# sfchronicle: access denied
# sfgate: no robots
# sputniknews: not private ssl error
# thegrio: default to google.com
# the hill: access denied
# wsj: no soup
# wvgazzettemail: unavailable in the EU
# zerohedge: article is archived or 404
