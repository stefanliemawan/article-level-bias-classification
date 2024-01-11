import pandas as pd
from bs4 import BeautifulSoup
import requests
import re


df = pd.read_csv("dataset/BAT/ad_fontes/articles_sorted_by_outlet_occurences.csv")
# print(df.head())


def parse_html(text):
    clean_text = re.sub(re.compile("<.*?>"), "", text)

    return clean_text


def scrape_19thnews(url):
    html_doc = requests.get(url).content
    soup = BeautifulSoup(html_doc, "html.parser")
    article = soup.find(id="article-data")

    content = article.get_text()
    content = re.search(
        r"<!-- wp:paragraph -->.+<!-- \\\/wp:paragraph -->", content
    ).group()

    content = parse_html(content)

    return content


def scrape_rt(url):
    html_doc = requests.get(url).content
    soup = BeautifulSoup(html_doc, "html.parser")
    article = soup.find(class_="article__text")

    content = ""
    for p in article.find_all("p"):
        if not p.find("strong"):  # for the share story stuff at the bottom
            p_text = p.get_text()
            content += p_text

    return content


def scrape_newsmax(url):
    html_doc = requests.get(url).content
    soup = BeautifulSoup(html_doc, "html.parser")
    article = soup.find(id="mainArticleDiv")

    content = ""
    for p in article.find_all("p"):
        content += p.get_text()

    return content


def scrape_content(row):
    url = row["article_url"]

    content = None
    if url.startswith("https://19thnews.org"):
        print(f"scraping {url}...")
        content = scrape_19thnews(url)
    elif url.startswith("https://www.rt.com"):
        print(f"scraping {url}...")
        content = scrape_rt(url)
    elif url.startswith("https://www.newsmax.com"):
        print(f"scraping {url}...")
        content = scrape_newsmax(url)

    return content


df["content"] = df.apply(scrape_content, axis=1)
print(df.head())

df.to_csv("scraped_2.csv", index=False)

# scrape_newsmax("")

# fix newsmax


# newsnation: This site is currently unavailable to visitors from the European Economic Area while we work to ensure your data is protected in accordance with applicable EU laws.
