import pandas as pd
from bs4 import BeautifulSoup
import requests
import re


df = pd.read_csv("dataset/BAT/ad_fontes/articles_sorted_by_outlet_occurences.csv")
# print(df.head())


def parse_html(text):
    clean_text = re.sub(re.compile("<.*?>"), "", text)

    return clean_text


def get_soup(url):
    html_doc = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content
    soup = BeautifulSoup(html_doc, "html.parser")

    return soup


def scrape_standard(url):
    ...


def scrape_19thnews(url):
    soup = get_soup(url)
    article = soup.find(id="article-data")

    content = article.get_text(strip=True)
    content = re.search(
        r"<!-- wp:paragraph -->.+<!-- \\\/wp:paragraph -->", content
    ).group()

    content = parse_html(content)

    return content


def scrape_rt(url):
    soup = get_soup(url)
    article = soup.find(class_="article__text")

    content = ""
    for p in article.find_all("p"):
        if not p.find("strong"):  # for the share story stuff at the bottom
            p_text = p.get_text(strip=True)
            content += p_text

    return content


def scrape_newsmax(url):
    soup = get_soup(url)
    article = soup.find(id="mainArticleDiv")

    content = ""
    for p in article.find_all("p"):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_newsweek(url):
    soup = get_soup(url)
    article = soup.find(class_="article-body")

    content = ""
    for p in article.find_all("p"):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


# url is redirected to https://scrippsnews.com/
def scrape_newsy(url):
    soup = get_soup(url)
    article = soup.find(class_="story-transcript")

    content = article.get_text(strip=True)

    return content


def scrape_nj(url):
    soup = get_soup(url)
    article = soup.find(class_="article__story")

    content = ""
    for p in article.find_all("p"):
        p_text = p.get_text(strip=True)
        if (
            "Our journalism needs your support. Please subscribe today to NJ.com."
            in p_text
        ):
            break
        content += p_text

    return content


def scrape_npr(url):
    soup = get_soup(url)
    article = soup.find(class_="storytext")

    content = ""
    for p in article.find_all("p", class_=None):
        parent_class = p.find_parent().attrs["class"][0]
        if parent_class != "caption":
            p_text = p.get_text(strip=True)
            content += p_text

    return content


def scrape_occupy_democrats(url):
    soup = get_soup(url)
    article = soup.find(class_="post-content")

    content = ""
    for p in article.find_all("p", class_=None):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_palmer_report(url):
    soup = get_soup(url)
    article = soup.find(class_="fl-post-content")

    content = ""
    for p in article.find_all("p", class_=None):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_pbs(url):
    soup = get_soup(url)
    article = soup.find(class_="body-text")

    content = article.get_text(strip=True)

    return content


def scrape_content(row):
    url = row["article_url"]

    content = None
    try:
        if url.startswith("https://19thnews.org"):
            print(f"scraping {url}...")
            content = scrape_19thnews(url)
        elif url.startswith("https://www.rt.com"):
            print(f"scraping {url}...")
            content = scrape_rt(url)
        elif url.startswith("https://www.newsmax.com"):
            print(f"scraping {url}...")
            content = scrape_newsmax(url)
        elif url.startswith("https://www.newsweek.com"):
            print(f"scraping {url}...")
            content = scrape_newsweek(url)
        elif url.startswith("https://www.newsy.com"):
            print(f"scraping {url}...")
            content = scrape_newsy(url)
        elif url.startswith("https://www.nj.com"):
            print(f"scraping {url}...")
            content = scrape_nj(url)
        elif url.startswith("https://www.npr.org"):
            print(f"scraping {url}...")
            content = scrape_npr(url)
        elif url.startswith("https://occupydemocrats.com"):
            print(f"scraping {url}...")
            content = scrape_occupy_democrats(url)
        elif url.startswith("https://www.palmerreport.com"):
            print(f"scraping {url}...")
            content = scrape_palmer_report(url)
        elif url.startswith("https://www.pbs.org"):
            print(f"scraping {url}...")
            content = scrape_pbs(url)
    except Exception as exception:
        print(f"failed, {exception}")

    return content


df["content"] = df.apply(scrape_content, axis=1)
print(df.head())

df.to_csv("scraped_4.csv", index=False)

# scrape_newsmax("")


# newsnation: This site is currently unavailable to visitors from the European Economic Area while we work to ensure your data is protected in accordance with applicable EU laws.
# oann: page not found
# ozy: page not found
