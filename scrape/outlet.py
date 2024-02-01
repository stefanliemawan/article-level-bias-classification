from bs4 import BeautifulSoup
import requests
import re

regex = re.compile(".*rticle|story|content|post.*")
# regex = re.compile(".*rticle|story|content|post|ssrcss|body|text|Afg.*")

# elitedaily = Afg

class Outlet:
    def __init__(self) -> None:
        pass


def parse_html(text):
    clean_text = re.sub(re.compile("<.*?>"), "", text)

    return clean_text


def get_soup(url):
    html_doc = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5
    ).content
    soup = BeautifulSoup(html_doc, "html.parser")

    return soup


def uniform_scrape(url):
    soup = get_soup(url)
    article = soup.find(class_=regex)

    if not article:
        article = soup.find(id=regex)

    content = ""
    for p in article.find_all("p"):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


# TODO - basically try and build functions that works on most outlets
# TODO - find a way to clean noisy text
# TODO - add a timeout function for 443 max retries, save some time
