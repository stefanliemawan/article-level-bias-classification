from bs4 import BeautifulSoup
import requests
import re


class Outlet:
    def __init__(self) -> None:
        pass


def scrape_outlet(url):
    outlet = url.split("/")[2].split(".")[-2]

    return globals()[f"scrape_{outlet}"](url)


def parse_html(text):
    clean_text = re.sub(re.compile("<.*?>"), "", text)

    return clean_text


def get_soup(url):
    html_doc = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content
    soup = BeautifulSoup(html_doc, "html.parser")

    return soup


def uniform_scrape(url, classname):
    soup = get_soup(url)
    article = soup.find(class_=classname)

    content = ""
    # for p in article.find_all("p", class_=None):
    for p in article.find_all("p"):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


# TODO - basically try and build functions that works on most outlets
