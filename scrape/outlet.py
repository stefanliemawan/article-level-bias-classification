from bs4 import BeautifulSoup
import requests
import re


REGEX_STRINGS = [
    r"article.*ody",
    r"article__text",
    r"article-content",
    r"full-article",
    r"tds-content",
    r"td-post-content",
    r"entry-content",
    r"story[-]?text",
    r"Afg.*",
    r"ssrcss",
    r"^wsw$",
    r"story-body.*",
    r"story_column",
    r".*-page-content",
    r"story-transcript",
    r"body-description",
    r"body-text",
    r"body-content",
    r"post-body",
    r"single-post",
    r"^article$",
    r".*-article",
    r".*-content",
    r".*_content",
    r"article",
    r"content",
    r"body",
]

REGEX_COUNT = {key: 0 for key in REGEX_STRINGS}
REGEX_COUNT["<article>"] = 0
REGEX_COUNT["is this even used"] = 0


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


def find_article(soup: BeautifulSoup, regex):
    article = None

    articles = soup.find_all(class_=regex)

    if articles:
        article = max(articles, key=lambda x: len(x.get_text()))

    if not (article and len(article.get_text()) > 50):
        articles = soup.find_all(id=regex)
        if articles:
            article = max(articles, key=lambda x: len(x.get_text()))

    if not (article and len(article.get_text()) > 50):
        articles = soup.find_all(itemprop=regex)
        if articles:
            article = max(articles, key=lambda x: len(x.get_text()))

    return article


def uniform_scrape(url):
    soup = get_soup(url)

    for regex_string in REGEX_STRINGS:
        print(regex_string)
        regex = re.compile(regex_string)
        article = find_article(soup, regex)

        if article and len(article.get_text()) > 50:
            break
        else:
            article = None

    if article:
        REGEX_COUNT[regex_string] += 1
    else:
        article = soup.find("article")
        REGEX_COUNT["<article>"] += 1

    content = ""
    for p in article.find_all("p"):
        try:
            parent_class = p.find_parent().attrs["class"][0]
        except:
            parent_class = None
        if not (
            parent_class
            and (
                "meta" in parent_class
                or "promo" in parent_class
                or "share" in parent_class
                or "footer" in parent_class
                or "credit" in parent_class
                or "response_content" in parent_class
                or "author" in parent_class
            )
        ):
            if not p.find(["strong", "em"]):
                p_text = p.get_text(strip=True)
                content += p_text

    if (not content or len(content) < 150) or url.startswith(
        "https://beforeitsnews.com/"
    ):
        content = ""
        for p in article.find_all("p"):
            try:
                parent_class = p.find_parent().attrs["class"][0]
            except:
                parent_class = None
            if not (
                parent_class
                and (
                    "meta" in parent_class
                    or "promo" in parent_class
                    or "share" in parent_class
                    or "footer" in parent_class
                    or "credit" in parent_class
                    or "response_content" in parent_class
                    or "author" in parent_class
                )
            ):
                p_text = p.get_text(strip=True)
                content += p_text
        REGEX_COUNT["is this even used"] += 1

    # TODO - tidy up later

    print(content)
    print()
    print(REGEX_COUNT)

    return content


# TODO - basically try and build functions that works on most outlets
# TODO - find a way to clean noisy text
# TODO - add a timeout function for 443 max retries, save some time
# TODO - style = p.find_parent().attrs["style"][0], inline style and keyword MORE: https://www.thecollegefix.com/professor-defends-teaching-students-to-question-covid-19-propaganda-as-nyu-investigation-continues/
