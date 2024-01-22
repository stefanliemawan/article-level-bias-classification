from bs4 import BeautifulSoup
import requests
import re


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


def scrape_occupydemocrats(url):
    soup = get_soup(url)
    article = soup.find(class_="post-content")

    content = ""
    for p in article.find_all("p", class_=None):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_palmerreport(url):
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


def scrape_pjmedia(url):
    soup = get_soup(url)
    article = soup.find(class_="post-body")

    content = ""
    for p in article.find_all("p", class_=None):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_politico(url):
    soup = get_soup(url)

    content = ""
    for p in soup.find_all("p", class_="story-text__paragraph"):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_politicususa(url):
    soup = get_soup(url)
    article = soup.find(class_="textcontent")

    content = ""
    for p in article.find_all("p", class_=None):
        parent_attrs = p.find_parent().attrs
        if not parent_attrs:
            p_text = p.get_text(strip=True)
            content += p_text

    return content


def scrape_politifact(url):
    soup = get_soup(url)
    article = soup.find(class_="m-textblock")

    content = ""
    for p in article.find_all("p", class_=None):
        if not p.find("strong"):  # for the related story stuff at the bottom
            p_text = p.get_text(strip=True)
            content += p_text

    return content


def scrape_popsugar(url):
    soup = get_soup(url)
    article = soup.find(class_="content")

    content = ""
    for p in article.find_all("p", class_=None):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_prageru(url):
    soup = get_soup(url)
    article = soup.find(class_="richText")

    content = article.get_text(strip=True)

    return content


def scrape_qz(url):
    soup = get_soup(url)
    article = soup.find(class_="js_post-content")

    content = article.get_text(strip=True)

    return content


def scrape_rawstory(url):
    soup = get_soup(url)
    article = soup.find(class_="body-description")

    content = ""
    for p in article.find_all("p", class_=None):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_reason(url):
    soup = get_soup(url)
    article = soup.find(class_="entry-content")

    content = ""
    for p in article.find_all("p", class_=None):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_newsandguts(url):
    soup = get_soup(url)
    article = soup.find(class_="post-content")

    content = ""
    for p in article.find_all("p"):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_defensenews(url):
    soup = get_soup(url)
    article = soup.find(class_="articleBody")

    content = ""
    for p in article.find_all("p", class_="body-paragraph"):
        p_text = p.get_text(strip=True)
        content += p_text

    return content


def scrape_stream(url):
    soup = get_soup(url)
    article = soup.find(class_="content")

    content = ""
    for p in article.find_all("p", class_=None):
        if p.find("i"):
            break
        p_text = p.get_text()
        content += p_text

    return content


# TODO - find pattern for class_name
# TODO - use several general scraping methods instead of outlet by outlet
# TODO - check on strip=True, in stream it connects multiple words into one
