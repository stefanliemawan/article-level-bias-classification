import re

import pandas as pd
import utils
from tqdm import tqdm

URL_REGEX = r"http(s)?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
CAPITAL_LETTER_AFTER_PERIOD_REGEX = r"\.(?=[A-Z])"

STANDALONE_LETTERS = {
    " s": "s",
}

# df = pd.read_csv(
#     "../dataset/scraped_merged_clean_v2_edited.csv",
#     index_col=0,
# )

df = pd.read_csv(
    "../dataset/scraped_merged_clean_v3_edited.csv",
    index_col=0,
)

urls_removed = []


def strip_url(content):
    def replace_urls(match):
        url = match.group(0)  # Get the matched URL
        urls_removed.append(url)

        return ""

    content = re.sub(URL_REGEX, replace_urls, content)

    return content


def dot(content):
    content = re.sub(
        CAPITAL_LETTER_AFTER_PERIOD_REGEX, lambda x: x.group(0) + " ", content
    )

    return content


def fix_standalone_letters(content):
    fixed_test = content

    for letter, expanded_form in STANDALONE_LETTERS.items():
        fixed_test = fixed_test.replace(letter, expanded_form)

    return fixed_test


tqdm.pandas()

# df["content"] = df["content"].progress_apply(strip_url)
# df["content"] = df["content"].progress_apply(dot)
# df["content"] = df["content"].progress_apply(utils.fix_conjoined_words)

df["content"] = df["content"].progress_apply(fix_standalone_letters)

df.to_csv("../dataset/scraped_merged_clean_v3_edited.csv")
