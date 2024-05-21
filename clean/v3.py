import re

import pandas as pd
import utils
from tqdm import tqdm

URL_REGEX = r"http(s)?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
CAPITAL_LETTER_AFTER_PERIOD_REGEX = r"\.(?=[A-Z])"
THE_REGEX = r"(?<=\s)the([a-z]+)"
TO_REGEX = r"(?<=\s)to([a-z]+)"
GETTY_REGEX = r"(\.).*via Getty Images(?:hide caption)?"

WORD_FIX_DICT = {
    " s ": "s ",
    " isstill": " is still",
    " hesaid": " he said",
    " arestill": " are still",
    " makesure": " make sure",
    " haveseen": "have seen",
    " insome": " in some",
    " onsocial": " on social",
    " asingle": " a single",
    " weshould": " we should",
    " astatement": " a statement",
    " whenshe": " when she",
    " issomething": " is something",
    " andsocial": "and social",
    " nationalsecurity": "national security",
    " Thestate": " The state",
    " itoriginatesfrom": " it originates from",
    " to xic ": " toxic ",
    " to wn": " town",
    " the rapy": " therapy",
    " theNew": " the New",
    " theWashington": " the Washington",
    " theNational": " the National",
    " PresidentDonald": " President Donald",
    "accordingto": "according to",
    "AdvertisementThe": "The",
    " the ir ": " their ",
    " the y ": " they ",
    " to ld ": " told ",
    " to gether ": " together ",
    " the se ": " these ",
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


def fix_more_words(content):
    fixed_content = content

    fixed_content = re.sub(THE_REGEX, r"the \1", fixed_content)
    fixed_content = re.sub(TO_REGEX, r"to \1", fixed_content)
    fixed_content = re.sub(GETTY_REGEX, r"\1 ", fixed_content)

    for word, fixed_word in WORD_FIX_DICT.items():
        fixed_content = fixed_content.replace(word, fixed_word)

    return fixed_content


tqdm.pandas()

df = df.sample(frac=1).reset_index(drop=True)

# df["content"] = df["content"].progress_apply(strip_url)
# df["content"] = df["content"].progress_apply(dot)
# df["content"] = df["content"].progress_apply(utils.fix_conjoined_words)

df["content"] = df["content"].progress_apply(fix_more_words)

# df.to_csv("../dataset/scraped_merged_clean_v3_new.csv")

# test_txt = "January 6, 2021. (Photo By Tom Williams/CQ-Roll Call, Inc via Getty Images)Eventually Capitol Police escor"

# print(fix_more_words(test_txt))
