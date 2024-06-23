import re

import pandas as pd
import utils
from tqdm import tqdm

ABBR_REGEX = r"([A-Z])\. ?([A-Z])\."
NAME_ABBR_REGEX = r"([A-Z])\. ([A-Z][a-z]+)"

ABBR_DICT = {
    "United State": "US",  # maybe this should be the other way around?
    "United States": "US",
    "United States of America": "US",
    "United Nations": "UN",
    "Ph. D": "PhD",
    "Gov.": "Governor",
    "Sen.": "Senator",
    "Rep.": "Representative",
    "No.": "Number",
    "b log": "blog",
    "b logs": "blog",
    " v.": " vs",
    " vs.": " vs",
    "Jan.": "January",
    "Feb.": "February",
    "Mar.": "March",
    "Apr.": "April",
    "May.": "May",
    "Jun.": "June",
    "Jul.": "July",
    "Aug.": "August",
    "Sep.": "September",
    "Oct.": "October",
    "Nov.": "November",
    "Dec.": "December",
}

CONTRACTION_DICT = {
    "won't": "will not",
    "n't": " not",
    "'m": " am",
    "'s": " is",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "won’t": "will not",
    "n’t": " not",
    "’m": " am",
    "’s": " is",
    "’re": " are",
    "’ve": " have",
    "’ll": " will",
}

df = pd.read_csv(
    "../dataset/scraped_clean_v3.csv",
    index_col=0,
)

urls_removed = []


def strip_abbr(content):
    stripped_abbr = content

    stripped_abbr = re.sub(ABBR_REGEX, r"\1\2", stripped_abbr)
    stripped_abbr = re.sub(NAME_ABBR_REGEX, r"\1 \2", stripped_abbr)

    for abbr, expanded_form in ABBR_DICT.items():
        stripped_abbr = stripped_abbr.replace(abbr, expanded_form)

    for contraction, expanded_form in CONTRACTION_DICT.items():
        stripped_abbr = stripped_abbr.replace(contraction, expanded_form)

    return stripped_abbr


tqdm.pandas()

df["title"] = df["title"].progress_apply(strip_abbr)
df["content"] = df["content"].progress_apply(strip_abbr)

df.to_csv("../dataset/scraped_clean_v4.csv")


# test_txt = "hasn’t explained why,"

# print(strip_abbr(test_txt))
