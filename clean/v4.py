import re

import pandas as pd
import utils
from tqdm import tqdm

ABBR_REGEX = r"([A-Z])\. ?([A-Z])\."
NAME_ABBR_REGEX = r"([A-Z])\. ([A-Z][a-z]+)"

ABBR_DICT = {
    "Ph. D": "PhD",
    "Gov.": "Governor",
    "No.": "Number",
    " v.": " vs",
    " vs.": " vs",
    "St. ": "Street",
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

df = pd.read_csv(
    "../dataset/scraped_merged_clean_v3.csv",
    index_col=0,
)

urls_removed = []


def strip_abbr(content):
    stripped_abbr = re.sub(ABBR_REGEX, r"\1\2", content)
    stripped_abbr = re.sub(NAME_ABBR_REGEX, r"\1 \2", stripped_abbr)

    for abbr, expanded_form in ABBR_DICT.items():
        stripped_abbr = stripped_abbr.replace(abbr, expanded_form)

    return stripped_abbr


tqdm.pandas()

df["title"] = df["title"].progress_apply(strip_abbr)
df["content"] = df["content"].progress_apply(strip_abbr)

df.to_csv("../dataset/scraped_merged_clean_v4.csv")
