import re
import pandas as pd


def merge_df():
    df = pd.read_csv("cleaned_dataset/scrape/scraped_uniform_test_4.csv", index_col=0)

    scraped_7_df = pd.read_csv("scrape/scraped_7.csv", index_col=0)

    df[df.isnull()] = scraped_7_df

    df.to_csv("cleaned_dataset/scraped_merged.csv")


def clean_df():
    df = pd.read_csv("cleaned_dataset/scraped_merged.csv", index_col=0)
    df.dropna(subset=["content"], inplace=True)

    # invalid content but notna
    df.drop(df[df["outlet"].str.startswith("hillreporter")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("quillette")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("www.newsday")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("www.oann")].index, inplace=True)
    df.drop(df[df["outlet"].str.startswith("www.financialbuzz")].index, inplace=True)
    df.drop(
        df[df["outlet"].str.startswith("www.libertynation")].index, inplace=True
    )  # theres a solution for this one, update if rescraped

    df.reset_index(inplace=True)
    df.to_csv("cleaned_dataset/scraped_merged_clean_v1.csv")


def delete_noise(row):
    content = row["content"].replace("\n", "")

    sentences = content.split(".")

    noisy_texts = [
        "Link Copied",
        "About this rating",
        "Forgot Your Password?New to?SubscribePrint subscriber?Activateyour online access",
        "This video can not be played",
    ]

    for index, sentence in enumerate(sentences.copy()):
        for noisy_text in noisy_texts:
            if noisy_text in sentence:
                sentences[index] = sentence[len(noisy_text) :]
                if len(sentences[index]) == 0:
                    return None
        if "Already a subscriber?" in sentence:
            sentences.pop(index)
        if re.match(r"[Cc]opyright.*\d{4}", sentence):
            return ". ".join(sentences[:index])

    content = ". ".join(sentences)

    return content


def clean_content():
    df = pd.read_csv("cleaned_dataset/scraped_merged_clean_v1.csv", index_col=0)

    df["content"] = df.apply(delete_noise, axis=1)

    df.dropna(subset=["content"], inplace=True)

    df.to_csv("cleaned_dataset/scraped_merged_clean_v2.csv")


# merge_df()
clean_df()
clean_content()


# MORE TEXT TO DELETE:

# AT SENTENCE BEGINNING
# "NewsNews|Dec 30, 2020csackariason@aspentimes. comShow CaptionsHide Captions" - email is different, use regex here
# "Join the 3,900+ MTFP members who believe in the power of independent news. This quality reporting was made possible due in part to your contribution.  Thank you for supporting in-depth journalism in Montana. Sign up to get our reporting sent straight to your inbox every weekday morning.",
