import re

import pandas as pd


def delete_noise(row):
    content = row["content"].replace("\n", "")

    sentences = content.split(". ")

    if sentences[0].startswith(
        "Join the 3,900+ MTFP members who believe in the power of independent news"
    ):
        sentences = sentences[3:]

    noisy_texts = [
        "Link Copied",
        "About this rating",
        "Forgot Your Password?New to?SubscribePrint subscriber?Activateyour online access",
        "This video can not be played",
    ]

    for index, sentence in enumerate(sentences.copy()):
        if sentence.startswith("NewsNews|"):
            sentences[index] = re.sub(
                r"NewsNews\|\w{3} \d{1,2}, \d{4}\S*@\S*\.com(?:Show CaptionsHide Captions)?",
                "",
                sentence,
            )
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


df = pd.read_csv("cleaned_dataset/scraped_merged_clean_v1.csv", index_col=0)

df["content"] = df.apply(delete_noise, axis=1)

df.dropna(subset=["content"], inplace=True)

df.to_csv("cleaned_dataset/scraped_merged_clean_v2.csv")
