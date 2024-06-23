import json
import re

ADDED_WORDS = ["ughest", "retweet"]

URL_REGEX = r"http(s)?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
CAPITAL_LETTER_AFTER_PERIOD_REGEX = r"\.(?=[A-Z])"


with open("word_fix.json", "r") as f:
    WORD_FIX_DICT = json.load(f)


with open("phrase_noise.json", "r") as f:
    PHRASE_NOISE_DICT = json.load(f)


def load_word_list():
    # with open("../word_list/words_alpha.txt") as word_file:
    with open("../word_list/words.txt") as word_file:
        word_list = set(word_file.read().split())

    return word_list


# word_list = load_word_list()
# print(len(word_list))


def delete_noise(content):
    content = content.replace("\n", "")

    # sentences = content.split(". ")
    sentences = content.split(".")

    noisy_texts = [
        "Link Copied",
        "About this rating",
        "Forgot Your Password?New to?SubscribePrint subscriber?Activateyour online access",
        "This video can not be played",
        "Readmore",
        "IMAGE:",
        "Thank you for relying on us to provide the journalism you can trust. Please consider supportingNJ.comwith a subscription.",
        "We're hiring!Please take a look at the new openings in our newsroom. See jobsPlease",
        "Add Changing America",
    ]

    noisy_footers = [
        "Contact reporter",
        "Newsweek is committed to challenging conventional wisdom and finding connections in the search for common ground",
        "— InsideJeffrey Epstein’s",
        "From the Archive:",
        "Dr. Michael Brown (www.askdrbrown.org) is the host of the nationally syndicatedLine of Fireradio program",
        "This <a target",
        "Production:Genevieve Montinar",
        "We hope you’ll join us (click to subscribe)",
        "Principled journalism that gets to the roots of the crises we face is more important today than ever.",
        "Want to see more stories like this? Sign up forThe Roundup,",
        "This story was originally publishedAugust",
        "Our journalism needs your support.",
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
        for noisy_footer in noisy_footers:
            if noisy_footer in sentence:
                sentences = sentences[:index]
                content = ".".join(sentences)
                content = content + "."

                return content

        if (
            "Already a subscriber?" in sentence
            or "Contact Us|Privacy Policy© 2024 Right Wing Watch, a project of People For the American Way."
            in sentence
        ):
            sentences.pop(index)
        if re.match(r"[Cc]opyright.*\d{4}", sentence):
            return ". ".join(sentences[:index])
        if "/Getty Imageshide caption" in sentence:
            pattern = r"\b[A-Z][a-z]+ [A-Z][a-z]+/Getty Imageshide caption "
            sentences[index] = re.sub(pattern, "", sentence)
        if "/APhide caption" in sentence:
            pattern = r"\b[A-Z][a-z]+ [A-Z][a-z]+/APhide caption"
            sentences[index] = re.sub(pattern, "", sentence)

        # elif "/AP" in sentence:
        #     pattern = r"\b[A-Z][a-z]+ [A-Z][a-z]+/AP"
        #     print(sentence)
        #     print(re.sub(pattern, "", sentence))
        #     print()
        #     sentences[index] = re.sub(pattern, "", sentence)

    content = ".".join(sentences)

    return content


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


def replace_words(text, replacement_dict):
    words = text.split()
    replaced_words = [replacement_dict.get(word, word) for word in words]
    return " ".join(replaced_words)


def fix_words_by_dict(content):
    words = content.split()
    # words = re.findall(r"(\w+)", content)

    replaced_words = [WORD_FIX_DICT.get(word, word) for word in words]
    fixed_content = " ".join(replaced_words)

    return fixed_content


def remove_noise_phrases(content):
    fixed_content = content
    for word, fixed_word in PHRASE_NOISE_DICT.items():
        fixed_content = fixed_content.replace(word, fixed_word)

    return fixed_content
