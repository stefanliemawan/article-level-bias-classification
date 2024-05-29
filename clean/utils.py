import re

ADDED_WORDS = ["ughest", "retweet"]

URL_REGEX = r"http(s)?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
CAPITAL_LETTER_AFTER_PERIOD_REGEX = r"\.(?=[A-Z])"

WORD_FIX_DICT = {
    " isstill": " is still",
    " arestill": " are still",
    " hesaid": " he said",
    " shesaid": " she said",
    " saidthe ": " said the",
    " asingle ": " a single ",
    " astatement": " a statement",
    " whenshe ": " when she ",
    " issomething ": " is something ",
    " andsocial ": "and social ",
    "Thestate": "The state",
    " itoriginatesfrom": " it originates from",
    " theNew": " the New",
    " theWashington": " the Washington",
    " theNational": " the National",
    "PresidentDonald": "President Donald",
    "PresidentJoe": "President Joe",
    "accordingto": "according to",
    "AdvertisementThe": "The",
    "InsideClimate": "Inside Climate",
    "ByEmily TannenbaumByAnna BaderByJenny McCoy": "",
    "Find your bookmarks in yourIndependent Premiumsection, under my profile": "",
    "andGoogle": "and Google",
    "By submitting your email, you agree to thePrivacy PolicyandTerms of Useand to receive email correspondence from us. You may opt out at any time. ": "",
    "Get a brief on the top business stories of the week, plus CEO interviews, market updates, tech and money news that matters to you. We've added you to our mailing list. By clicking subscribe, you agree to the Fox NewsPrivacy PolicyandTerms of Use, and agree to receive content and promotional communications from Fox News. You understand that you can opt-out at any time.": "",
    "US launches more strikes against Houthis in YemenTaylor Swift makes history at the Grammy AwardsChina gives detained Australian suspended death sentenceHow a Pakistani woman is making history this election. VideoHow a Pakistani woman is making history this electionChinese ship's port call fans India tensionsGrammy Awards red carpet and ceremony in pictures": "",
}


def load_word_list():
    with open("../word_list/words_alpha.txt") as word_file:
        word_list = set(word_file.read().split())

    return word_list


word_list = load_word_list()
print(len(word_list))


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


def fix_words_by_dict(content):
    fixed_content = content
    for word, fixed_word in WORD_FIX_DICT.items():
        fixed_content = fixed_content.replace(word, fixed_word)

    return fixed_content
