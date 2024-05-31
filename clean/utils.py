import json
import re

ADDED_WORDS = ["ughest", "retweet"]

URL_REGEX = r"http(s)?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
CAPITAL_LETTER_AFTER_PERIOD_REGEX = r"\.(?=[A-Z])"


with open("word_fix_dict_gpt.json", "r") as f:
    WORD_FIX_DICT_GPT = json.load(f)

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
    "Get a brief on the top business stories of the week, plus CEO interviews, market updates, tech and money news that matters to you. We've added you to our mailing list. By clicking subscribe, you agree to the Fox NewsPrivacy PolicyandTerms of Use, and agree to receive content and promotional communications from Fox News. You understand that you can opt-out at any time.": "",
    "US launches more strikes against Houthis in YemenTaylor Swift makes history at the Grammy AwardsChina gives detained Australian suspended death sentenceHow a Pakistani woman is making history this election. VideoHow a Pakistani woman is making history this electionChinese ship's port call fans India tensionsGrammy Awards red carpet and ceremony in pictures": "",
    "reportedthat": "reported that",
    "ByChris MurphyByNick BiltonByErin Vanderhoof": "",
    "ByChris MurphyByChris MurphyByDavid Canfield": "",
    "herstatement": "her statement",
    "ollow Stephen Robinson onTwitter. Want to just donate once?": "",
    '''This <a target=""_blank"" href="""">article</a> first appeared on <a target=""_blank"" href="""">The Real News Network</a> and is republished here under a Creative Commons license.<img src="";ssl=1"" style=""width:1em;height:1em;margin-left:10px;""><img id=""republication-tracker-tool-source"" src="";ga4=G-7LYS8R7V51"" style=""width:1px;height:1px;"">''': "",
    "We inhale a credit card's worth of microplastics each week. VideoWe inhale a credit card's worth of microplastics each weekOrlando, Beijing and... the village of Stewartby?What are routes out of this 'dangerous moment' in Middle East?Did bodybuilding bring on my early perimenopause?I almost died up a mountain scattering dad's ashesThey fled as lava spilled into town - and they may never returnHow one rhino became a global celebrityThe literary scandal that rocked US high societyWhen employers gut middle managers, everyone hurts© 2024 BBC. The BBC is not responsible for the content of external sites. Read about our approach to external linking.": "",
    "Israel’s War on GazaRyan Grim, Nick TurseThe U. S. risks complicity with Israeli atrocities, experts say. Schuyler MitchellProposed laws to curtail the use of PVC plastics have failed amid heavy lobbying. Prem ThakkerA new report shows Norfolk Southern spent $2.3 million to lobby the government in 2023.": "",
    "Or with:By signing up you agree to ourTerms of ServiceandPrivacy Policy": "",
    "Become a business insider with the latest news. ": "",
    "At Vox, we believe that clarity is power, and that power shouldn’t only be available to those who can afford to pay. That’s why we keep our work free. Millions rely on Vox’s clear, high-quality journalism to understand the forces shaping today’s world. Support our mission and help keep Vox free for all by making a financial contribution to Vox today.$5/month$10/month$25/month$50/monthWe accept credit card, Apple Pay, and Google Pay. You can also contribute via": "",
    "Copyright © 2024 The Washington Times, LLC. Click                            here for reprint permission. Click to Read More and View CommentsClick to Hide": "",
    " ByP. Claire DodsonBySara DelgadoBySara Delgado": "",
    "andthe": "",
    "ByVeronica CristinoByHannah Coates": "",
    "This content can also be viewed on the site it originates from.": "",
    "Help expose the Far-Right's extreme and intolerant agenda. Browse our archive of posts by topicBrowse our archive of posts on key right-wing groupsBrowse our archive of posts on key right-wing figuresRead in-depth reports on key people, organizations and issues on the Right": "",
    "theAmerican": "the American",
    "Alex Wong/Getty Images": "",
    "Kevin Winter/Getty Images": "",
    "Drew Angerer/Getty Images": "",
    "Joe Raedle/Getty Images": "",
    "Amy Beth Bennett / South Florida Sun Sentinel": "",
    "Ursula Seemann / Sun SentinelWilfredo Lee / APA. ENRIQUE VALENTIN / Sun SentinelJOHN BAZEMORE / APAtlanta ": "",
    "Amy Beth Bennett / South Florida Sun Sentinel": "",
    "The City received a $400,000 prize from the federal government as one of the initial winners of a multi-phase competitionThe biennial census is federally mandated and goes a long way in funding support services in The CityDoty was acquitted last month of assaulting a former San Francisco fire commissioner": "",
    "theAssociated": "the Associated",
    "theUnited": "the United",
    "announcedthat": "announced that",
    "Rahmat Gul/APhide caption": ". ",
    "Get the latest updates from the 2024 campaign trail, exclusive interviews and more Fox News politics content.": "",
    "SubscribedYou've successfully subscribed to this newsletter!": "",
    "AdvertisementSupported by": "",
    "Michelle GoldbergByMichelle GoldbergOpinion Columnist": "",
    "Paul KrugmanByPaul KrugmanOpinion Columnist": "",
    "BuzzFeed News ReporterReporting From": "",
    "TheNew": "The New",
    "By submitting your email, you agree to thePrivacy PolicyandTerms of Useand to receive email correspondence from us. You may opt out at any time.": "",
    "By submitting your email, you agree to ourTermsandPrivacy Noticeand to receive email correspondence from us.": "",
    "Andrew WilliamsMatt BurgessScharon Harding, Ars TechnicaMedea Giordano": "",
    "theCOVID": "the COVID",
    "Maïa Booker atmaia.booker@time.comand Wilder Davies atwilder.davies@time.com.": "",
    "Write toHillary Leungathillary.leung@time.com": "",
    "Contact us atletters@time.com": "",
    "Bidensaid": "Biden said",
    "thecoronaviruspandemic": "the corona virus pandemic",
    "thelargest": "the largest",
    "Congressmember": "Congress member",
    "clerkhere": "clerk here",
    "aletterto": "a letter to",
    "studyfrom": "study from",
    "tweetedthat": "tweeted that",
    "Asubscriptionhelps you access more of the local stories that keep you connected to the community. Subscribe to our Daily Headlines newsletter.": "",
    "reportedon": "reported on",
    "sdecision": "s decision",
    "againstthe": "against the",
    "thenumber": "the number",
    "nomineeJoe": "nominee Joe",
    "2021We": "2021 We",
    "positivefor": "positive for",
    "includingthe": "including the",
    "21stcentury": "21st century",
    "2021Trump": "2021 Trump",
    "toWashington": "to Washington",
    "andWashington": "and Washington",
    " toan ": " to an ",
    "Subscribe to our free News Alerts newsletterWant more of our free, weekly newsletters in your inbox?": "",
    "Subscribe toBlazeTVtoday with our BEST DEAL EVER for $30 off with promo code GLENN.": "",
    "theEconomic": "the Economic",
    "Join thought-provoking conversations, follow other Independent readers and see their replies": "Join thought-provoking conversations, follow other Independent readers and see their replies",
    "likelyto": "likelyto",
    "toConsortium": "to Consortium",
    "saidon ": "said on ,",
    "lawsuitfiled": "lawsuit filed",
    "theDemocratic": "the Democratic",
    "thepodcast": "the podcast",
    "TheAssociated": "The Associated",
}


def load_word_list():
    with open("../word_list/words_alpha.txt") as word_file:
        word_list = set(word_file.read().split())

    return word_list


word_list = load_word_list()
print(len(word_list))


def delete_phrases(content):
    phrases = [
        "Dr. Michael Brown (www.askdrbrown.org) is the host of the nationally syndicatedLine of Fireradio program.",
        "— InsideJeffrey Epstein’s Decades-Long RelationshipWith Leslie Wexner—",
        "This <a target",
        "Production:Genevieve Montinar",
        "We hope you’ll join us (click to subscribe)",
        "Principled journalism that gets to the roots of the crises we face is more important today than ever.",
        "Want to see more stories like this? Sign up forThe Roundup,",
        "This story was originally publishedAugust",
        "Our journalism needs your support.",
        "ExploreExplore",
    ]

    for phrase in phrases:
        phrase_position = content.find(phrase)

        if phrase_position != -1:
            content = content[:phrase_position]

    return content


def delete_noise(content):
    content = content.replace("\n", "")

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
        # if "/APhide caption" in sentence:
        #     pattern = r"\b[A-Z][a-z]+ [A-Z][a-z]+/APhide caption"
        #     print(sentence)
        #     print(re.sub(pattern, "", sentence))
        #     print()
        #     sentences[index] = re.sub(pattern, "", sentence)
        # elif "/AP" in sentence:
        #     pattern = r"\b[A-Z][a-z]+ [A-Z][a-z]+/AP"
        #     print(sentence)
        #     print(re.sub(pattern, "", sentence))
        #     print()
        #     sentences[index] = re.sub(pattern, "", sentence)

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


def fix_words_by_gpt_dict(content):
    fixed_content = content
    for word, fixed_word in WORD_FIX_DICT_GPT.items():
        fixed_content = fixed_content.replace(word, fixed_word)

    return fixed_content
