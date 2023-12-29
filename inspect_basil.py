import pandas as pd
import json

BASIL_PATH = "dataset/BASIL/articles_annotations.json"
BASIL_ARTICLE_1_PATH = "dataset/BASIL/article_annotation_1.json"

basil_df = pd.read_json(BASIL_PATH)

print(basil_df.iloc[0])

with open(BASIL_PATH) as f:
    basil = json.load(f)

# with open(BASIL_ARTICLE_1_PATH) as f:
#     basil_article = json.load(f)

basil_article = basil[0]

print(json.dumps(basil_article, indent=4))

article = basil_article["body-paragraphs"]
article = [" ".join(ar) for ar in article]
article = " ".join(article)

print(article)
print()

phrase_level_annotations = basil_article["phrase-level-annotations"]

annotation = phrase_level_annotations[-1]
start = annotation["start"]
end = annotation["end"]
bias = annotation["bias"]
txt = annotation["txt"]

word_level_article = article.split()
# print(word_level_article)
bias_phrase = " ".join(word_level_article[start : end + 1])
# print(bias_phrase)
print(txt)
print(bias_phrase)
# print(word_level_article.index(txt))
print()
