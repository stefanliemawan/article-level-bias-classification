import json
import os

BASIL_PATH = "dataset/BASIL"

articles = []

for root, dirs, files in os.walk("dataset/basil/articles"):
    path = root.split(os.sep)
    print((len(path) - 1) * "---", os.path.basename(root))
    for file in sorted(files):
        print(len(path) * "---", file)
        full_path = "/".join(path)
        with open(f"{full_path}/{file}") as f:
            articles.append(json.load(f))


annotations = []

for root, dirs, files in os.walk("dataset/basil/annotations"):
    for dirname in sorted(dirs):
        print(dirname)
    path = root.split(os.sep)
    print((len(path) - 1) * "---", os.path.basename(root))
    for file in sorted(files):
        print(len(path) * "---", file)
        full_path = "/".join(path)
        with open(f"{full_path}/{file}") as f:
            annotations.append(json.load(f))


basil = []

for article, annotation in zip(articles, annotations):
    basil.append({**article, **annotation})

with open("articles_annotations.json", "w") as f:
    json.dump(basil, f)
