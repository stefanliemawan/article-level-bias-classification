import os
import platform
import random

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-cased"
# MODEL_NAME = "mediabiasgroup/magpie-babe-ft"

DATASET_VERSION = "v4"

print(f"MODEL: {MODEL_NAME}")
print(f"dataset {DATASET_VERSION}")

df = pd.read_csv(
    "../dataset/scraped_clean_v4.csv",
    index_col=0,
)

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")


def get_embeddings(sentence):
    tokens = tokeniser.encode(
        sentence, add_special_tokens=False, return_tensors="pt"
    ).to("mps")
    output = model(tokens)
    last_hidden_state = output.last_hidden_state

    return last_hidden_state


def calculate_similarity_score(title_embeddings, sentence_embeddings):
    title_embeddings = title_embeddings.squeeze(0)
    sentence_embeddings = sentence_embeddings.squeeze(0)

    title_embeddings = F.normalize(title_embeddings, p=2, dim=-1)
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)

    title_embeddings = title_embeddings.cpu().detach().numpy()
    sentence_embeddings = sentence_embeddings.cpu().detach().numpy()

    similarity_matrix = cosine_similarity(title_embeddings, sentence_embeddings)
    similarity_score = similarity_matrix[0][0]

    return similarity_score


def rank_content(row):
    title = row["title"]
    content = row["content"]

    title_embeddings = get_embeddings(title)

    sentences = content.split(". ")

    sentences_score = []
    for sentence in sentences:
        # print(sentence)
        if len(sentence) == 1:
            print(sentence)

        try:
            if len(sentence) > 512:
                sentence_embeddings = get_embeddings(sentence[:512])
            elif len(sentence) != 0:
                sentence_embeddings = get_embeddings(sentence)

            sentence_score = calculate_similarity_score(
                title_embeddings, sentence_embeddings
            )
        except Exception as exception:
            print(len(sentence), exception)
            sentence_score = 0.0

        sentences_score.append((sentence, sentence_score))

    sorted_sentences_score = sorted(sentences_score, key=lambda x: x[1], reverse=True)

    ranked_content = ". ".join([sentence[0] for sentence in sorted_sentences_score])

    # r = random.randint(0, 10)
    # if r > 9:
    #     print()
    #     print(row["content"])
    #     print("!@#%^&*()")
    #     print(ranked_content)
    #     print()

    return ranked_content


tqdm.pandas()

df["content"] = df.progress_apply(rank_content, axis=1)


df.to_csv("../dataset/scraped_clean_v4_ranked.csv")

# give or take 1 hour?

# do this within tokenise_dataset instead?
