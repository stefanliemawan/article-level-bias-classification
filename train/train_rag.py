import os
import platform

import pandas as pd
import torch
import utils.functions as functions
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "facebook/rag-token-base"

print(f"MODEL: {MODEL_NAME}")
print("dataset v3")

train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)


tokeniser = RagTokenizer.from_pretrained(MODEL_NAME)
retriever = RagRetriever.from_pretrained(MODEL_NAME)
generator = RagTokenForGeneration.from_pretrained(MODEL_NAME)


def retrieve_passages(input_text, retriever, tokenizer, num_passages=5):
    inputs = tokenizer([input_text], return_tensors="pt")
    with torch.no_grad():
        passage_ids, relevance_scores = retriever(
            inputs["input_ids"], return_tensors="pt"
        )

    passage_ids = passage_ids[0][:num_passages]
    passages = [retriever.get_doc_text(int(p_id)) for p_id in passage_ids]
    return passages


# Example usage: Retrieve passages for input article
input_article = train_df["content"][0]
passages = retrieve_passages(input_article, retriever, tokeniser)
print(len(passages))
print("Retrieved Passages:")
for passage in passages:
    print(passage)

encoded_inputs = tokeniser(
    input_article, passages, return_tensors="pt", padding=True, truncation=True
)

outputs = generator.generate(
    input_ids=encoded_inputs["input_ids"],
    attention_mask=encoded_inputs["attention_mask"],
    num_beams=4,  # Adjust beam size for generation
    max_length=50,  # Adjust max length of generated text
    early_stopping=True,
)

generated_text = tokeniser.batch_decode(outputs, skip_special_tokens=True)
print("Generated Classification Text:")
print(generated_text)
