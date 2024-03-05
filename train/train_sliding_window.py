import functions
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
)

SEED = 42
CLASS_RANGES = [(0, 29.32), (29.33, 43.98), (43.98, 58.67)]

MODEL_NAME = "distilbert-base-uncased"

WINDOW_SIZE = 128
STRIDE = 128

train_df = pd.read_csv("dataset/train.csv", index_col=0)
test_df = pd.read_csv("dataset/test.csv", index_col=0)
valid_df = pd.read_csv("dataset/valid.csv", index_col=0)

train_df["content"] = train_df.apply(functions.preprocess_content, axis=1)
test_df["content"] = test_df.apply(functions.preprocess_content, axis=1)
valid_df["content"] = valid_df.apply(functions.preprocess_content, axis=1)


tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenise_dataset(x):
    input_ids = tokeniser.encode(x, return_tensors="pt")[0]
    if len(input_ids) < WINDOW_SIZE:
        input_ids = torch_f.pad(
            input=input_ids,
            pad=(0, WINDOW_SIZE - len(input_ids)),
            mode="constant",
            value=tokeniser.pad_token_id,
        )

    return input_ids.unfold(0, WINDOW_SIZE, STRIDE)


test = tokenise_dataset(train_df["content"].iloc[123])
print(test)
print(test.shape)


# train_labels = train_dataset["labels"]
# class_weights = compute_class_weight(
#     class_weight="balanced", classes=np.unique(train_labels), y=train_labels
# )


# print(tokenised_dataset["train"]["features"][0])

# functions.print_class_distribution(tokenised_dataset)

# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME, num_labels=len(CLASS_RANGES)
# )


# print(tokenised_dataset["train"]["input_ids"])

# def chunk_text_to_window_size_and_predict_proba(input_ids, attention_mask, total_len):
#     """
#     This function splits the given input text into chunks of a specified window length,
#     applies transformer model to each chunk and computes probabilities of each class for each chunk.
#     The computed probabilities are then appended to a list.

#     Args:
#         input_ids (List[int]): List of token ids representing the input text.
#         attention_mask (List[int]): List of attention masks corresponding to input_ids.
#         total_len (int): Total length of the input_ids.

#     Returns:
#         proba_list (List[torch.Tensor]): List of probability tensors for each chunk.
#     """
#     proba_list = []

#     start = 0
#     window_length = 510

#     loop = True

#     while loop:
#         end = start + window_length
#         # If the end index exceeds total length, set the flag to False and adjust the end index
#         if end >= total_len:
#             loop = False
#             end = total_len

#         # 1 => Define the text chunk
#         input_ids_chunk = input_ids[start:end]
#         attention_mask_chunk = attention_mask[start:end]

#         # 2 => Append [CLS] and [SEP]
#         input_ids_chunk = [101] + input_ids_chunk + [102]
#         attention_mask_chunk = [1] + attention_mask_chunk + [1]

#         # 3 Convert regular python list to Pytorch Tensor
#         input_dict = {
#             "input_ids": torch.Tensor([input_ids_chunk]).long(),
#             "attention_mask": torch.Tensor([attention_mask_chunk]).int(),
#         }

#         outputs = model(**input_dict)

#         probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)
#         proba_list.append(probabilities)


# proba_list = chunk_text_to_window_size_and_predict_proba(
#     input_ids, attention_mask, total_len
# )


# def get_mean_from_proba(proba_list):
#     """
#     This function computes the mean probabilities of class predictions over all the chunks.

#     Args:
#         proba_list (List[torch.Tensor]): List of probability tensors for each chunk.

#     Returns:
#         mean (torch.Tensor): Mean of the probabilities across all chunks.
#     """

#     # Ensures that gradients are not computed, saving memory
#     with torch.no_grad():
#         # Stack the list of tensors into a single tensor
#         stacks = torch.stack(proba_list)

#         # Resize the tensor to match the dimensions needed for mean computation
#         stacks = stacks.resize(stacks.shape[0], stacks.shape[2])

#         # Compute the mean along the zeroth dimension (i.e., the chunk dimension)
#         mean = stacks.mean(dim=0)

#     return mean


# mean = get_mean_from_proba(proba_list)
# # tensor([0.0767, 0.1188, 0.8045])

# torch.argmax(mean).item()
