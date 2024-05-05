import platform

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.functions as functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# train_df = pd.read_csv("../dataset/v3/train.csv", index_col=0)
# test_df = pd.read_csv("../dataset/v3/test.csv", index_col=0)
# valid_df = pd.read_csv("../dataset/v3/valid.csv", index_col=0)


# train_df, test_df, valid_df = functions.generate_title_content_features(
#     train_df, test_df, valid_df
# )

text = """
Washtenaw County's new prosecutor is decriminalizing ""consensual sex work,"" his office announced Thursday. Prosecutor Eli Savit announced in a press release that his office is ending the practice of prosecuting sex workers who engage in consensual sex.""The Washtenaw County Prosecutor’s Office is well aware that sex work carries an increased risk for violence, human trafficking, and coercion. Data and experience, however, have shown that criminal izing sex work does little to alleviate those harms,"" Savit said in the statement. ""Indeed the criminalization of sex work actually increases the risk of sex work adjacent harm. Accordingly, the Washtenaw County Prosecutor’s Office will henceforth decline to bring charges related to consensual sex work per se. The Prosecutor’s Office, however, will continue to charge sex work-adjacent crime — including human trafficking, violence, and offenses involving children — that directly harm county residents.""Efforts to reach Savit for further comment Thursday were not successful. Elected in November, Savit said with the exception of Nevada, sex work is generally criminal i zed in the United States but that ""America’s prohibitionist stance on sex work is increasingly out of step with international norms. Consensual sex work is legal — at least in some form — in nearly 100 countries across the globe.""Savit added the laws against ""consensual"" sex work conflict with the U. S. constitution as adults have a right to ""engage in private conduct"" such as sex.​Savit added that ""Laws banning consensual sex between adults thus generally violate the United States ConstitutionIt is only when sex is exchanged for money that such activity may be banned. But even once money enters the equation, sex is not consistently criminalized.""In Oakland County, Chief Assistant Prosecutor David Williams told The News Thursday that while his office's main focus is human trafficking, Oakland County Prosecutor Karen McDonald will be ""rolling out"" policy changes in prosecutions.""I applaud what (Savit) is doing and he certainly is headed in the right direction,"" said Williams. ""There's no question. There absolutely will be changes (in Oakland County).""Last week, Savit made headlines when he announced his office would end cash bail for defendants, saying the practice punishes the poor.""America’s system of cash bail is unfair, inequitable, and imposes severe harm on communities,"" Savit said last week. ""Cash bail is a system under which a defendant who has been accused of a crime is required to post money in order to secure release from jail pending trial. Importantly, cash bail forces defendants to pay for their release before they have been convicted. In function, then, cash bail imposes pre-conviction punishment on criminal defendants who cannot afford to pay.""Savit is a faculty member at the University of Michigan Law School. A former school teacher, he also is a former law clerk to the late U. S. Supreme Court Justice Ruth Bader Ginsburg. In Michigan, the criminal penalty for prostitution ranges from a misdemeanor, which is punishable by up to 93 days in jail and a $500 fine, to a felony, punishable by up to two years in prison and/or a $2,000 fine.
"""

text = text.split(". ")
print(text)

# one graph = one article
# nodes = sentences
# do magpie on each sentence? returns 0 or 1, do edges based on this?

# FROM CHATGPT!

# Data Representation:
# Each article is represented as a separate graph.
# Each sentence within the article is represented as a node in the corresponding graph.
# You'll need to tokenize your text into sentences and represent them as nodes in the graph. The node features may include word embeddings, positional embeddings, or other linguistic features.

# Graph Construction:
# Build an adjacency matrix for each graph (article) based on the relationships between sentences. The adjacency matrix encodes the connectivity between nodes (sentences) in the graph.
# Determine the relationships between sentences, which could be based on sequential order, syntactic dependencies, semantic similarity, or other criteria.

# Node Features:
# Assign features to each node (sentence) based on its content. This could involve using pre-trained word embeddings (e.g., Word2Vec, GloVe, BERT embeddings) or other linguistic features.
# Optionally, incorporate additional features such as sentence length, position within the article, or metadata associated with the sentence.

# Graph Convolutional Network (GCN):
# Implement a GCN architecture to process the graph data.
# Each graph (article) is passed through the GCN model, and the node representations are updated based on information from neighboring nodes.
# The GCN layers aggregate information from neighboring nodes and update the node representations iteratively through message passing.

# Task-Specific Layers:
# After processing the graph data through the GCN layers, add task-specific layers (e.g., fully connected layers, attention mechanisms) to perform the desired downstream task (e.g., classification, summarization).

# Training and Evaluation:
# Train the model using labeled data for the target task (e.g., article classification).
# Evaluate the model's performance on a separate validation set to monitor generalization.
# Fine-tune hyperparameters and model architecture based on validation performance.
# Finally, evaluate the trained model on a held-out test set to assess its performance in real-world scenarios.
