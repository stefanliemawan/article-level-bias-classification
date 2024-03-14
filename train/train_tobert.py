"""
Given that Transformers’ edge over recurrent networks is
their ability to effectively capture long distance relationships
between words in a sequence [1], we experiment with replacing the LSTM recurrent layer in favor of a small Transformer
model (2 layers of transformer building block containing selfattention, fully connected, etc.). To investigate if preserving
the information about the input sequence order is important,
we also build a variant of ToBERT which learns positional
embeddings at the segment-level representations (but
"""

# do transformers... model() on sequence, then input for finetuning?
