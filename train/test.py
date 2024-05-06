from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Check if "US" and "United States" are in the vocabulary
us_token_id = tokenizer.convert_tokens_to_ids("us")
united_states_token_id = tokenizer.convert_tokens_to_ids(
    "united"
)  # Check for the "united" token, as it's more likely to be in the vocabulary
print("US token ID:", us_token_id)
print("United States token ID:", united_states_token_id)
