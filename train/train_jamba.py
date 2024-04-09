import platform

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
)

JAMBA = "ai21labs/Jamba-v0.1"

tokenizer = AutoTokenizer.from_pretrained(JAMBA)
model = AutoModelForCausalLM.from_pretrained(JAMBA, trust_remote_code=True)
print(model)
# model = AutoModelForSequenceClassification.from_pretrained(
#     JAMBA, trust_remote_code=True
# )

if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")


input_ids = tokenizer("In the recent Super Bowl LVIII,", return_tensors="pt").to(
    model.device
)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=216)
print(outputs)

print(tokenizer.batch_decode(outputs))
