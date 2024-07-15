import pandas as pd

v1_df = pd.read_csv("figures/word_not_in_dict_v1.csv")
vx_df = pd.read_csv("figures/word_not_in_dict_vx.csv")

cleaned = v1_df[~v1_df["word"].isin(vx_df["word"])]

print(cleaned)
print(sum(cleaned["count"]))
print(sum(vx_df["count"]))
