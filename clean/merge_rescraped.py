import pandas as pd

v1_df = pd.read_csv("../dataset/scraped_clean_v1.csv", index_col=0)
rescraped_df = pd.read_csv("../dataset/rescraped_2_edited.csv", index_col=0)

merged_df = pd.concat([v1_df, rescraped_df], ignore_index=True)
merged_df.reset_index(drop=True, inplace=True)

merged_df.to_csv("../dataset/scraped_clean_v1_new.csv")
