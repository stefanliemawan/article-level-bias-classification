import pandas as pd

NLPCSS_PATH = "dataset/NLPCSS-20/released_data.json"

nlpcss_df = pd.read_json(NLPCSS_PATH)
print(nlpcss_df.head())

print(nlpcss_df["adfontes_fair"].value_counts())
print(nlpcss_df["adfontes_political"].value_counts())
print(nlpcss_df["allsides_bias"].value_counts())
