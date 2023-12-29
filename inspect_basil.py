import pandas as pd

BASIL_PATH = "dataset/BASIL/articles_annotations.json"


basil_df = pd.read_json(BASIL_PATH)

print(basil_df.iloc[0])
