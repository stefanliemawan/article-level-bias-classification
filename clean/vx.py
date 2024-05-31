import pandas as pd
import utils
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv("../dataset/scraped_merged_clean_v1.csv", index_col=0)
df.dropna(subset=["content"], inplace=True)

df["content"] = df["content"].progress_apply(utils.delete_noise)
df["content"] = df["content"].progress_apply(utils.delete_phrases)
df["content"] = df["content"].progress_apply(utils.strip_url)
df["content"] = df["content"].progress_apply(utils.dot)
df["content"] = df["content"].progress_apply(utils.fix_words_by_dict)
df["content"] = df["content"].progress_apply(utils.fix_words_by_gpt_dict)

df.dropna(subset=["content"], inplace=True)

df.to_csv("../dataset/scraped_merged_clean_vx.csv")


# x = " their President Trump"
# x = utils.fix_more_words(x)
# print(x)
