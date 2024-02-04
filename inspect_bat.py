import pandas as pd

df = pd.read_csv("dataset/BAT/ad_fontes/articles_dates.csv", index_col=0)

df["outlet"] = df["article_url"].str.split("/").str[2]
outlet_story_count = df["outlet"].value_counts()
outlets_list = outlet_story_count.index.tolist()

df_sorted = df.sort_values(
    by="outlet",
    key=lambda column: column.map(lambda x: outlets_list.index(x)),
)

df_sorted["outlet_story_count"] = df["outlet"].map(outlet_story_count)

print(df_sorted.head())

df_sorted.to_csv("articles_sorted_by_outlet_occurences.csv", index=False)
