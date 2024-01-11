import pandas as pd

bat_adfontes_df = pd.read_csv("dataset/BAT/ad_fontes/articles_dates.csv", index_col=0)

bat_adfontes_df["outlet"] = bat_adfontes_df["article_url"].str.split("/").str[2]
outlets_count = bat_adfontes_df["outlet"].value_counts()
outlets_list = outlets_count.index.tolist()

bat_adfontes_df_sorted = bat_adfontes_df.sort_values(
    by="outlet",
    key=lambda column: column.map(lambda x: outlets_list.index(x)),
)

print(bat_adfontes_df_sorted.head())

bat_adfontes_df_sorted.to_csv("articles_sorted_by_outlet_occurences.csv", index=False)
