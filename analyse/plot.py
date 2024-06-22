import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("../cleaned_dataset/scraped_clean_v2.csv", index_col=0)
df = df.head(50)

cnt_pro = df["content"].value_counts()

plt.figure(figsize=(12, 4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel("Number of Occurrences", fontsize=12)
plt.xlabel("content", fontsize=12)
plt.xticks(rotation=90)
plt.show()

# does not work darling
