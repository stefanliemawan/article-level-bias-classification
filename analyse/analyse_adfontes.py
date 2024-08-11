import matplotlib.pyplot as plt

# Define the ranges and labels for both sources
ranges_website = [(40, 64), (24, 40), (0, 24)]
labels_website = ["Generally Good", "Variety of Factors", "Problematic"]

ranges_paper = [(48, 64), (32, 48), (24, 32), (16, 24), (0, 16)]
labels_paper = [
    "Fact Reporting",
    "Analysis",
    "Opinions",
    "Reliability Issues",
    "False/Misleading",
]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot ranges for the website
for i, (start, end) in enumerate(ranges_website):
    bar = ax.barh(
        f"Website: {labels_website[i]}",
        end - start,
        left=start,
        color="skyblue",
        edgecolor="black",
    )
    # Add descriptive label on the bar
    ax.text(
        start + (end - start) / 2,
        f"Website: {labels_website[i]}",
        labels_website[i],
        va="center",
        ha="center",
        color="black",
        fontsize=10,
        fontweight="bold",
    )

# Plot ranges for the paper
for i, (start, end) in enumerate(ranges_paper):
    bar = ax.barh(
        f"Paper: {labels_paper[i]}",
        end - start,
        left=start,
        color="salmon",
        edgecolor="black",
    )
    # Add descriptive label on the bar
    ax.text(
        start + (end - start) / 2,
        f"Paper: {labels_paper[i]}",
        labels_paper[i],
        va="center",
        ha="center",
        color="black",
        fontsize=10,
        fontweight="bold",
    )

# Labeling
ax.set_xlabel("Reliability Score")
ax.set_title("Comparison of Reliability Score Classifications")

# Adjust legend to correctly display categories
handles = [
    plt.Line2D([0], [0], color="skyblue", lw=6, label="Website Methodology"),
    plt.Line2D([0], [0], color="salmon", lw=6, label="Paper Methodology"),
]
ax.legend(handles=handles, loc="upper right")

plt.tight_layout()
plt.savefig("figures/reliability_score_diff.png")
