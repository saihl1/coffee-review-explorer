"""
visualize.py
------------
Generates the charts for the project README, saved as PNGs in output/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

CLEAN_PATH = "data/coffee_ratings_clean.csv"
OUTPUT_DIR = "output"


def plot_top_countries(df: pd.DataFrame):
    grouped = df.groupby("country_of_origin")["total_cup_points"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 5].sort_values("mean", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=grouped["mean"], y=grouped.index, hue=grouped.index, palette="YlOrBr_r", legend=False, ax=ax)
    ax.set_xlim(78, 87)
    ax.set_xlabel("Average Rating (0-100 scale)")
    ax.set_ylabel("")
    ax.set_title("Top 10 Countries by Average Coffee Rating (min. 5 reviews)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/top_countries.png", dpi=150)
    plt.close(fig)


def plot_processing_method(df: pd.DataFrame):
    order = (df.groupby("processing_method")["total_cup_points"]
             .mean().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="total_cup_points", y="processing_method", hue="processing_method",
                order=order, palette="BuGn_r", legend=False, ax=ax)
    ax.set_xlabel("Rating (0-100 scale)")
    ax.set_ylabel("")
    ax.set_title("Rating Distribution by Processing Method")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/processing_method.png", dpi=150)
    plt.close(fig)


def plot_altitude_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x="altitude_mean_meters", y="total_cup_points",
                scatter_kws={"alpha": 0.4, "s": 20}, line_kws={"color": "firebrick"}, ax=ax)
    ax.set_xlabel("Altitude (meters)")
    ax.set_ylabel("Rating (0-100 scale)")
    ax.set_title("Altitude vs. Rating (weak correlation: r = 0.21)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/altitude_vs_rating.png", dpi=150)
    plt.close(fig)


def plot_subscore_correlations(df: pd.DataFrame):
    subscores = ["aroma", "flavor", "acidity", "body", "balance"]
    corrs = df[subscores + ["total_cup_points"]].corr()["total_cup_points"].drop("total_cup_points")
    corrs = corrs.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=corrs.values, y=corrs.index, hue=corrs.index, palette="Purples", legend=False, ax=ax)
    ax.set_xlabel("Correlation with Overall Rating")
    ax.set_ylabel("")
    ax.set_title("Which Tasting Attribute Predicts the Overall Score Best?")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/subscore_correlations.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(CLEAN_PATH)
    plot_top_countries(df)
    plot_processing_method(df)
    plot_altitude_scatter(df)
    plot_subscore_correlations(df)
    print(f"Saved 4 charts to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
