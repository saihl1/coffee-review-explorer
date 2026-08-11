"""
analyze.py
----------
Answers a few real questions about coffee quality using the cleaned dataset.

Key pandas concept used throughout: `groupby()`
    df.groupby("some_column")["value_column"].mean()
is the pandas equivalent of an Excel pivot table: "for each unique value in
some_column, compute the average of value_column."
"""

import pandas as pd

CLEAN_PATH = "data/coffee_ratings_clean.csv"


def top_countries_by_rating(df: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    """Average rating by country, only for countries with enough reviews
    to be meaningful (a country with 1 review isn't a fair comparison)."""
    grouped = df.groupby("country_of_origin")["total_cup_points"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= min_reviews]
    return grouped.sort_values("mean", ascending=False).round(2)


def rating_by_processing_method(df: pd.DataFrame) -> pd.DataFrame:
    """Does how the coffee is processed (washed, natural, honey, etc.)
    relate to its rating?"""
    grouped = df.groupby("processing_method")["total_cup_points"].agg(["mean", "count"])
    return grouped.sort_values("mean", ascending=False).round(2)


def altitude_correlation(df: pd.DataFrame) -> float:
    """Does growing altitude correlate with rating? Higher altitude coffee
    is often considered higher quality -- let's check if the data agrees."""
    return df[["altitude_mean_meters", "total_cup_points"]].corr().iloc[0, 1]


def subscore_correlations(df: pd.DataFrame) -> pd.Series:
    """Which individual tasting attribute (aroma, flavor, acidity, body,
    balance) correlates most strongly with the overall score?"""
    subscores = ["aroma", "flavor", "acidity", "body", "balance"]
    corrs = df[subscores + ["total_cup_points"]].corr()["total_cup_points"].drop("total_cup_points")
    return corrs.sort_values(ascending=False).round(3)


def main():
    df = pd.read_csv(CLEAN_PATH)

    print("=" * 60)
    print("TOP-RATED COUNTRIES (min 5 reviews)")
    print("=" * 60)
    print(top_countries_by_rating(df).head(10))

    print()
    print("=" * 60)
    print("RATING BY PROCESSING METHOD")
    print("=" * 60)
    print(rating_by_processing_method(df))

    print()
    print("=" * 60)
    print("ALTITUDE vs RATING")
    print("=" * 60)
    corr = altitude_correlation(df)
    print(f"Correlation: {corr:.3f}  "
          f"({'weak' if abs(corr) < 0.3 else 'moderate' if abs(corr) < 0.6 else 'strong'})")

    print()
    print("=" * 60)
    print("WHICH TASTING ATTRIBUTE MATTERS MOST?")
    print("=" * 60)
    print(subscore_correlations(df))


if __name__ == "__main__":
    main()
