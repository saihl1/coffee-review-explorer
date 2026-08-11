"""
clean_data.py
-------------
Loads the raw Coffee Quality Institute review data and produces a cleaned
version ready for analysis.

Cleaning decisions (and why):
1. Drop rows with total_cup_points == 0 -> these are data-entry errors,
   not real reviews (every other coffee scores 60-90).
2. Keep only Arabica coffees -> Robusta has too few rows (28) to compare fairly.
3. Fill missing `variety` / `processing_method` with "Unknown" rather than
   dropping those rows -> we don't want to throw away otherwise-good data
   just because one text field is missing.
4. Keep only the columns we actually need for analysis, to keep things readable.
"""

import pandas as pd

RAW_PATH = "data/coffee_ratings.csv"
CLEAN_PATH = "data/coffee_ratings_clean.csv"

COLUMNS_TO_KEEP = [
    "country_of_origin",
    "variety",
    "processing_method",
    "total_cup_points",
    "aroma",
    "flavor",
    "acidity",
    "body",
    "balance",
    "altitude_mean_meters",
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove the data-entry error(s)
    df = df[df["total_cup_points"] > 60]

    # 2. Keep only Arabica (the dominant species in this dataset)
    df = df[df["species"] == "Arabica"]

    # 3. Trim to the columns we care about
    df = df[COLUMNS_TO_KEEP].copy()

    # 4. Fill missing categorical values instead of dropping rows
    df["variety"] = df["variety"].fillna("Unknown")
    df["processing_method"] = df["processing_method"].fillna("Unknown")

    # 5. Drop rows still missing a country (can't group by origin without it)
    df = df.dropna(subset=["country_of_origin"])

    # 6. Altitude has some absurd outliers (e.g. 190,164 meters - Everest is 8,849m!)
    #    Cap it to a plausible coffee-growing range.
    df.loc[(df["altitude_mean_meters"] > 3000) | (df["altitude_mean_meters"] < 0),
           "altitude_mean_meters"] = None

    df = df.reset_index(drop=True)
    return df


def main():
    raw = load_raw()
    print(f"Raw data: {raw.shape[0]} rows, {raw.shape[1]} columns")

    cleaned = clean(raw)
    print(f"Cleaned data: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")

    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
