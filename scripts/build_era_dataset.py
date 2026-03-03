"""
Build a dataset CSV from poem_features.csv: keep era buckets and all features,
drop exact Year. Output: data/poem_features_by_era.csv
"""

import pandas as pd

FEATURES_CACHE = "data/poem_features.csv"
OUTPUT_CSV = "data/poem_features_by_era.csv"

def main():
    df = pd.read_csv(FEATURES_CACHE)
    out = df.drop(columns=["Year"])
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUTPUT_CSV}")
    print(f"Columns: {list(out.columns)}")

if __name__ == "__main__":
    main()
