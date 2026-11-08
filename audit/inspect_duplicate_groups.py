from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "results" / "image_duplicate_audit.csv"
OUTPUT_PATH = ROOT / "results" / "cross_split_duplicate_groups.csv"

df = pd.read_csv(INPUT_PATH)

# Keep perceptual-hash groups spanning multiple dataset splits.
suspects = df[
    (df["duplicate_type"] == "identical_phash")
    & (df["cross_split"] == True)
].copy()

if suspects.empty:
    print("No cross-split perceptual-hash groups found.")
    raise SystemExit(0)

# Keep one row per image/group combination.
suspects = suspects.drop_duplicates(
    subset=["hash", "path", "split"]
)

# Summarize each suspected group.
summary = (
    suspects.groupby("hash")
    .agg(
        image_count=("path", "nunique"),
        splits=("split", lambda x: ", ".join(sorted(set(x)))),
        paths=("path", lambda x: " || ".join(x)),
    )
    .reset_index()
)

summary["split_count"] = summary["splits"].apply(
    lambda value: len(value.split(", "))
)

summary = summary.sort_values(
    ["split_count", "image_count"],
    ascending=False,
)

summary.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 68)
print("CROSS-SPLIT DUPLICATE GROUP INSPECTION")
print("=" * 68)
print(f"Suspected groups: {len(summary):,}")
print(f"Groups spanning multiple splits: "
      f"{(summary['split_count'] > 1).sum():,}")

print("\nFirst 20 groups:")
print(
    summary[
        ["hash", "image_count", "splits"]
    ].head(20).to_string(index=False)
)

print(f"\nDetailed report saved to: {OUTPUT_PATH}")
print("=" * 68)