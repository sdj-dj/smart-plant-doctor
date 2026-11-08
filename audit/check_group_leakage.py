from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "audit" / "training_manifest.csv"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(MANIFEST)

    required = {"group_id", "split", "class_name"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.lower().str.strip()

    df = df[df["split"].isin(["train", "validation", "test"])].copy()

    print("=" * 65)
    print("SMART PLANT DOCTOR - GROUP LEAKAGE AUDIT")
    print("=" * 65)

    # Exclude empty group IDs because they cannot establish group identity.
    group_values = df["group_id"].astype("string").str.strip()
    valid_group = group_values.notna() & group_values.ne("")

    df["group_id"] = group_values
    grouped_df = df.loc[valid_group].copy()

    if grouped_df.empty:
        print("No usable group IDs found.")
        return

    # Count how many distinct splits contain each group.
    group_summary = (
        grouped_df.groupby("group_id")
        .agg(
            split_count=("split", "nunique"),
            splits=("split", lambda x: ", ".join(sorted(set(x)))),
            image_count=("split", "size"),
            class_count=("class_name", "nunique"),
            classes=("class_name", lambda x: ", ".join(sorted(set(x)))),
        )
        .reset_index()
    )

    crossing_groups = group_summary[
        group_summary["split_count"] > 1
    ].copy()

    print(f"\nManifest rows: {len(df):,}")
    print(f"Rows with usable group IDs: {len(grouped_df):,}")
    print(f"Unique groups: {grouped_df['group_id'].nunique():,}")
    print(f"Groups appearing in multiple splits: {len(crossing_groups):,}")

    output = RESULTS / "group_split_leakage.csv"
    crossing_groups.to_csv(output, index=False)

    print(f"Report saved to: {output}")

    if crossing_groups.empty:
        print("\nPASS: No group IDs cross train/validation/test splits.")
    else:
        print("\nWARNING: Some groups occur across multiple splits.")
        print("\nFirst 25 groups to investigate:")
        print(
            crossing_groups
            .sort_values("image_count", ascending=False)
            .head(25)
            .to_string(index=False)
        )

    # Show the number of groups assigned to each split.
    print("\nUnique groups per split:")
    for split in ["train", "validation", "test"]:
        subset = grouped_df[grouped_df["split"] == split]
        print(f"{split:12}: {subset['group_id'].nunique():,}")

    print("\nAudit completed. No dataset or model files were modified.")


if __name__ == "__main__":
    main()