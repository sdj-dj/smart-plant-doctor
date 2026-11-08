from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "audit" / "training_manifest.csv"
DUPLICATES_PATH = ROOT / "results" / "image_duplicate_audit.csv"
OUTPUT_PATH = ROOT / "results" / "duplicate_label_analysis.csv"

IMAGE_ROOT = ROOT / "data" / "Image Data base"


def resolve_manifest_path(value):
    """Resolve a manifest path using the same convention as PlantDataset."""
    relative_path = Path(str(value))
    parts = list(relative_path.parts)

    while parts and parts[0] in {"data", "Image Data base"}:
        parts.pop(0)

    return (IMAGE_ROOT / Path(*parts)).resolve()


def resolve_duplicate_path(value):
    """Resolve an image path from the duplicate audit."""
    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    duplicates = pd.read_csv(DUPLICATES_PATH)

    required_manifest = {
        "archive_path",
        "split",
        "class_name",
        "plant_species",
        "condition",
    }
    required_duplicates = {
        "duplicate_type",
        "cross_split",
        "hash",
        "path",
    }

    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    if missing := required_duplicates - set(duplicates.columns):
        raise ValueError(f"Missing duplicate columns: {sorted(missing)}")

    duplicates = duplicates[
        (duplicates["duplicate_type"] == "identical_phash")
        & (
            duplicates["cross_split"]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
        )
    ].copy()

    manifest["path_key"] = manifest["archive_path"].map(
        resolve_manifest_path
    )
    duplicates["path_key"] = duplicates["path"].map(
        resolve_duplicate_path
    )

    manifest_columns = [
        "path_key",
        "split",
        "class_name",
        "plant_species",
        "condition",
    ]

    if "species_id" in manifest.columns:
        manifest_columns.append("species_id")

    if "condition_id" in manifest.columns:
        manifest_columns.append("condition_id")

    manifest_subset = manifest[manifest_columns].rename(
        columns={"split": "manifest_split"}
    )

    # A physical image should map to a single manifest row.
    manifest_subset = manifest_subset.drop_duplicates(
        subset=["path_key"]
    )

    merged = duplicates.merge(
        manifest_subset,
        on="path_key",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    matched = merged[merged["_merge"] == "both"].copy()
    unmatched = merged[merged["_merge"] != "both"].copy()

    records = []

    for group_hash, group in matched.groupby("hash"):
        classes = sorted(group["class_name"].dropna().astype(str).unique())
        species = sorted(
            group["plant_species"].dropna().astype(str).unique()
        )
        conditions = sorted(
            group["condition"].dropna().astype(str).unique()
        )
        splits = sorted(
            group["manifest_split"].dropna().astype(str).unique()
        )

        records.append({
            "hash": group_hash,
            "image_count": group["path_key"].nunique(),
            "splits": ", ".join(splits),
            "class_count": len(classes),
            "classes": " | ".join(classes),
            "species_count": len(species),
            "species": " | ".join(species),
            "condition_count": len(conditions),
            "conditions": " | ".join(conditions),
            "conflicting_species": len(species) > 1,
            "conflicting_conditions": len(conditions) > 1,
        })

    report = pd.DataFrame(records)
    report.to_csv(OUTPUT_PATH, index=False)

    print("=" * 68)
    print("DUPLICATE LABEL ANALYSIS")
    print("=" * 68)
    print(f"Suspected cross-split groups: {duplicates['hash'].nunique():,}")
    print(f"Matched image rows: {len(matched):,}")
    print(f"Unmatched image rows: {len(unmatched):,}")
    print(f"Matched groups analyzed: {len(report):,}")

    if not report.empty:
        print(
            "Groups with conflicting species labels:",
            int(report["conflicting_species"].sum()),
        )
        print(
            "Groups with conflicting condition labels:",
            int(report["conflicting_conditions"].sum()),
        )

        print("\nFirst 15 matched groups:")
        print(
            report.head(15)[
                [
                    "hash",
                    "image_count",
                    "splits",
                    "classes",
                    "species",
                    "conditions",
                ]
            ].to_string(index=False)
        )

    if not unmatched.empty:
        unmatched_path = ROOT / "results" / "unmatched_duplicate_paths.csv"
        unmatched[
            ["hash", "path", "path_key", "_merge"]
        ].to_csv(unmatched_path, index=False)
        print(f"\nUnmatched paths saved to: {unmatched_path}")

    print(f"\nReport saved to: {OUTPUT_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()