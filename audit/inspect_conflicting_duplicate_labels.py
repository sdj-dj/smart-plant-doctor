from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "audit" / "training_manifest.csv"

LABEL_REPORT_PATH = (
    ROOT / "results" / "duplicate_label_analysis.csv"
)

DUPLICATE_REPORT_PATH = (
    ROOT / "results" / "image_duplicate_audit.csv"
)

OUTPUT_PATH = (
    ROOT / "results" / "conflicting_duplicate_details.csv"
)

DATASET_ROOT = ROOT / "data" / "Image Data base"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resolve_manifest_path(value):
    """
    Convert a manifest archive_path into an absolute image path.

    Supports relative paths such as:
        Apple Apple scab/Apple_Apple_scab1.jpg
        data/Image Data base/Apple Apple scab/Apple_Apple_scab1.jpg

    Also supports absolute paths.
    """

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return None

    path = Path(value)

    # Keep existing absolute paths.
    if path.is_absolute():
        return str(path.resolve())

    # Normalize separators for paths recorded on different platforms.
    parts = list(Path(value.replace("\\", "/")).parts)

    # Remove dataset-root prefixes if present.
    while parts and parts[0] in {"data", "Image Data base"}:
        parts.pop(0)

    return str((DATASET_ROOT / Path(*parts)).resolve())


def resolve_duplicate_path(value):
    """
    Convert a duplicate-report image path into an absolute path.
    """

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return None

    path = Path(value)

    if path.is_absolute():
        return str(path.resolve())

    return resolve_manifest_path(value)


def require_columns(dataframe, required_columns, filename):
    """
    Check required columns and give a readable error if any are missing.
    """

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing required columns in {filename}: {missing}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CONFLICTING DUPLICATE LABEL DETAILS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check that all required files exist
    # --------------------------------------------------------

    required_files = [
        MANIFEST_PATH,
        LABEL_REPORT_PATH,
        DUPLICATE_REPORT_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"\nRequired file not found:\n{file_path}"
            )

    # --------------------------------------------------------
    # 2. Load CSV files
    # --------------------------------------------------------

    manifest = pd.read_csv(MANIFEST_PATH)

    label_report = pd.read_csv(LABEL_REPORT_PATH)

    duplicate_report = pd.read_csv(DUPLICATE_REPORT_PATH)

    print(f"\nManifest rows: {len(manifest):,}")
    print(f"Label-analysis rows: {len(label_report):,}")
    print(f"Duplicate-report rows: {len(duplicate_report):,}")

    # --------------------------------------------------------
    # 3. Validate required columns
    # --------------------------------------------------------

    require_columns(
        label_report,
        [
            "hash",
            "conflicting_conditions",
            "conflicting_species",
        ],
        LABEL_REPORT_PATH.name,
    )

    require_columns(
        duplicate_report,
        [
            "hash",
            "path",
            "duplicate_type",
        ],
        DUPLICATE_REPORT_PATH.name,
    )

    require_columns(
        manifest,
        [
            "archive_path",
            "class_name",
        ],
        MANIFEST_PATH.name,
    )

    # --------------------------------------------------------
    # 4. Find groups with conflicting labels
    # --------------------------------------------------------

    # Convert the flags safely in case they were read as strings.
    def as_boolean(value):
        if isinstance(value, bool):
            return value

        if pd.isna(value):
            return False

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
        }

    label_report["conflicting_conditions"] = (
        label_report["conflicting_conditions"].map(as_boolean)
    )

    label_report["conflicting_species"] = (
        label_report["conflicting_species"].map(as_boolean)
    )

    conflicts = label_report[
        label_report["conflicting_conditions"]
        | label_report["conflicting_species"]
    ].copy()

    if conflicts.empty:
        print("\nNo conflicting duplicate-label groups were found.")
        return

    conflict_hashes = set(
        conflicts["hash"].dropna().astype(str)
    )

    print(
        f"\nConflicting groups found: {len(conflict_hashes):,}"
    )

    # --------------------------------------------------------
    # 5. Select the matching perceptual-hash image rows
    # --------------------------------------------------------

    duplicates = duplicate_report[
        (duplicate_report["duplicate_type"] == "identical_phash")
        & duplicate_report["hash"].astype(str).isin(conflict_hashes)
    ].copy()

    if duplicates.empty:
        print(
            "\nNo matching image rows were found in "
            "image_duplicate_audit.csv."
        )
        return

    duplicates["path_key"] = duplicates["path"].map(
        resolve_duplicate_path
    )

    # --------------------------------------------------------
    # 6. Prepare manifest paths
    # --------------------------------------------------------

    manifest["path_key"] = manifest["archive_path"].map(
        resolve_manifest_path
    )

    # Only include columns that actually exist in the manifest.
    manifest_columns = [
        "path_key",
        "archive_path",
        "class_name",
        "split",
        "plant_species",
        "condition",
        "condition_id",
        "species_id",
        "species_mask",
        "condition_mask",
        "review_required",
    ]

    manifest_columns = [
        column
        for column in manifest_columns
        if column in manifest.columns
    ]

    manifest_subset = manifest[manifest_columns].copy()

    # --------------------------------------------------------
    # 7. Merge image rows with manifest labels
    # --------------------------------------------------------

    details = duplicates.merge(
        manifest_subset,
        on="path_key",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 8. Report unmatched paths
    # --------------------------------------------------------

    unmatched = details["archive_path"].isna().sum()

    print(f"Matched image rows: {len(details) - unmatched:,}")
    print(f"Unmatched image rows: {unmatched:,}")

    # --------------------------------------------------------
    # 9. Add conflict flags for easy inspection
    # --------------------------------------------------------

    conflict_flags = conflicts[
        [
            "hash",
            "conflicting_conditions",
            "conflicting_species",
        ]
    ].drop_duplicates(subset=["hash"])

    details = details.merge(
        conflict_flags,
        on="hash",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 10. Sort and save
    # --------------------------------------------------------

    sort_columns = [
        column
        for column in ["hash", "split", "archive_path"]
        if column in details.columns
    ]

    if sort_columns:
        details = details.sort_values(sort_columns)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    details.to_csv(OUTPUT_PATH, index=False)

    # --------------------------------------------------------
    # 11. Print the details safely
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFLICTING DUPLICATE DETAILS")
    print("=" * 70)

    columns_to_show = [
        column
        for column in [
            "hash",
            "path",
            "archive_path",
            "split",
            "class_name",
            "plant_species",
            "condition",
            "species_id",
            "condition_id",
            "conflicting_species",
            "conflicting_conditions",
        ]
        if column in details.columns
    ]

    if details.empty:
        print("\nNo matching image details to display.")
    else:
        print(
            details[columns_to_show].to_string(
                index=False,
                max_rows=100,
            )
        )

    print("\n" + "=" * 70)
    print(f"Full report saved to:\n{OUTPUT_PATH}")
    print("=" * 70)

    if unmatched:
        print(
            "\nWARNING: Some image paths could not be matched to "
            "the manifest. Check the paths before drawing conclusions."
        )


if __name__ == "__main__":
    main()