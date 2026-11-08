from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = ROOT / "audit" / "training_manifest.csv"
IMAGE_ROOT = ROOT / "data" / "Image Data base"

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {"train", "validation", "test"}


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image_path(value):
    """Resolve an image path from the training manifest."""

    relative_path = Path(str(value))
    parts = list(relative_path.parts)

    while parts and parts[0] in {"data", "Image Data base"}:
        parts.pop(0)

    candidates = [
        IMAGE_ROOT / Path(*parts),
        ROOT / relative_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


# ============================================================
# FILE HASHING
# ============================================================

def sha256_file(path):
    """Calculate a file's SHA-256 hash."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# MAIN AUDIT
# ============================================================

def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    print("=" * 65)
    print("SMART PLANT DOCTOR - DATA SPLIT LEAKAGE AUDIT")
    print("=" * 65)

    print("\nManifest columns:")
    print(df.columns.tolist())

    required_columns = {
        "archive_path",
        "species_id",
        "condition_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    if "split" not in df.columns:
        raise ValueError(
            "The manifest has no 'split' column. "
            "Inspect the columns printed above and update the script "
            "to use the column that identifies train/validation/test."
        )

    df["split"] = df["split"].astype(str).str.lower().str.strip()

    df = df[df["split"].isin(SPLITS)].copy()

    print("\nSplit counts:")
    print(df["split"].value_counts().to_string())

    # --------------------------------------------------------
    # 1. Check exact path overlap
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("1. EXACT PATH OVERLAP")
    print("=" * 65)

    df["resolved_path"] = df["archive_path"].apply(
        resolve_image_path
    )

    missing_paths = df["resolved_path"].isna()

    print(f"Rows: {len(df):,}")
    print(f"Images resolved: {(~missing_paths).sum():,}")
    print(f"Missing images: {missing_paths.sum():,}")

    valid_paths = df.dropna(subset=["resolved_path"])

    path_splits = defaultdict(set)

    for row in valid_paths.itertuples():
        path_splits[str(row.resolved_path)].add(row.split)

    path_leaks = {
        path: splits
        for path, splits in path_splits.items()
        if len(splits) > 1
    }

    print(f"Paths occurring across multiple splits: {len(path_leaks):,}")

    # --------------------------------------------------------
    # 2. Check duplicate file contents across splits
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("2. EXACT IMAGE DUPLICATES")
    print("=" * 65)

    print(
        "Hashing resolved images. This may take several minutes..."
    )

    hash_splits = defaultdict(set)
    hash_paths = defaultdict(list)

    resolved_rows = valid_paths[
        ["resolved_path", "split"]
    ].drop_duplicates()

    total = len(resolved_rows)

    for index, row in enumerate(
        resolved_rows.itertuples(index=False),
        start=1,
    ):
        path = Path(row.resolved_path)

        try:
            file_hash = sha256_file(path)
            hash_splits[file_hash].add(row.split)
            hash_paths[file_hash].append(str(path))
        except OSError as error:
            print(f"Could not hash {path}: {error}")

        if index % 10000 == 0 or index == total:
            print(f"Hashed {index:,}/{total:,} images")

    duplicate_hashes = {
        file_hash: splits
        for file_hash, splits in hash_splits.items()
        if len(splits) > 1
    }

    print(
        "\nUnique image hashes appearing across multiple splits:",
        len(duplicate_hashes),
    )

    duplicate_rows = []

    for file_hash, splits in duplicate_hashes.items():
        duplicate_rows.append({
            "sha256": file_hash,
            "splits": ", ".join(sorted(splits)),
            "paths": " | ".join(hash_paths[file_hash]),
        })

    duplicate_report = OUTPUT_DIR / "split_duplicate_images.csv"

    pd.DataFrame(
        duplicate_rows,
        columns=["sha256", "splits", "paths"],
    ).to_csv(duplicate_report, index=False)

    print(f"Duplicate report: {duplicate_report}")

    # --------------------------------------------------------
    # 3. Check class support per split
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("3. SPECIES COUNTS BY SPLIT")
    print("=" * 65)

    species_counts = pd.crosstab(
        df["species_id"],
        df["split"],
    ).reindex(columns=["train", "validation", "test"], fill_value=0)

    print(species_counts.to_string())

    print("\n" + "=" * 65)
    print("4. CONDITION COUNTS BY SPLIT")
    print("=" * 65)

    condition_counts = pd.crosstab(
        df["condition_id"],
        df["split"],
    ).reindex(columns=["train", "validation", "test"], fill_value=0)

    print(condition_counts.to_string())

    species_counts.to_csv(
        OUTPUT_DIR / "species_split_counts.csv"
    )

    condition_counts.to_csv(
        OUTPUT_DIR / "condition_split_counts.csv"
    )

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("AUDIT SUMMARY")
    print("=" * 65)

    if missing_paths.any():
        print("WARNING: Some manifest image paths could not be resolved.")

    if path_leaks:
        print("WARNING: Identical resolved paths occur in multiple splits.")
    else:
        print("No cross-split path overlap found.")

    if duplicate_hashes:
        print(
            "WARNING: Identical file contents occur across multiple splits."
        )
        print(
            "Inspect results/split_duplicate_images.csv before "
            "trusting the evaluation scores."
        )
    else:
        print("No cross-split exact file duplicates found.")

    print(
        "\nNote: This checks exact file duplicates, not visually similar "
        "images or duplicate images with different encodings."
    )

    print("\nAudit completed. No model or dataset files were modified.")


if __name__ == "__main__":
    main()