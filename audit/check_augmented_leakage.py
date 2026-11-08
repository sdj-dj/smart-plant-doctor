from pathlib import Path
from collections import defaultdict
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "audit" / "training_manifest.csv"
OUTPUT_PATH = ROOT / "results" / "augmented_leakage_audit.csv"

AUGMENTATION_PATTERNS = [
    r"_Horizontally_flip",
    r"_Vertically_flip",
    r"_Horizontally_and_Vertically",
    r"_Flip_Horizontally_and_Vertically",
    r"_Flip_Vertically",
    r"_Flip_Horizontally",
    r"_rotated",
    r"_rotation",
    r"_augmented",
    r"_augmentation",
    r"_zoom",
    r"_brightness",
    r"_contrast",
]

def normalize_filename(path):
    name = Path(str(path)).name

    # Remove extension, then repeatedly remove recognized
    # augmentation suffixes.
    name = re.sub(r"\.[^.]+$", "", name)

    changed = True
    while changed:
        changed = False

        for pattern in AUGMENTATION_PATTERNS:
            updated = re.sub(
                pattern + r".*$",
                "",
                name,
                flags=re.IGNORECASE,
            )

            if updated != name:
                name = updated
                changed = True

    return name.lower()


print("\n" + "=" * 70)
print("AUGMENTED IMAGE CROSS-SPLIT AUDIT")
print("=" * 70)

manifest = pd.read_csv(MANIFEST_PATH)

required = {"archive_path", "split"}
missing = required - set(manifest.columns)

if missing:
    raise ValueError(f"Manifest is missing columns: {missing}")

manifest["split"] = manifest["split"].astype(str).str.lower()
manifest["normalized_filename"] = manifest["archive_path"].apply(
    normalize_filename
)

groups = defaultdict(list)

for index, row in manifest.iterrows():
    groups[row["normalized_filename"]].append({
        "index": index,
        "split": row["split"],
        "path": row["archive_path"],
    })

leakage_records = []

for normalized_name, rows in groups.items():
    splits = sorted({row["split"] for row in rows})

    if len(splits) > 1:
        for row in rows:
            leakage_records.append({
                "normalized_filename": normalized_name,
                "split": row["split"],
                "archive_path": row["path"],
                "splits_sharing_name": ", ".join(splits),
                "group_size": len(rows),
            })

leakage = pd.DataFrame(leakage_records)

if leakage.empty:
    leakage = pd.DataFrame(columns=[
        "normalized_filename",
        "split",
        "archive_path",
        "splits_sharing_name",
        "group_size",
    ])

leakage.to_csv(OUTPUT_PATH, index=False)

print(f"Manifest rows: {len(manifest):,}")
print(f"Normalized filename groups: {len(groups):,}")
print(f"Cross-split filename matches: {len(leakage):,}")

if not leakage.empty:
    print("\nPotential cross-split matches:")
    print(leakage.head(30).to_string(index=False))
    print(
        "\nWARNING: These are filename-based candidates, "
        "not proof of identical image content."
    )
else:
    print(
        "\nNo cross-split matches were found using the "
        "recognized filename patterns."
    )

print(f"\nAudit saved to:\n{OUTPUT_PATH}")
print("=" * 70)