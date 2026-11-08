from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd
from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "audit" / "training_manifest.csv"
OUTPUT_PATH = ROOT / "results" / "image_duplicate_audit.csv"

df = pd.read_csv(MANIFEST_PATH)

def resolve_path(value):
    relative = Path(str(value))
    parts = list(relative.parts)

    while parts and parts[0] in {"data", "Image Data base"}:
        parts.pop(0)

    return ROOT / "data" / "Image Data base" / Path(*parts)

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

def perceptual_hash(path):
    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("RGB")))

# Exact duplicates: same file content, regardless of filename.
exact_groups = defaultdict(list)

# Perceptual hashes: useful for detecting visually similar images.
phash_groups = defaultdict(list)

missing = []
processed = 0

for row_index, row in df.iterrows():
    path = resolve_path(row["archive_path"])

    if not path.is_file():
        missing.append({
            "manifest_row": row_index,
            "path": str(path),
            "split": row.get("split", ""),
        })
        continue

    try:
        split = str(row["split"])
        exact_hash = sha256_file(path)
        visual_hash = perceptual_hash(path)

        exact_groups[exact_hash].append({
            "manifest_row": row_index,
            "path": str(path),
            "split": split,
        })

        phash_groups[visual_hash].append({
            "manifest_row": row_index,
            "path": str(path),
            "split": split,
        })

        processed += 1

    except Exception as error:
        print(f"Could not process {path}: {error}")

records = []

for duplicate_type, groups in [
    ("exact_content", exact_groups),
    ("identical_phash", phash_groups),
]:
    for group_hash, items in groups.items():
        if len(items) < 2:
            continue

        splits = sorted({item["split"] for item in items})
        cross_split = len(splits) > 1

        for item in items:
            records.append({
                "duplicate_type": duplicate_type,
                "hash": group_hash,
                "group_size": len(items),
                "cross_split": cross_split,
                "splits": ", ".join(splits),
                **item,
            })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

columns = [
    "duplicate_type",
    "hash",
    "group_size",
    "cross_split",
    "splits",
    "manifest_row",
    "path",
    "split",
]

pd.DataFrame(records, columns=columns).to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 68)
print("IMAGE DUPLICATE AUDIT")
print("=" * 68)
print(f"Manifest rows: {len(df):,}")
print(f"Images processed: {processed:,}")
print(f"Missing files: {len(missing):,}")

for duplicate_type, groups in [
    ("exact_content", exact_groups),
    ("identical_phash", phash_groups),
]:
    repeated = [items for items in groups.values() if len(items) > 1]
    cross_split = [
        items for items in repeated
        if len({item["split"] for item in items}) > 1
    ]

    print(f"\n{duplicate_type}:")
    print(f"  Repeated groups: {len(repeated):,}")
    print(f"  Cross-split groups: {len(cross_split):,}")

print(f"\nReport saved to: {OUTPUT_PATH}")
print("=" * 68)

if missing:
    missing_path = ROOT / "results" / "missing_image_files.csv"
    pd.DataFrame(missing).to_csv(missing_path, index=False)
    print(f"Missing-file report saved to: {missing_path}")