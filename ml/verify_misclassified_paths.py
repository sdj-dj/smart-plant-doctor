from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from ml.dataset import PlantDataset
from ml.model import IMAGE_SIZE


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "audit" / "training_manifest.csv"
ERRORS_PATH = ROOT / "results" / "misclassified_conditions.csv"

OUTPUT_PATH = ROOT / "results" / "misclassified_path_audit.csv"

# Check up to 200 samples from the beginning of the test set.
MAX_CHECKS = 200

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def resolve_image_path(value):
    relative = Path(str(value))

    # Handle paths recorded with different project-root prefixes.
    parts = list(relative.parts)

    while parts and parts[0] in {"data", "Image Data base"}:
        parts.pop(0)

    return ROOT / "data" / "Image Data base" / Path(*parts)


print("\n" + "=" * 70)
print("MISCLASSIFIED IMAGE PATH AUDIT")
print("=" * 70)

manifest = pd.read_csv(MANIFEST_PATH)

test_manifest = manifest[
    manifest["split"].astype(str).str.lower() == "test"
].reset_index(drop=True)

dataset = PlantDataset(
    split="test",
    transform=transform,
)

if len(test_manifest) != len(dataset):
    raise RuntimeError(
        f"Dataset has {len(dataset)} samples, but the manifest "
        f"has {len(test_manifest)} test rows. "
        "Do not trust positional path matching."
    )

errors = pd.read_csv(ERRORS_PATH)

if errors.empty:
    print("No misclassified rows to audit.")
    raise SystemExit(0)

checks = min(MAX_CHECKS, len(dataset))
records = []

for index in range(checks):
    row = test_manifest.iloc[index]
    manifest_path = resolve_image_path(row["archive_path"])

    record = {
        "dataset_index": index,
        "manifest_path": str(manifest_path),
        "manifest_path_exists": manifest_path.is_file(),
        "image_tensor_matches": False,
        "status": "",
    }

    if not manifest_path.is_file():
        record["status"] = "PATH_NOT_FOUND"
        records.append(record)
        continue

    try:
        with Image.open(manifest_path) as image:
            expected_tensor = transform(image.convert("RGB"))

        actual_tensor = dataset[index]["image"]

        matches = torch.allclose(
            actual_tensor,
            expected_tensor,
            rtol=1e-5,
            atol=1e-6,
        )

        record["image_tensor_matches"] = matches
        record["status"] = "MATCH" if matches else "ORDER_MISMATCH"

    except Exception as exc:
        record["status"] = f"ERROR: {type(exc).__name__}: {exc}"

    records.append(record)

audit = pd.DataFrame(records)
audit.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset samples checked: {checks}")
print(f"Matching paths: {(audit['status'] == 'MATCH').sum()}")
print(f"Potential order mismatches: {(audit['status'] == 'ORDER_MISMATCH').sum()}")
print(f"Missing paths: {(audit['status'] == 'PATH_NOT_FOUND').sum()}")

print("\nFirst 10 audit results:")
print(audit.head(10).to_string(index=False))

print(f"\nAudit saved to:\n{OUTPUT_PATH}")

if (audit["status"] == "ORDER_MISMATCH").any():
    print(
        "\nWARNING: Manifest row order does not match dataset order. "
        "Rebuild the misclassified-image report using the dataset's "
        "actual paths before interpreting individual errors."
    )
elif (audit["status"] == "MATCH").all():
    print(
        "\nAll checked paths match the dataset image tensors. "
        "This supports positional alignment for the checked samples, "
        "but does not verify every remaining sample."
    )

print("=" * 70)