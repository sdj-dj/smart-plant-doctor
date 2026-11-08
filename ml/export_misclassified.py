from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ml.dataset import PlantDataset
from ml.model import load_model, IMAGE_SIZE


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "experiments"
    / "smart_plant_doctor_balanced.pth"
)

MANIFEST = ROOT / "audit" / "training_manifest.csv"
OUTPUT = ROOT / "results" / "misclassified_conditions.csv"

BATCH_SIZE = 16

CONDITION_NAMES = [
    "Algal leaf",
    "Anthracnose",
    "Apple scab",
    "Bacterial leaf blight",
    "Bacterial spot",
    "Bird eye spot",
    "Black rot",
    "Blight",
    "Brown blight",
    "Brown spot",
    "Cabbage looper",
    "Canker",
    "Cedar apple rust",
    "Cercospora leaf spot",
    "Citrus greening",
    "Common rust",
    "Early blight",
    "Esca black measles",
    "Gray leaf spot",
    "Healthy",
    "Hollow heart",
    "Late blight",
    "Leaf blight",
    "Leaf mold",
    "Leaf scorch",
    "Leaf smut",
    "Nitrogen deficiency",
    "Potassium deficiency",
    "Powdery mildew",
    "Red leaf spot",
    "Septoria leaf spot",
    "Sogatella",
    "Spider mites",
    "Target spot",
    "Tomato mosaic virus",
    "Waterlogging",
]


# ============================================================
# CHECK FILES
# ============================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")


# ============================================================
# LOAD TEST DATA
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = PlantDataset(
    split="test",
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

manifest = pd.read_csv(MANIFEST)

test_manifest = manifest[
    manifest["split"].astype(str).str.lower() == "test"
].reset_index(drop=True)

if len(test_manifest) != len(dataset):
    raise ValueError(
        "Test manifest and dataset lengths differ. "
        "Cannot safely match predictions to file paths."
    )

model, device = load_model(str(CHECKPOINT))

records = []
row_offset = 0


# ============================================================
# PREDICT AND COLLECT ERRORS
# ============================================================

model.eval()

with torch.inference_mode():

    for batch in tqdm(loader, desc="Finding misclassified images"):

        images = batch["image"].to(device)

        _, condition_logits = model(images)

        probabilities = torch.softmax(condition_logits, dim=1)

        confidence, predicted = probabilities.max(dim=1)

        true_labels = batch["condition"].cpu().tolist()
        predicted_labels = predicted.cpu().tolist()
        confidences = confidence.cpu().tolist()

        masks = batch["condition_mask"].cpu().tolist()

        batch_size = len(true_labels)

        batch_manifest = test_manifest.iloc[
            row_offset:row_offset + batch_size
        ]

        for i in range(batch_size):

            if not masks[i]:
                continue

            true_id = int(true_labels[i])
            predicted_id = int(predicted_labels[i])

            if true_id == predicted_id:
                continue

            manifest_row = batch_manifest.iloc[i]

            records.append({
                "archive_path": manifest_row["archive_path"],
                "true_condition_id": true_id,
                "true_condition": CONDITION_NAMES[true_id],
                "predicted_condition_id": predicted_id,
                "predicted_condition": CONDITION_NAMES[predicted_id],
                "prediction_confidence": round(
                    float(confidences[i]), 6
                ),
                "confidence_percent": round(
                    float(confidences[i]) * 100, 2
                ),
            })

        row_offset += batch_size


# ============================================================
# SAVE RESULTS
# ============================================================

columns = [
    "archive_path",
    "true_condition_id",
    "true_condition",
    "predicted_condition_id",
    "predicted_condition",
    "prediction_confidence",
    "confidence_percent",
]

results = pd.DataFrame(records, columns=columns)

if not results.empty:
    results = results.sort_values(
        "prediction_confidence",
        ascending=False,
    )

results.to_csv(OUTPUT, index=False)

print("\n" + "=" * 70)
print("MISCLASSIFIED CONDITION IMAGES")
print("=" * 70)

print(f"Test images: {len(dataset):,}")
print(f"Misclassified labelled images: {len(results):,}")

if len(results):
    print("\nMost frequent confusion pairs:")

    pairs = (
        results.groupby(
            ["true_condition", "predicted_condition"]
        )
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    for (actual, predicted), count in pairs.items():
        print(f"{actual} -> {predicted}: {count}")

    print("\nHighest-confidence errors:")
    print(
        results[
            [
                "true_condition",
                "predicted_condition",
                "confidence_percent",
                "archive_path",
            ]
        ].head(10).to_string(index=False)
    )
else:
    print("No misclassified labelled condition images found.")

print(f"\nFull report saved to:\n{OUTPUT}")
print("=" * 70)