from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from ml.dataset import PlantDataset
from ml.model import load_model, IMAGE_SIZE


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = {
    "previous": ROOT / "checkpoints/experiments/smart_plant_doctor_best.pth",
    "balanced": ROOT / "checkpoints/experiments/smart_plant_doctor_balanced.pth",
}

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0

SPECIES_NAMES = [
    "Apple",
    "Bell pepper",
    "Blueberry",
    "Cherry",
    "Corn",
    "Garlic",
    "Ginger",
    "Grape",
    "Lemon",
    "Onion",
    "Orange",
    "Peach",
    "Potato",
    "Raspberry",
    "Rice",
    "Soybean",
    "Strawberry",
    "Tea",
    "Tomato",
]

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
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 65)
print("SMART PLANT DOCTOR — TEST MODEL COMPARISON")
print("=" * 65)

test_dataset = PlantDataset(
    split="test",
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print(f"Test images: {len(test_dataset):,}")
print(f"Batch size: {BATCH_SIZE}")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_checkpoint(model_name, checkpoint_path):

    print("\n" + "-" * 65)
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 65)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model, device = load_model(str(checkpoint_path))
    model.eval()

    species_true = []
    species_pred = []

    condition_true = []
    condition_pred = []

    with torch.inference_mode():

        for batch in tqdm(
            test_loader,
            desc=f"Testing {model_name}",
        ):

            images = batch["image"].to(device)

            species_logits, condition_logits = model(images)

            predicted_species = (
                species_logits.argmax(dim=1).cpu().numpy()
            )

            predicted_conditions = (
                condition_logits.argmax(dim=1).cpu().numpy()
            )

            true_species = batch["species"].cpu().numpy()
            true_conditions = batch["condition"].cpu().numpy()

            species_mask = (
                batch["species_mask"].cpu().numpy().astype(bool)
            )

            condition_mask = (
                batch["condition_mask"].cpu().numpy().astype(bool)
            )

            species_true.extend(true_species[species_mask].tolist())
            species_pred.extend(
                predicted_species[species_mask].tolist()
            )

            condition_true.extend(
                true_conditions[condition_mask].tolist()
            )

            condition_pred.extend(
                predicted_conditions[condition_mask].tolist()
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    species_accuracy = accuracy_score(
        species_true,
        species_pred,
    )

    species_macro_f1 = f1_score(
        species_true,
        species_pred,
        labels=list(range(len(SPECIES_NAMES))),
        average="macro",
        zero_division=0,
    )

    condition_accuracy = accuracy_score(
        condition_true,
        condition_pred,
    )

    condition_macro_f1 = f1_score(
        condition_true,
        condition_pred,
        labels=list(range(len(CONDITION_NAMES))),
        average="macro",
        zero_division=0,
    )

    balanced_macro_f1 = (
        species_macro_f1 + condition_macro_f1
    ) / 2

    metrics = {
        "model": model_name,
        "species_accuracy": species_accuracy,
        "species_macro_f1": species_macro_f1,
        "condition_accuracy": condition_accuracy,
        "condition_macro_f1": condition_macro_f1,
        "balanced_macro_f1": balanced_macro_f1,
    }

    print("\nRESULTS")
    print(f"Species accuracy:       {species_accuracy:.4f}")
    print(f"Species macro F1:       {species_macro_f1:.4f}")
    print(f"Condition accuracy:     {condition_accuracy:.4f}")
    print(f"Condition macro F1:     {condition_macro_f1:.4f}")
    print(f"Balanced macro F1:      {balanced_macro_f1:.4f}")

    # --------------------------------------------------------
    # CLASSIFICATION REPORTS
    # --------------------------------------------------------

    species_report = classification_report(
        species_true,
        species_pred,
        labels=list(range(len(SPECIES_NAMES))),
        target_names=SPECIES_NAMES,
        output_dict=True,
        zero_division=0,
    )

    condition_report = classification_report(
        condition_true,
        condition_pred,
        labels=list(range(len(CONDITION_NAMES))),
        target_names=CONDITION_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(species_report).transpose().to_csv(
        RESULTS_DIR / f"{model_name}_test_species_report.csv"
    )

    pd.DataFrame(condition_report).transpose().to_csv(
        RESULTS_DIR / f"{model_name}_test_condition_report.csv"
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    species_cm = confusion_matrix(
        species_true,
        species_pred,
        labels=list(range(len(SPECIES_NAMES))),
    )

    condition_cm = confusion_matrix(
        condition_true,
        condition_pred,
        labels=list(range(len(CONDITION_NAMES))),
    )

    pd.DataFrame(
        species_cm,
        index=SPECIES_NAMES,
        columns=SPECIES_NAMES,
    ).to_csv(
        RESULTS_DIR / f"{model_name}_test_species_confusion_matrix.csv"
    )

    pd.DataFrame(
        condition_cm,
        index=CONDITION_NAMES,
        columns=CONDITION_NAMES,
    ).to_csv(
        RESULTS_DIR / f"{model_name}_test_condition_confusion_matrix.csv"
    )

    del model

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics, condition_report


# ============================================================
# RUN BOTH MODELS
# ============================================================

all_metrics = []
all_condition_reports = {}

for model_name, checkpoint_path in CHECKPOINTS.items():

    metrics, condition_report = evaluate_checkpoint(
        model_name,
        checkpoint_path,
    )

    all_metrics.append(metrics)
    all_condition_reports[model_name] = condition_report


# ============================================================
# COMPARE RESULTS
# ============================================================

comparison = pd.DataFrame(all_metrics).set_index("model")

comparison.to_csv(
    RESULTS_DIR / "balanced_vs_previous_test_comparison.csv"
)

print("\n" + "=" * 65)
print("FINAL TEST COMPARISON")
print("=" * 65)

print(comparison.round(4).to_string())

print("\nCHANGE: BALANCED MODEL MINUS PREVIOUS MODEL")

metric_columns = [
    "species_accuracy",
    "species_macro_f1",
    "condition_accuracy",
    "condition_macro_f1",
    "balanced_macro_f1",
]

if "previous" in comparison.index and "balanced" in comparison.index:

    difference = (
        comparison.loc["balanced", metric_columns]
        - comparison.loc["previous", metric_columns]
    )

    print(difference.round(4).to_string())

    print("\nINTERPRETATION")

    if difference["condition_macro_f1"] > 0:
        print("Condition macro F1 improved.")
    elif difference["condition_macro_f1"] < 0:
        print("Condition macro F1 decreased.")
    else:
        print("Condition macro F1 stayed the same.")

    if difference["species_macro_f1"] < 0:
        print("WARNING: Species macro F1 decreased.")
    elif difference["species_macro_f1"] > 0:
        print("Species macro F1 improved.")

print("\nReports saved in:", RESULTS_DIR)
print("Original checkpoints were not modified.")
print("=" * 65)