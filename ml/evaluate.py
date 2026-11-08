import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml.dataset import PlantDataset
from ml.model import load_model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "results" / "leakage_reduced_manifest.csv"
SPECIES_PATH = PROJECT_ROOT / "audit" / "species_classes.json"
CONDITION_PATH = PROJECT_ROOT / "audit" / "condition_classes.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "experiments"
    / "leakage_reduced"
    / "condition_balanced_best_v1.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "leakage_reduced" / "test_evaluation"
BATCH_SIZE = 32


# ============================================================
# LOAD CLASS MAPPINGS
# ============================================================

def load_class_mapping(path):
    with open(path, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    return [
        name
        for name, class_id in sorted(
            mapping.items(), key=lambda item: item[1]
        )
    ]


# ============================================================
# EVALUATION
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)

    species_names = load_class_mapping(SPECIES_PATH)
    condition_names = load_class_mapping(CONDITION_PATH)

    # Use the existing validation split.
    validation = manifest[
        manifest["new_split"] == "test"
    ].copy()

    print("Held-out test rows:", len(validation))

    # Only evaluate rows with a valid label for each task.
    species_valid = validation[
        validation["species_mask"].astype(bool)
        & validation["species_id"].between(
            0, len(species_names) - 1
        )
    ]

    condition_valid = validation[
        validation["condition_mask"].astype(bool)
        & validation["condition_id"].between(
            0, len(condition_names) - 1
        )
    ]

    print("Valid species labels:", len(species_valid))
    print("Valid condition labels:", len(condition_valid))

    from ml.model import ResNetAgriculturalAI, get_device

    device = get_device()
    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    model = ResNetAgriculturalAI(
        num_species=19, num_conditions=36, pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Device:", device)

    # Load the validation dataset using the existing dataset class.
    dataset = PlantDataset(split="test")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    species_true = []
    species_pred = []

    condition_true = []
    condition_pred = []

    model.eval()

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)

            species = batch["species"]
            condition = batch["condition"]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                species_logits, condition_logits = model(images)

            species_predictions = species_logits.argmax(dim=1).cpu()
            condition_predictions = condition_logits.argmax(dim=1).cpu()

            # Handle each task's labels independently.
            species_mask = (
                (species >= 0)
                & (species < len(species_names))
            )

            condition_mask = (
                (condition >= 0)
                & (condition < len(condition_names))
            )

            species_true.extend(
                species[species_mask].tolist()
            )
            species_pred.extend(
                species_predictions[species_mask].tolist()
            )

            condition_true.extend(
                condition[condition_mask].tolist()
            )
            condition_pred.extend(
                condition_predictions[condition_mask].tolist()
            )

    # --------------------------------------------------------
    # SPECIES METRICS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SPECIES EVALUATION")
    print("=" * 60)

    species_accuracy = accuracy_score(
        species_true, species_pred
    )

    species_macro_f1 = f1_score(
        species_true,
        species_pred,
        labels=list(range(len(species_names))),
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy: {species_accuracy:.4f}")
    print(f"Macro F1: {species_macro_f1:.4f}")

    species_report = classification_report(
        species_true,
        species_pred,
        labels=list(range(len(species_names))),
        target_names=species_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(species_report).transpose().to_csv(
        RESULTS_DIR / "species_report.csv"
    )

    # --------------------------------------------------------
    # CONDITION METRICS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONDITION EVALUATION")
    print("=" * 60)

    condition_accuracy = accuracy_score(
        condition_true, condition_pred
    )

    condition_macro_f1 = f1_score(
        condition_true,
        condition_pred,
        labels=list(range(len(condition_names))),
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy: {condition_accuracy:.4f}")
    print(f"Macro F1: {condition_macro_f1:.4f}")

    condition_report = classification_report(
        condition_true,
        condition_pred,
        labels=list(range(len(condition_names))),
        target_names=condition_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(condition_report).transpose().to_csv(
        RESULTS_DIR / "condition_report.csv"
    )

    # --------------------------------------------------------
    # SAVE CONFUSION MATRICES
    # --------------------------------------------------------

    species_cm = confusion_matrix(
        species_true,
        species_pred,
        labels=list(range(len(species_names))),
    )

    condition_cm = confusion_matrix(
        condition_true,
        condition_pred,
        labels=list(range(len(condition_names))),
    )

    pd.DataFrame(
        species_cm,
        index=species_names,
        columns=species_names,
    ).to_csv(RESULTS_DIR / "species_confusion_matrix.csv")

    pd.DataFrame(
        condition_cm,
        index=condition_names,
        columns=condition_names,
    ).to_csv(RESULTS_DIR / "condition_confusion_matrix.csv")

    print("\nReports saved to:", RESULTS_DIR)
    print("Species report: results/species_report.csv")
    print("Condition report: results/condition_report.csv")
    print("Confusion matrices saved.")


if __name__ == "__main__":
    main()