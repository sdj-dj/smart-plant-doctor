"""
Smart Plant Doctor
Leakage-reduced, condition-balanced training experiment.

Dataset:
    results/leakage_reduced_manifest.csv

Splits:
    new_split = train / validation / test

Model:
    Fresh ImageNet-pretrained ResNet18
    Head 1: plant species (19 classes)
    Head 2: plant condition (36 classes)

Safety:
    - Does not load the old project checkpoint.
    - Does not load test images during training.
    - Does not overwrite existing experiment outputs.
    - Saves all outputs in isolated directories.
"""

from collections import Counter
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from ml.dataset import PlantDataset
from ml.model import (
    NUM_SPECIES,
    NUM_CONDITIONS,
    IMAGE_SIZE,
    ResNetAgriculturalAI,
    get_device,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "results" / "leakage_reduced_manifest.csv"

OUTPUT_DIR = (
    ROOT
    / "checkpoints"
    / "experiments"
    / "leakage_reduced"
)

RESULTS_DIR = ROOT / "results" / "leakage_reduced"

# These filenames are dedicated to this experiment.
CHECKPOINT_PATH = OUTPUT_DIR / "condition_balanced_v1.pth"

BEST_CHECKPOINT_PATH = (
    OUTPUT_DIR / "condition_balanced_best_v1.pth"
)

HISTORY_PATH = (
    RESULTS_DIR / "condition_balanced_history_v1.csv"
)

SEED = 42

BATCH_SIZE = 16
EPOCHS = 1
NUM_WORKERS = 0

BACKBONE_LR = 1e-5
HEAD_LR = 1e-4

MAX_WEIGHT = 4.0
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0

# First run: True.
# After the smoke test succeeds, change to False for real training.
SMOKE_TEST = False

# Number of training batches processed by the smoke test.
SMOKE_TEST_BATCHES = 2


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    """Set random seeds for more reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. MANIFEST VALIDATION
# ============================================================

def parse_mask(series):
    """Convert manifest mask values into Boolean values."""

    if series.dtype == bool:
        return series

    normalized = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    result = normalized.map(mapping)

    if result.isna().any():
        invalid = normalized[result.isna()].unique().tolist()

        raise ValueError(
            f"Invalid mask values: {invalid}"
        )

    return result.astype(bool)


def read_manifest():
    """
    Validate the leakage-reduced manifest.

    This function does not load image pixels.
    """

    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    manifest = pd.read_csv(MANIFEST)

    required_columns = {
        "archive_path",
        "new_split",
        "leakage_group_id",
        "species_id",
        "condition_id",
        "species_mask",
        "condition_mask",
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    manifest["new_split"] = (
        manifest["new_split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_splits = {"train", "validation", "test"}

    unexpected_splits = (
        set(manifest["new_split"].unique()) - allowed_splits
    )

    if unexpected_splits:
        raise ValueError(
            f"Unexpected split values: {unexpected_splits}"
        )

    if manifest["archive_path"].isna().any():
        raise ValueError(
            "Manifest contains missing archive_path values."
        )

    if manifest["archive_path"].duplicated().any():
        raise ValueError(
            "Manifest contains duplicate archive_path values. "
            "Inspect the manifest before training."
        )

    if manifest["leakage_group_id"].isna().any():
        raise ValueError(
            "Manifest contains missing leakage_group_id values."
        )

    # Every leakage group must belong to exactly one split.
    group_split_counts = (
        manifest.groupby("leakage_group_id")["new_split"]
        .nunique()
    )

    leaking_groups = group_split_counts[
        group_split_counts > 1
    ]

    if not leaking_groups.empty:
        raise ValueError(
            f"{len(leaking_groups)} leakage groups cross splits. "
            "Fix the manifest before training."
        )

    manifest["species_mask"] = parse_mask(
        manifest["species_mask"]
    )

    manifest["condition_mask"] = parse_mask(
        manifest["condition_mask"]
    )

    manifest["species_id"] = pd.to_numeric(
        manifest["species_id"],
        errors="coerce",
    )

    manifest["condition_id"] = pd.to_numeric(
        manifest["condition_id"],
        errors="coerce",
    )

    # Validate species labels wherever species supervision exists.
    species_rows = manifest["species_mask"]

    if manifest.loc[
        species_rows, "species_id"
    ].isna().any():
        raise ValueError(
            "A species-supervised row has a missing species_id."
        )

    species_values = manifest.loc[
        species_rows, "species_id"
    ]

    if not species_values.between(
        0, NUM_SPECIES - 1
    ).all():
        raise ValueError(
            "A species label is outside the valid class range."
        )

    # Validate condition labels wherever condition supervision exists.
    condition_rows = manifest["condition_mask"]

    if manifest.loc[
        condition_rows, "condition_id"
    ].isna().any():
        raise ValueError(
            "A condition-supervised row has a missing condition_id."
        )

    condition_values = manifest.loc[
        condition_rows, "condition_id"
    ]

    if not condition_values.between(
        0, NUM_CONDITIONS - 1
    ).all():
        raise ValueError(
            "A condition label is outside the valid class range."
        )

    manifest["species_id"] = (
        manifest["species_id"].fillna(-1).astype(int)
    )

    manifest["condition_id"] = (
        manifest["condition_id"].fillna(-1).astype(int)
    )

    return manifest


# ============================================================
# 4. TRAINING DATA AND BALANCED SAMPLER
# ============================================================

def get_training_rows(manifest):
    """Return only training rows."""

    train_rows = manifest[
        manifest["new_split"] == "train"
    ].copy()

    train_rows = train_rows.reset_index(drop=True)

    if train_rows.empty:
        raise ValueError(
            "No training rows found in the manifest."
        )

    return train_rows


def create_sampler(train_rows):
    """
    Create a condition-balanced weighted sampler.

    Rare condition classes receive greater sampling weight.
    Common classes receive lower sampling weight.

    Rows without a condition label receive a neutral weight.
    """

    labels = train_rows["condition_id"].to_numpy()
    masks = train_rows["condition_mask"].to_numpy(dtype=bool)

    valid_labels = labels[masks]

    if len(valid_labels) == 0:
        raise ValueError(
            "No valid condition labels found in training data."
        )

    counts = Counter(valid_labels.tolist())

    # Inverse-square-root weighting reduces imbalance without
    # being as aggressive as full inverse-frequency weighting.
    class_weights = {
        class_id: 1.0 / np.sqrt(count)
        for class_id, count in counts.items()
    }

    # Normalize weights so the median class weight is 1.
    median_weight = float(np.median(
        list(class_weights.values())
    ))

    class_weights = {
        class_id: min(
            weight / median_weight,
            MAX_WEIGHT,
        )
        for class_id, weight in class_weights.items()
    }

    sample_weights = []

    for label, has_condition in zip(labels, masks):
        if has_condition and label in class_weights:
            sample_weights.append(class_weights[label])
        else:
            sample_weights.append(1.0)

    sample_weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double,
    )

    print("\nCondition-label counts used by sampler:")

    for class_id in sorted(counts):
        print(
            f"  Class {class_id:2d}: "
            f"{counts[class_id]:7,d} images, "
            f"weight={class_weights[class_id]:.3f}"
        )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


# ============================================================
# 5. LOSS CALCULATION
# ============================================================

def calculate_loss(
    species_logits,
    condition_logits,
    batch,
    criterion,
):
    """
    Calculate loss only for labels with valid supervision.

    Species and condition supervision are handled independently.
    """

    species_targets = batch["species"].long()
    condition_targets = batch["condition"].long()

    species_mask = batch["species_mask"].bool()
    condition_mask = batch["condition_mask"].bool()

    device = species_logits.device

    species_mask = species_mask.to(device)
    condition_mask = condition_mask.to(device)

    species_targets = species_targets.to(device)
    condition_targets = condition_targets.to(device)

    loss = torch.zeros((), device=device)

    has_species = species_mask.any()
    has_condition = condition_mask.any()

    if has_species:
        valid_species = species_targets[species_mask]

        if (
            (valid_species < 0).any()
            or (valid_species >= NUM_SPECIES).any()
        ):
            raise ValueError(
                "Invalid species target encountered in a batch."
            )

        species_loss = criterion(
            species_logits[species_mask],
            valid_species,
        )

        loss = loss + species_loss

    if has_condition:
        valid_conditions = condition_targets[condition_mask]

        if (
            (valid_conditions < 0).any()
            or (valid_conditions >= NUM_CONDITIONS).any()
        ):
            raise ValueError(
                "Invalid condition target encountered in a batch."
            )

        condition_loss = criterion(
            condition_logits[condition_mask],
            valid_conditions,
        )

        loss = loss + condition_loss

    if not has_species and not has_condition:
        raise ValueError(
            "A batch contains no valid supervised labels."
        )

    return loss


# ============================================================
# 6. VALIDATION
# ============================================================

def evaluate(model, loader, device, criterion):
    """
    Evaluate on validation data only.

    The test split is not loaded by this script.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    species_true = []
    species_pred = []

    condition_true = []
    condition_pred = []

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Validation",
            leave=False,
        ):
            images = batch["image"].to(device)

            species_logits, condition_logits = model(images)

            loss = calculate_loss(
                species_logits,
                condition_logits,
                batch,
                criterion,
            )

            total_loss += float(loss.item())
            total_batches += 1

            species_targets = batch["species"].long()
            condition_targets = batch["condition"].long()

            species_mask = batch["species_mask"].bool()
            condition_mask = batch["condition_mask"].bool()

            predicted_species = species_logits.argmax(dim=1).cpu()
            predicted_conditions = condition_logits.argmax(dim=1).cpu()

            valid_species = species_mask & (
                (species_targets >= 0)
                & (species_targets < NUM_SPECIES)
            )

            valid_conditions = condition_mask & (
                (condition_targets >= 0)
                & (condition_targets < NUM_CONDITIONS)
            )

            species_true.extend(
                species_targets[valid_species].tolist()
            )

            species_pred.extend(
                predicted_species[valid_species].tolist()
            )

            condition_true.extend(
                condition_targets[valid_conditions].tolist()
            )

            condition_pred.extend(
                predicted_conditions[valid_conditions].tolist()
            )

    if total_batches == 0:
        raise ValueError(
            "Validation loader produced no batches."
        )

    species_f1 = (
        f1_score(
            species_true,
            species_pred,
            labels=list(range(NUM_SPECIES)),
            average="macro",
            zero_division=0,
        )
        if species_true
        else float("nan")
    )

    condition_f1 = (
        f1_score(
            condition_true,
            condition_pred,
            labels=list(range(NUM_CONDITIONS)),
            average="macro",
            zero_division=0,
        )
        if condition_true
        else float("nan")
    )

    return {
        "validation_loss": total_loss / total_batches,
        "species_macro_f1": species_f1,
        "condition_macro_f1": condition_f1,
        "species_label_count": len(species_true),
        "condition_label_count": len(condition_true),
    }


# ============================================================
# 7. CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
):
    """Save a checkpoint for this isolated experiment."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "num_species": NUM_SPECIES,
        "num_conditions": NUM_CONDITIONS,
        "image_size": IMAGE_SIZE,
        "initialization": "ImageNet-pretrained ResNet18",
        "manifest": str(MANIFEST),
        "seed": SEED,
    }

    torch.save(checkpoint, path)


# ============================================================
# 8. MAIN TRAINING FUNCTION
# ============================================================

def main():
    set_seed()

    # Ensure that new experiment outputs cannot overwrite
    # files from an earlier experiment.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SMOKE_TEST:
        for output_path in (
            CHECKPOINT_PATH,
            BEST_CHECKPOINT_PATH,
            HISTORY_PATH,
        ):
            if output_path.exists():
                raise FileExistsError(
                    f"Output already exists: {output_path}\n"
                    "Choose new output filenames before training."
                )

    print("=" * 66)
    print("SMART PLANT DOCTOR")
    print("LEAKAGE-REDUCED TRAINING EXPERIMENT")
    print("=" * 66)

    print(f"Manifest: {MANIFEST}")

    # Read metadata for validation; this does not load test images.
    manifest = read_manifest()

    split_counts = manifest["new_split"].value_counts()

    train_count = int(split_counts.get("train", 0))
    validation_count = int(split_counts.get("validation", 0))
    test_count = int(split_counts.get("test", 0))

    if train_count == 0 or validation_count == 0:
        raise ValueError(
            "Training and validation splits must both be nonempty."
        )

    print(f"Training images in manifest: {train_count:,}")
    print(f"Validation images in manifest: {validation_count:,}")
    print(f"Test images in manifest: {test_count:,}")
    print("Test images loaded: NO")

    # The dataset class reads only the requested split.
    train_dataset = PlantDataset(split="train")
    validation_dataset = PlantDataset(split="validation")

    if len(train_dataset) != train_count:
        raise ValueError(
            "Training dataset length does not match the manifest: "
            f"{len(train_dataset)} versus {train_count}."
        )

    if len(validation_dataset) != validation_count:
        raise ValueError(
            "Validation dataset length does not match the manifest: "
            f"{len(validation_dataset)} versus "
            f"{validation_count}."
        )

    train_rows = get_training_rows(manifest)
    sampler = create_sampler(train_rows)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    device = get_device()

    print(f"\nDevice: {device}")
    print(f"Training dataset: {len(train_dataset):,}")
    print(f"Validation dataset: {len(validation_dataset):,}")
    print("Starting checkpoint: FRESH MODEL")
    print("Initialization: ImageNet-pretrained ResNet18")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    if SMOKE_TEST:
        print(
            "\nMODE: smoke test; "
            f"{SMOKE_TEST_BATCHES} training batches, "
            "no checkpoint or history saved."
        )
    else:
        print(f"\nMODE: full training; {EPOCHS} epoch(s)")

    # --------------------------------------------------------
    # Initialize a fresh model.
    # This intentionally does not call load_model().
    # --------------------------------------------------------

    model = ResNetAgriculturalAI(
        num_species=NUM_SPECIES,
        num_conditions=NUM_CONDITIONS,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Use a lower learning rate for the pretrained backbone
    # and a higher rate for the newly initialized task heads.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": BACKBONE_LR,
            },
            {
                "params": model.species_head.parameters(),
                "lr": HEAD_LR,
            },
            {
                "params": model.condition_head.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_validation_loss = float("inf")

    epochs_to_run = 1 if SMOKE_TEST else EPOCHS

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, epochs_to_run + 1):
        model.train()

        running_loss = 0.0
        batches_seen = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs_to_run}",
        )

        for batch in progress:
            images = batch["image"].to(device)

            optimizer.zero_grad(set_to_none=True)

            species_logits, condition_logits = model(images)

            loss = calculate_loss(
                species_logits,
                condition_logits,
                batch,
                criterion,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP,
            )

            optimizer.step()

            running_loss += float(loss.item())
            batches_seen += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

            if (
                SMOKE_TEST
                and batches_seen >= SMOKE_TEST_BATCHES
            ):
                break

        if batches_seen == 0:
            raise RuntimeError(
                "No training batches were processed."
            )

        mean_training_loss = running_loss / batches_seen

        print(
            f"\nEpoch {epoch} training loss: "
            f"{mean_training_loss:.4f}"
        )

        # The smoke test checks that forward/backward/optimizer
        # operations work. It does not run full validation.
        if SMOKE_TEST:
            print("\nSmoke test passed.")
            print(f"Training batches processed: {batches_seen}")
            print(
                f"Mean training loss: {mean_training_loss:.4f}"
            )
            print("No checkpoint or history was saved.")
            return

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            criterion,
        )

        print(
            f"Validation loss: "
            f"{validation_metrics['validation_loss']:.4f}"
        )

        print(
            f"Species macro F1: "
            f"{validation_metrics['species_macro_f1']:.4f}"
        )

        print(
            f"Condition macro F1: "
            f"{validation_metrics['condition_macro_f1']:.4f}"
        )

        row = {
            "epoch": epoch,
            "training_loss": mean_training_loss,
            **validation_metrics,
        }

        history.append(row)

        # Save latest checkpoint.
        save_checkpoint(
            CHECKPOINT_PATH,
            model,
            optimizer,
            epoch,
            validation_metrics,
        )

        # Save best checkpoint according to validation loss.
        current_validation_loss = (
            validation_metrics["validation_loss"]
        )

        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss

            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                model,
                optimizer,
                epoch,
                validation_metrics,
            )

        # Write experiment history.
        pd.DataFrame(history).to_csv(
            HISTORY_PATH,
            index=False,
        )

    print("\nTraining finished.")
    print(f"Latest checkpoint: {CHECKPOINT_PATH}")
    print(f"Best validation checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()