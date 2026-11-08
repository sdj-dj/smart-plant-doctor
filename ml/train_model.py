# ml/train_model.py

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

from ml.dataset import PlantDataset
from ml.model import NUM_SPECIES, NUM_CONDITIONS, load_model


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

BASELINE_CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "resnet18_agricultural_ai_step1.pth"
)

EXPERIMENT_DIR = PROJECT_DIR / "checkpoints" / "experiments"
RESULTS_DIR = PROJECT_DIR / "results"

MANIFEST_PATH = PROJECT_DIR / "audit" / "training_manifest.csv"

IMAGE_SIZE = 224
SEED = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DEVICE SELECTION
# ============================================================

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 4. IMAGE TRANSFORMATIONS
# ============================================================

def get_training_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# 5. CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    manifest_path,
    label_column,
    mask_column,
    num_classes,
    device,
):
    manifest = pd.read_csv(manifest_path)

    # Use only valid labels from the training split.
    valid_mask = (
        manifest["split"].eq("train")
        & manifest[mask_column].astype(str).str.lower().eq("true")
    )

    labels = pd.to_numeric(
        manifest.loc[valid_mask, label_column],
        errors="coerce",
    )

    labels = labels[
        labels.notna()
        & (labels >= 0)
        & (labels < num_classes)
    ].astype(int)

    counts = np.bincount(
        labels.to_numpy(),
        minlength=num_classes,
    ).astype(np.float64)

    weights = np.ones(num_classes, dtype=np.float32)
    valid_classes = counts > 0

    if valid_classes.any():
        # Moderate inverse-frequency weighting.
        weights[valid_classes] = (
            1.0 / np.sqrt(counts[valid_classes])
        )

        median_weight = np.median(weights[valid_classes])

        weights[valid_classes] = np.minimum(
            weights[valid_classes],
            median_weight * 3.0,
        )

        weights[valid_classes] /= weights[valid_classes].mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# 6. MASKED MULTI-TASK LOSS
# ============================================================

def calculate_loss(
    outputs: tuple,
    batch: dict,
    species_criterion,
    condition_criterion,
    device,
):
    # IMPORTANT:
    # ResNetAgriculturalAI returns a tuple:
    # (species_logits, condition_logits)
    species_logits, condition_logits = outputs

    species_targets = batch["species"].to(
        device,
        dtype=torch.long,
    )

    condition_targets = batch["condition"].to(
        device,
        dtype=torch.long,
    )

    species_mask = batch["species_mask"].to(
        device,
        dtype=torch.bool,
    )

    condition_mask = batch["condition_mask"].to(
        device,
        dtype=torch.bool,
    )

    # Exclude invalid labels even if their masks are incorrect.
    species_mask = (
        species_mask
        & (species_targets >= 0)
        & (species_targets < NUM_SPECIES)
    )

    condition_mask = (
        condition_mask
        & (condition_targets >= 0)
        & (condition_targets < NUM_CONDITIONS)
    )

    # Differentiable zero losses if a batch has no valid labels.
    species_loss = species_logits.sum() * 0.0
    condition_loss = condition_logits.sum() * 0.0

    if species_mask.any():
        species_loss = species_criterion(
            species_logits[species_mask],
            species_targets[species_mask],
        )

    if condition_mask.any():
        condition_loss = condition_criterion(
            condition_logits[condition_mask],
            condition_targets[condition_mask],
        )

    # Both tasks contribute to the training objective.
    total_loss = species_loss + condition_loss

    return (
        total_loss,
        species_loss.detach(),
        condition_loss.detach(),
    )


# ============================================================
# 7. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    species_criterion,
    condition_criterion,
    device,
    epoch,
    max_batches=None,
):
    model.train()

    total_loss_sum = 0.0
    species_loss_sum = 0.0
    condition_loss_sum = 0.0
    batches_processed = 0

    progress = tqdm(
        loader,
        desc=f"Training epoch {epoch}",
    )

    for batch_index, batch in enumerate(progress):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = batch["image"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # IMPORTANT: model returns a tuple, not a dictionary.
        species_logits, condition_logits = model(images)

        # Package the two outputs into a tuple.
        outputs = (species_logits, condition_logits)

        total_loss, species_loss, condition_loss = calculate_loss(
            outputs=outputs,
            batch=batch,
            species_criterion=species_criterion,
            condition_criterion=condition_criterion,
            device=device,
        )

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss at batch {batch_index}: "
                f"{total_loss.item()}"
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss_sum += total_loss.item()
        species_loss_sum += species_loss.item()
        condition_loss_sum += condition_loss.item()

        batches_processed += 1

        progress.set_postfix(
            loss=f"{total_loss.item():.4f}",
            species=f"{species_loss.item():.4f}",
            condition=f"{condition_loss.item():.4f}",
        )

    if batches_processed == 0:
        raise RuntimeError("No training batches were processed.")

    return {
        "train_loss": total_loss_sum / batches_processed,
        "train_species_loss": species_loss_sum / batches_processed,
        "train_condition_loss": condition_loss_sum / batches_processed,
        "train_batches": batches_processed,
    }


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    max_batches=None,
):
    model.eval()

    species_true = []
    species_pred = []

    condition_true = []
    condition_pred = []

    batches_processed = 0

    progress = tqdm(
        loader,
        desc="Validating",
    )

    for batch_index, batch in enumerate(progress):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = batch["image"].to(device)

        # IMPORTANT: unpack the tuple returned by the model.
        species_logits, condition_logits = model(images)

        species_targets = batch["species"].to(
            device,
            dtype=torch.long,
        )

        condition_targets = batch["condition"].to(
            device,
            dtype=torch.long,
        )

        species_mask = batch["species_mask"].to(
            device,
            dtype=torch.bool,
        )

        condition_mask = batch["condition_mask"].to(
            device,
            dtype=torch.bool,
        )

        species_mask = (
            species_mask
            & (species_targets >= 0)
            & (species_targets < NUM_SPECIES)
        )

        condition_mask = (
            condition_mask
            & (condition_targets >= 0)
            & (condition_targets < NUM_CONDITIONS)
        )

        # Species predictions.
        if species_mask.any():
            species_true.extend(
                species_targets[species_mask].cpu().tolist()
            )

            species_pred.extend(
                species_logits[species_mask]
                .argmax(dim=1)
                .cpu()
                .tolist()
            )

        # Condition predictions.
        if condition_mask.any():
            condition_true.extend(
                condition_targets[condition_mask].cpu().tolist()
            )

            condition_pred.extend(
                condition_logits[condition_mask]
                .argmax(dim=1)
                .cpu()
                .tolist()
            )

        batches_processed += 1

    if batches_processed == 0:
        raise RuntimeError("No validation batches were processed.")

    def calculate_metrics(y_true, y_pred, num_classes):
        if not y_true:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "samples": 0,
            }

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            ),
            "samples": len(y_true),
        }

    return {
        "species": calculate_metrics(
            species_true,
            species_pred,
            NUM_SPECIES,
        ),
        "condition": calculate_metrics(
            condition_true,
            condition_pred,
            NUM_CONDITIONS,
        ),
        "validation_batches": batches_processed,
    }


# ============================================================
# 9. SAVE EXPERIMENTAL CHECKPOINTS
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    destination,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": metrics,
            "num_species": NUM_SPECIES,
            "num_conditions": NUM_CONDITIONS,
            "image_size": IMAGE_SIZE,
        },
        destination,
    )


# ============================================================
# 10. SAVE TRAINING HISTORY
# ============================================================

def append_history(path, record):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(record.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)


# ============================================================
# 11. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Smart Plant Doctor."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.workers < 0:
        parser.error("--workers cannot be negative.")

    set_seed(args.seed)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("SMART PLANT DOCTOR — TRAINING PIPELINE")
    print("=" * 65)

    print(f"Project: {PROJECT_DIR}")
    print(f"Baseline checkpoint: {BASELINE_CHECKPOINT}")

    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}"
        )

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    device = select_device()

    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Smoke test: {args.smoke_test}")

    # --------------------------------------------------------
    # Load the original model.
    # --------------------------------------------------------

    print("\nLoading existing ResNet-18 checkpoint...")

    model, _ = load_model(str(BASELINE_CHECKPOINT))
    model = model.to(device)

    print("Existing checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Load datasets.
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = PlantDataset(
        split="train",
        transform=get_training_transform(),
    )

    validation_dataset = PlantDataset(
        split="validation",
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Class weighting.
    # --------------------------------------------------------

    print("\nCalculating class weights...")

    species_weights = calculate_class_weights(
        MANIFEST_PATH,
        "species_id",
        "species_mask",
        NUM_SPECIES,
        device,
    )

    condition_weights = calculate_class_weights(
        MANIFEST_PATH,
        "condition_id",
        "condition_mask",
        NUM_CONDITIONS,
        device,
    )

    species_criterion = nn.CrossEntropyLoss(
        weight=species_weights,
    )

    condition_criterion = nn.CrossEntropyLoss(
        weight=condition_weights,
    )

    # --------------------------------------------------------
    # Optimizer with separate learning rates.
    # --------------------------------------------------------

    backbone_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if "backbone" in name:
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.lr_backbone,
            },
            {
                "params": head_parameters,
                "lr": args.lr_head,
            },
        ],
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Train and evaluate.
    # --------------------------------------------------------

    history_path = RESULTS_DIR / "training_history.csv"

    best_score = float("-inf")

    epochs_to_run = 1 if args.smoke_test else args.epochs
    max_batches = 2 if args.smoke_test else None

    for epoch in range(1, epochs_to_run + 1):

        start_time = time.time()

        print("\n" + "-" * 65)
        print(f"EPOCH {epoch}/{epochs_to_run}")
        print("-" * 65)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            species_criterion=species_criterion,
            condition_criterion=condition_criterion,
            device=device,
            epoch=epoch,
            max_batches=max_batches,
        )

        validation_metrics = evaluate(
            model=model,
            loader=validation_loader,
            device=device,
            max_batches=max_batches,
        )

        species_metrics = validation_metrics["species"]
        condition_metrics = validation_metrics["condition"]

        balanced_score = (
            species_metrics["macro_f1"]
            + condition_metrics["macro_f1"]
        ) / 2.0

        elapsed = time.time() - start_time

        print("\nTRAINING RESULTS")
        print(f"Training loss: {train_metrics['train_loss']:.4f}")
        print(
            f"Species loss: "
            f"{train_metrics['train_species_loss']:.4f}"
        )
        print(
            f"Condition loss: "
            f"{train_metrics['train_condition_loss']:.4f}"
        )

        print("\nVALIDATION RESULTS")
        print(
            f"Species accuracy: "
            f"{species_metrics['accuracy']:.4f}"
        )
        print(
            f"Species macro F1: "
            f"{species_metrics['macro_f1']:.4f}"
        )
        print(
            f"Condition accuracy: "
            f"{condition_metrics['accuracy']:.4f}"
        )
        print(
            f"Condition macro F1: "
            f"{condition_metrics['macro_f1']:.4f}"
        )

        print(f"Balanced macro F1: {balanced_score:.4f}")
        print(f"Duration: {elapsed / 60:.2f} minutes")

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            **train_metrics,
            "species_accuracy": species_metrics["accuracy"],
            "species_macro_f1": species_metrics["macro_f1"],
            "species_validation_samples": species_metrics["samples"],
            "condition_accuracy": condition_metrics["accuracy"],
            "condition_macro_f1": condition_metrics["macro_f1"],
            "condition_validation_samples": condition_metrics["samples"],
            "balanced_macro_f1": balanced_score,
            "duration_seconds": round(elapsed, 2),
            "smoke_test": args.smoke_test,
        }

        append_history(history_path, record)

        if args.smoke_test:
            print("\nSmoke test completed. No checkpoint was saved.")
            continue

        save_checkpoint(
            model,
            optimizer,
            epoch,
            validation_metrics,
            EXPERIMENT_DIR / "smart_plant_doctor_last.pth",
        )

        if balanced_score > best_score:
            best_score = balanced_score

            save_checkpoint(
                model,
                optimizer,
                epoch,
                validation_metrics,
                EXPERIMENT_DIR / "smart_plant_doctor_best.pth",
            )

            print("Best experimental checkpoint saved.")

    print("\n" + "=" * 65)
    print("TRAINING PIPELINE FINISHED")
    print("=" * 65)

    print(f"History: {history_path}")

    if not args.smoke_test:
        print(f"Experimental checkpoints: {EXPERIMENT_DIR}")

    print("Original baseline checkpoint remains untouched.")


if __name__ == "__main__":
    main()