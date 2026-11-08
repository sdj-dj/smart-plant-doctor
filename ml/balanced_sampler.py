from pathlib import Path
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "audit" / "training_manifest.csv"

BATCH_SIZE = 16
NUM_SAMPLES = 152590
MAX_WEIGHT = 4.0


def main():
    df = pd.read_csv(MANIFEST)

    train = df[df["split"].astype(str).str.lower() == "train"].copy()

    valid = (
        train["condition_mask"].astype(str).str.lower()
        .isin(["true", "1", "yes"])
        & (pd.to_numeric(train["condition_id"], errors="coerce") >= 0)
    )

    labels = pd.to_numeric(
        train["condition_id"], errors="coerce"
    ).fillna(-1).astype(int)

    counts = Counter(labels[valid].tolist())

    if not counts:
        raise ValueError("No valid training condition labels found.")

    # Square-root inverse frequency: gives rare classes more chances
    # without the extreme weights of full inverse frequency.
    class_weights = {
        class_id: 1.0 / (count ** 0.5)
        for class_id, count in counts.items()
    }

    median_weight = sorted(class_weights.values())[
        len(class_weights) // 2
    ]

    class_weights = {
        class_id: min(weight / median_weight, MAX_WEIGHT)
        for class_id, weight in class_weights.items()
    }

    # Invalid/missing condition labels receive neutral weight.
    sample_weights = []
    for label, is_valid in zip(labels.tolist(), valid.tolist()):
        if is_valid and label in class_weights:
            sample_weights.append(class_weights[label])
        else:
            sample_weights.append(1.0)

    weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=NUM_SAMPLES,
        replacement=True,
    )

    print("=" * 60)
    print("CONDITION-BALANCED SAMPLING PREVIEW")
    print("=" * 60)
    print(f"Training rows: {len(train):,}")
    print(f"Valid condition labels: {int(valid.sum()):,}")
    print(f"Condition classes: {len(counts)}")
    print(f"Samples per simulated epoch: {NUM_SAMPLES:,}")

    print("\nWeak-class sampling weights:")
    for class_id in [5, 11, 26, 27, 31]:
        count = counts.get(class_id, 0)
        weight = class_weights.get(class_id, 1.0)
        print(
            f"Condition ID {class_id:2}: "
            f"{count:5,} images | relative weight {weight:.3f}"
        )

    # Simulate an epoch to show the expected sampling distribution.
    sampled_indices = list(iter(sampler))
    sampled_labels = labels.iloc[sampled_indices].tolist()
    sampled_valid = valid.iloc[sampled_indices].tolist()

    sampled_counts = Counter(
        label
        for label, is_valid in zip(sampled_labels, sampled_valid)
        if is_valid
    )

    print("\nSimulated counts for weak conditions:")
    for class_id in [5, 11, 26, 27, 31]:
        print(
            f"Condition ID {class_id:2}: "
            f"{sampled_counts.get(class_id, 0):,} sampled examples"
        )

    print("\nPreview completed. No model was trained or modified.")
    print(
        "\nImportant: use this sampler only after integrating it into "
        "the training DataLoader. Do not combine it blindly with strong "
        "class-weighted loss, as that can overemphasize rare classes."
    )


if __name__ == "__main__":
    main()