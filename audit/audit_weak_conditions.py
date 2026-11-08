from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "audit" / "training_manifest.csv"
OUTPUT_DIR = ROOT / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEAK_CONDITIONS = [
    "Sogatella",
    "Potassium deficiency",
    "Nitrogen deficiency",
    "Bird eye spot",
    "Canker",
]


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    required = {
        "archive_path",
        "split",
        "condition",
        "condition_id",
        "condition_mask",
        "group_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["condition"] = df["condition"].astype("string").str.strip()
    df["condition_mask"] = (
        df["condition_mask"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    print("=" * 70)
    print("SMART PLANT DOCTOR - WEAK CONDITION DATA AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Class counts across splits
    # ---------------------------------------------------------

    class_counts = (
        df[df["condition"].isin(WEAK_CONDITIONS)]
        .groupby(["condition", "split"])
        .agg(
            image_count=("archive_path", "size"),
            valid_label_count=("condition_mask", "sum"),
            unique_groups=("group_id", "nunique"),
        )
        .reset_index()
    )

    counts_path = OUTPUT_DIR / "weak_condition_split_counts.csv"
    class_counts.to_csv(counts_path, index=False)

    print("\n1. COUNTS BY CONDITION AND SPLIT")
    if class_counts.empty:
        print("No matching condition names found in the manifest.")
        print("Actual condition names:")
        print(
            df["condition"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .to_string(index=False)
        )
        return

    print(class_counts.to_string(index=False))
    print(f"\nSaved: {counts_path}")

    # ---------------------------------------------------------
    # 2. Label ID consistency
    # ---------------------------------------------------------

    print("\n2. CONDITION NAME / ID CONSISTENCY")

    selected = df[df["condition"].isin(WEAK_CONDITIONS)]

    consistency = (
        selected.groupby("condition")
        .agg(
            distinct_ids=("condition_id", "nunique"),
            condition_ids=(
                "condition_id",
                lambda values: ", ".join(
                    sorted(set(values.dropna().astype(str)))
                ),
            ),
            masked_rows=("condition_mask", lambda values: (~values).sum()),
        )
        .reset_index()
    )

    print(consistency.to_string(index=False))

    consistency.to_csv(
        OUTPUT_DIR / "weak_condition_label_consistency.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Export example paths for manual image review
    # ---------------------------------------------------------

    print("\n3. TRAINING EXAMPLES FOR INSPECTION")

    examples = df[
        df["condition"].isin(WEAK_CONDITIONS)
        & (df["split"] == "train")
    ].copy()

    examples["condition_mask"] = examples["condition_mask"].astype(bool)

    examples = examples.sort_values(
        ["condition", "condition_mask", "archive_path"],
        ascending=[True, False, True],
    )

    # Export up to 30 examples per condition, prioritizing valid labels.
    examples = (
        examples.groupby("condition", group_keys=False)
        .head(30)
        .reset_index(drop=True)
    )

    example_path = OUTPUT_DIR / "weak_condition_training_examples.csv"

    columns = [
        "condition",
        "condition_id",
        "condition_mask",
        "split",
        "archive_path",
        "group_id",
    ]

    examples[columns].to_csv(example_path, index=False)

    for condition in WEAK_CONDITIONS:
        count = int((examples["condition"] == condition).sum())
        print(f"{condition}: {count} exported examples")

    print(f"\nExample paths saved: {example_path}")

    # ---------------------------------------------------------
    # 4. Rare class warning
    # ---------------------------------------------------------

    print("\n4. RECOMMENDATIONS")

    train_counts = (
        df[
            (df["split"] == "train")
            & df["condition"].isin(WEAK_CONDITIONS)
            & df["condition_mask"]
        ]
        .groupby("condition")
        .size()
    )

    for condition in WEAK_CONDITIONS:
        count = int(train_counts.get(condition, 0))

        if count < 50:
            print(
                f"DATA WARNING: {condition} has only {count} "
                "valid training examples. Collecting more correctly "
                "labelled images should be considered."
            )
        else:
            print(
                f"{condition}: {count} valid training examples. "
                "Inspect the images and confusion patterns."
            )

    print("\nAudit completed. No model or dataset files were modified.")


if __name__ == "__main__":
    main()