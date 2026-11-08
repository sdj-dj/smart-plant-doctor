from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "test_condition_report.csv"
OUTPUT = ROOT / "results" / "condition_improvement_plan.csv"


def main():
    if not REPORT.exists():
        raise FileNotFoundError(
            f"Test report not found: {REPORT}\n"
            "Run python -m ml.evaluate_test first."
        )

    df = pd.read_csv(REPORT, index_col=0)

    required = {"precision", "recall", "f1-score", "support"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in test report: {sorted(missing)}"
        )

    # Keep actual condition classes, excluding summary rows.
    summary_rows = {
        "accuracy",
        "macro avg",
        "weighted avg",
        "micro avg",
    }

    df = df.loc[~df.index.astype(str).isin(summary_rows)].copy()

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["f1-score", "support"])
    df["support"] = df["support"].astype(int)

    # A class with zero support cannot be evaluated on this test set.
    df["status"] = "Review"
    df.loc[df["support"] == 0, "status"] = "No test examples"
    df.loc[
        (df["support"] > 0) & (df["f1-score"] >= 0.90),
        "status",
    ] = "Strong"
    df.loc[
        (df["support"] > 0)
        & (df["f1-score"] >= 0.75)
        & (df["f1-score"] < 0.90),
        "status",
    ] = "Moderate"
    df.loc[
        (df["support"] > 0) & (df["f1-score"] < 0.75),
        "status",
    ] = "Priority"

    # A simple priority score: low F1 matters more, while support
    # helps distinguish repeated problems from tiny test samples.
    df["priority_score"] = (
        (1 - df["f1-score"].clip(0, 1))
        * (df["support"] / (df["support"] + 20))
    )

    df = df.sort_values(
        ["priority_score", "support"],
        ascending=[False, False],
    )

    df.to_csv(OUTPUT, index_label="condition")

    print("=" * 65)
    print("SMART PLANT DOCTOR - CONDITION IMPROVEMENT PLAN")
    print("=" * 65)

    print("\nClass status summary:")
    print(df["status"].value_counts().to_string())

    print("\nHighest-priority classes:")
    print(
        df[
            [
                "precision",
                "recall",
                "f1-score",
                "support",
                "status",
            ]
        ].head(15).to_string()
    )

    print(f"\nFull report saved to: {OUTPUT}")

    print("\nRecommended interpretation:")
    print("- Priority: inspect mistakes and training examples first.")
    print("- Moderate: investigate confusing classes.")
    print("- Strong: monitor during future experiments.")
    print("- No test examples: collect labelled evaluation data.")

    print(
        "\nThis analysis does not retrain the model or modify checkpoints."
    )


if __name__ == "__main__":
    main()