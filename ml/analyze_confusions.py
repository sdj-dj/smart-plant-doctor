from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MATRIX_FILE = RESULTS / "balanced_test_condition_confusion_matrix.csv"
REPORT_FILE = RESULTS / "balanced_test_condition_report.csv"

OUTPUT_FILE = RESULTS / "condition_confusion_analysis.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("SMART PLANT DOCTOR — CONFUSION ANALYSIS")
print("=" * 75)

if not MATRIX_FILE.exists():
    raise FileNotFoundError(
        f"Missing confusion matrix: {MATRIX_FILE}\n"
        "Run: python -m ml.evaluate_test"
    )

if not REPORT_FILE.exists():
    raise FileNotFoundError(
        f"Missing classification report: {REPORT_FILE}"
    )

matrix = pd.read_csv(MATRIX_FILE, index_col=0)
report = pd.read_csv(REPORT_FILE, index_col=0)

matrix.index = matrix.index.astype(str).str.strip()
matrix.columns = matrix.columns.astype(str).str.strip()
report.index = report.index.astype(str).str.strip()

# Ensure the confusion matrix contains numeric values.
matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0)

# Restrict analysis to actual condition classes.
summary_rows = {
    "accuracy",
    "macro avg",
    "weighted avg",
    "micro avg",
    "samples avg",
}

report = report.loc[
    ~report.index.str.lower().isin(summary_rows)
]


# ============================================================
# ANALYZE EACH CONDITION
# ============================================================

records = []

for condition in matrix.index:

    actual_count = int(matrix.loc[condition].sum())

    # No actual examples means this class cannot be evaluated.
    if actual_count == 0:
        records.append({
            "condition": condition,
            "actual_test_images": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "recall": None,
            "f1_score": (
                report.loc[condition, "f1-score"]
                if condition in report.index
                else None
            ),
            "most_common_confusion": "NOT EVALUABLE",
            "confused_prediction_count": 0,
        })
        continue

    correct = int(matrix.loc[condition, condition])
    incorrect = actual_count - correct
    recall = correct / actual_count

    # Find the incorrect predictions for this actual class.
    incorrect_predictions = matrix.loc[condition].drop(
        labels=[condition],
        errors="ignore",
    )

    incorrect_predictions = incorrect_predictions[
        incorrect_predictions > 0
    ].sort_values(ascending=False)

    if len(incorrect_predictions) > 0:
        most_common_confusion = str(incorrect_predictions.index[0])
        confusion_count = int(incorrect_predictions.iloc[0])
    else:
        most_common_confusion = "None"
        confusion_count = 0

    f1 = (
        float(report.loc[condition, "f1-score"])
        if condition in report.index
        else None
    )

    records.append({
        "condition": condition,
        "actual_test_images": actual_count,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "recall": recall,
        "f1_score": f1,
        "most_common_confusion": most_common_confusion,
        "confused_prediction_count": confusion_count,
    })


# ============================================================
# SAVE RESULTS
# ============================================================

analysis = pd.DataFrame(records)

analysis.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# DISPLAY THE WEAKEST EVALUABLE CONDITIONS
# ============================================================

evaluable = analysis[
    analysis["actual_test_images"] > 0
].copy()

evaluable = evaluable.sort_values(
    ["f1_score", "incorrect_predictions"],
    ascending=[True, False],
)

print("\n1. TEN WEAKEST EVALUABLE CONDITIONS")
print("-" * 75)

for _, row in evaluable.head(10).iterrows():

    print(f"\nCondition: {row['condition']}")
    print(f"  Test images: {row['actual_test_images']}")
    print(f"  Correct: {row['correct_predictions']}")
    print(f"  Incorrect: {row['incorrect_predictions']}")
    print(f"  Recall: {row['recall']:.4f}")

    if pd.notna(row["f1_score"]):
        print(f"  F1 score: {row['f1_score']:.4f}")

    print(
        f"  Most common confusion: "
        f"{row['most_common_confusion']}"
    )

    print(
        f"  Confusion count: "
        f"{row['confused_prediction_count']}"
    )


# ============================================================
# DISPLAY ALL INCORRECT PREDICTION PAIRS
# ============================================================

print("\n\n2. MOST COMMON CONDITION CONFUSIONS")
print("-" * 75)

confusions = []

for actual in matrix.index:

    for predicted in matrix.columns:

        if actual == predicted:
            continue

        count = int(matrix.loc[actual, predicted])

        if count > 0:
            confusions.append({
                "actual_condition": actual,
                "predicted_condition": predicted,
                "incorrect_count": count,
            })

if confusions:

    confusion_df = pd.DataFrame(confusions)

    confusion_df = confusion_df.sort_values(
        "incorrect_count",
        ascending=False,
    )

    for _, row in confusion_df.head(20).iterrows():
        print(
            f"Actual: {row['actual_condition']}"
            f" -> Predicted: {row['predicted_condition']}"
            f" | Incorrect images: {row['incorrect_count']}"
        )

else:
    print("No incorrect condition predictions found.")


# ============================================================
# SUMMARY
# ============================================================

print("\n\n3. SUMMARY")
print("-" * 75)

print(f"Condition classes: {len(analysis)}")
print(f"Evaluable classes: {len(evaluable)}")
print(
    "Classes with at least one incorrect prediction:",
    int((evaluable["incorrect_predictions"] > 0).sum()),
)

not_evaluable = analysis[
    analysis["actual_test_images"] == 0
]["condition"].tolist()

print("Classes with no test examples:", not_evaluable or "None")

print(f"\nFull report saved to:\n{OUTPUT_FILE}")
print("\nAnalysis complete.")
print("=" * 75)