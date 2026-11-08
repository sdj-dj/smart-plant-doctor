from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PREVIOUS_REPORT = RESULTS / "previous_test_condition_report.csv"
BALANCED_REPORT = RESULTS / "balanced_test_condition_report.csv"

OUTPUT_FILE = RESULTS / "condition_class_improvement.csv"


# ============================================================
# LOAD REPORTS
# ============================================================

print("\n" + "=" * 75)
print("SMART PLANT DOCTOR — CONDITION PERFORMANCE ANALYSIS")
print("=" * 75)

for file_path in [PREVIOUS_REPORT, BALANCED_REPORT]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"\nRequired report not found: {file_path}\n"
            "Run: python -m ml.evaluate_test"
        )


previous = pd.read_csv(PREVIOUS_REPORT, index_col=0)
balanced = pd.read_csv(BALANCED_REPORT, index_col=0)


# ============================================================
# CLEAN REPORTS
# ============================================================

def clean_report(report, model_name):
    report = report.copy()

    report.index = report.index.astype(str).str.strip()

    required_columns = ["f1-score", "support"]

    for column in required_columns:
        if column not in report.columns:
            raise ValueError(
                f"{model_name} report is missing column: {column}"
            )

    # Keep actual classes; remove summary rows.
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

    report = report[["f1-score", "support"]].copy()

    report = report.rename(
        columns={
            "f1-score": f"{model_name}_f1",
            "support": f"{model_name}_support",
        }
    )

    return report


previous = clean_report(previous, "previous")
balanced = clean_report(balanced, "balanced")


# ============================================================
# COMPARE EACH CONDITION
# ============================================================

comparison = previous.join(
    balanced,
    how="outer",
)

comparison["f1_change"] = (
    comparison["balanced_f1"] - comparison["previous_f1"]
)

comparison["f1_change_percentage_points"] = (
    comparison["f1_change"] * 100
)

comparison["status"] = comparison["f1_change"].apply(
    lambda value: (
        "Improved" if value > 0.000001
        else "Declined" if value < -0.000001
        else "Unchanged"
    )
)

comparison = comparison.sort_values(
    "balanced_f1",
    ascending=True,
)

comparison.to_csv(OUTPUT_FILE)


# ============================================================
# SHOW WEAKEST CONDITIONS
# ============================================================

print("\n1. TEN WEAKEST CONDITIONS IN THE BALANCED MODEL")
print("-" * 75)

weakest = comparison.head(10)

for condition, row in weakest.iterrows():
    f1 = row["balanced_f1"]
    support = row["balanced_support"]
    previous_f1 = row["previous_f1"]
    change = row["f1_change_percentage_points"]

    if pd.isna(f1):
        f1_text = "N/A"
    else:
        f1_text = f"{f1:.4f}"

    if pd.isna(support):
        support_text = "N/A"
    else:
        support_text = str(int(support))

    if pd.isna(previous_f1):
        previous_text = "N/A"
    else:
        previous_text = f"{previous_f1:.4f}"

    if pd.isna(change):
        change_text = "N/A"
    else:
        change_text = f"{change:+.2f} percentage points"

    print(f"\nCondition: {condition}")
    print(f"  Balanced F1: {f1_text}")
    print(f"  Test examples: {support_text}")
    print(f"  Previous F1: {previous_text}")
    print(f"  Change: {change_text}")


# ============================================================
# BIGGEST IMPROVEMENTS
# ============================================================

print("\n\n2. FIVE BIGGEST IMPROVEMENTS")
print("-" * 75)

valid_changes = comparison.dropna(subset=["f1_change"])

for condition, row in valid_changes.sort_values(
    "f1_change",
    ascending=False,
).head(5).iterrows():

    print(
        f"{condition}: "
        f"{row['f1_change_percentage_points']:+.2f} "
        f"percentage points "
        f"(F1 {row['previous_f1']:.4f} -> "
        f"{row['balanced_f1']:.4f})"
    )


# ============================================================
# BIGGEST DECLINES
# ============================================================

print("\n\n3. CONDITIONS THAT DECLINED")
print("-" * 75)

declined = valid_changes[
    valid_changes["f1_change"] < -0.000001
].sort_values("f1_change")

if declined.empty:
    print("No condition classes declined.")
else:
    for condition, row in declined.iterrows():
        print(
            f"{condition}: "
            f"{row['f1_change_percentage_points']:+.2f} "
            f"percentage points "
            f"(F1 {row['previous_f1']:.4f} -> "
            f"{row['balanced_f1']:.4f})"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n\n4. SUMMARY")
print("-" * 75)

print(f"Condition classes compared: {len(comparison)}")
print(
    "Improved:",
    int((comparison["f1_change"] > 0.000001).sum()),
)
print(
    "Declined:",
    int((comparison["f1_change"] < -0.000001).sum()),
)
print(
    "Unchanged:",
    int((comparison["status"] == "Unchanged").sum()),
)

print(f"\nFull comparison saved to:\n{OUTPUT_FILE}")

print("\nAnalysis complete.")
print("=" * 75)