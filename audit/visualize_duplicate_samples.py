from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "results" / "cross_split_duplicate_groups.csv"
OUTPUT_PATH = ROOT / "results" / "duplicate_samples_contact_sheet.png"

df = pd.read_csv(INPUT_PATH)

# Select groups containing images from different splits.
df = df[df["split_count"] > 1].head(12)

if df.empty:
    print("No cross-split groups available to inspect.")
    raise SystemExit(0)

fig, axes = plt.subplots(
    len(df),
    3,
    figsize=(15, max(5, len(df) * 3.5)),
    squeeze=False,
)

for row_index, (_, row) in enumerate(df.iterrows()):
    paths = str(row["paths"]).split(" || ")
    group_hash = row["hash"]

    for col_index in range(3):
        ax = axes[row_index][col_index]
        ax.axis("off")

        if col_index >= len(paths):
            continue

        path = Path(paths[col_index])

        try:
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))

            # Identify split from the full dataset path.
            matching = None

            # Find the matching image row in the duplicate audit.
            audit = pd.read_csv(
                ROOT / "results" / "image_duplicate_audit.csv"
            )
            matching = audit[
                (audit["hash"] == group_hash)
                & (audit["path"] == str(path))
                & (audit["duplicate_type"] == "identical_phash")
            ]

            split = (
                str(matching.iloc[0]["split"])
                if not matching.empty
                else "unknown split"
            )

            ax.set_title(
                f"Group {row_index + 1} | {split}\n"
                f"{path.name}",
                fontsize=8,
            )

        except Exception as error:
            ax.text(
                0.5,
                0.5,
                f"Could not load image\n{error}",
                ha="center",
                va="center",
            )

fig.suptitle(
    "Suspected Cross-Split Visual Duplicates",
    fontsize=14,
)
plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Contact sheet saved to:\n{OUTPUT_PATH}")
print("Open the image and compare each row.")