import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Image Data base"
OUT = ROOT / "data" / "vision" / "batch_masks"
MANIFEST = OUT / "manifest.csv"
REVIEW = OUT / "review"
REVIEW.mkdir(parents=True, exist_ok=True)

with MANIFEST.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

tiles = []

for row in rows[:20]:
    image_path = DATA / row["image_path"]
    image = cv2.imread(str(image_path))
    if image is None:
        continue

    image = cv2.resize(image, (240, 180))
    mask_file = ROOT / row["output_path"]

    with np.load(mask_file) as saved:
        masks = saved["masks"]

    overlay = image.copy()

    for mask in masks:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        tint = np.zeros_like(overlay)
        tint[:] = (0, 255, 0)
        overlay[mask] = (
            0.55 * overlay[mask] + 0.45 * tint[mask]
        ).astype(np.uint8)

    label = f"{Path(row['image_path']).name[:25]} | masks={len(masks)}"
    cv2.putText(
        overlay, label, (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    tiles.append(overlay)

if not tiles:
    raise RuntimeError("No images could be loaded for review.")

columns = 4
rows_count = (len(tiles) + columns - 1) // columns
sheet = np.zeros(
    (rows_count * 180, columns * 240, 3), dtype=np.uint8
)

for i, tile in enumerate(tiles):
    y = (i // columns) * 180
    x = (i % columns) * 240
    sheet[y:y + 180, x:x + 240] = tile

output = REVIEW / "first_20_contact_sheet.jpg"
cv2.imwrite(str(output), sheet)
print(f"Saved contact sheet: {output}")
