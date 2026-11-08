from pathlib import Path
import cv2
import numpy as np
from ultralytics import SAM

IMAGE = Path(
    "data/Image Data base/anthracnose in tea/Anthracnose_in_tea39.jpg"
)
OUTPUT = Path("data/vision/sam_tests")
OUTPUT.mkdir(parents=True, exist_ok=True)

if not IMAGE.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE}")

# Use the SAM 2 checkpoint already downloaded for your project.
model = SAM("sam2.1_b.pt")

# Generate candidate masks automatically.
results = model(
    str(IMAGE),
    device="mps",
    verbose=False,
)

result = results[0]
image = cv2.imread(str(IMAGE))

if result.masks is None:
    print("SAM returned no masks.")
    raise SystemExit(0)

masks = result.masks.data.cpu().numpy()
print("Candidate masks generated:", len(masks))

# Visualize candidate masks in a separate output image.
overlay = image.copy()
rng = np.random.default_rng(42)

for mask in masks:
    mask = cv2.resize(
        mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    color = rng.integers(60, 256, size=3).tolist()
    overlay[mask] = (
        0.55 * overlay[mask] + 0.45 * np.array(color)
    ).astype(np.uint8)

output_path = OUTPUT / "tea_sam_candidates.jpg"
cv2.imwrite(str(output_path), overlay)

print("Original image unchanged:", IMAGE)
print("Candidate visualization saved:", output_path)
print(
    "Note: these are generic object masks, not verified leaf masks."
)
