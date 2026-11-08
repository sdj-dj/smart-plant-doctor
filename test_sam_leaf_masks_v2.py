from pathlib import Path
import cv2
import numpy as np
from ultralytics import SAM

IMAGE = Path(
    "data/Image Data base/anthracnose in tea/Anthracnose_in_tea39.jpg"
)
OUT = Path("data/vision/sam_tests_v2")
OUT.mkdir(parents=True, exist_ok=True)

if not IMAGE.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE}")

image = cv2.imread(str(IMAGE))
if image is None:
    raise RuntimeError(f"Could not read image: {IMAGE}")

model = SAM("sam2.1_b.pt")

# Generic SAM prediction without unsupported arguments.
results = model.predict(
    source=str(IMAGE),
    imgsz=1024,
    conf=0.25,
    save=False,
    verbose=False,
)

result = results[0]

if result.masks is None:
    print("No masks returned.")
    raise SystemExit(0)

masks = result.masks.data.cpu().numpy()
height, width = image.shape[:2]
overlay = image.copy()
rng = np.random.default_rng(7)

print("Candidate masks:", len(masks))

for i, raw_mask in enumerate(masks):
    mask = cv2.resize(
        raw_mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    area_fraction = float(mask.mean())

    if area_fraction < 0.005:
        continue

    color = rng.integers(50, 256, size=3).astype(np.uint8)
    overlay[mask] = (
        0.55 * overlay[mask] + 0.45 * color
    ).astype(np.uint8)

    print(f"Mask {i}: {area_fraction:.1%} of image")

output = OUT / "tea_candidates_v2.jpg"
if not cv2.imwrite(str(output), overlay):
    raise RuntimeError(f"Could not save image: {output}")

print("Saved visualization:", output)
print("Note: candidate masks are not verified leaf annotations.")
