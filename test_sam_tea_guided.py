from pathlib import Path
import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path.cwd()
DATA = ROOT / "data" / "Image Data base"
OUT = ROOT / "data" / "vision" / "sam_guided_test"

CATEGORIES = [
    "algal leaf in tea",
    "anthracnose in tea",
    "bird eye spot in tea",
    "brown blight in tea",
    "healthy tea leaf",
    "red leaf spot in tea",
]

OUT.mkdir(parents=True, exist_ok=True)
MASK_DIR = OUT / "masks"
OVERLAY_DIR = OUT / "overlays"
MASK_DIR.mkdir(exist_ok=True)
OVERLAY_DIR.mkdir(exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("Loading SAM model...")
model = SAM(str(ROOT / "sam2.1_b.pt"))

for category in CATEGORIES:
    folder = DATA / category

    if not folder.exists():
        print(f"SKIP: category not found: {category}")
        continue

    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        print(f"SKIP: no images in {category}")
        continue

    image_path = images[0]
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"SKIP: could not read {image_path}")
        continue

    h, w = image.shape[:2]

    # Positive point near the image center; negatives near the corners.
    positive = [w // 2, h // 2]
    negatives = [
        [int(w * 0.03), int(h * 0.03)],
        [int(w * 0.97), int(h * 0.03)],
        [int(w * 0.03), int(h * 0.97)],
        [int(w * 0.97), int(h * 0.97)],
    ]

    points = [positive] + negatives
    labels = [1, 0, 0, 0, 0]

    print(f"Testing: {category} / {image_path.name}")

    try:
        result = model.predict(
            source=str(image_path),
            points=[points],
            labels=[labels],
            imgsz=1024,
            device="mps",
            save=False,
            verbose=False,
        )[0]

        if result.masks is None or len(result.masks.data) == 0:
            print("  No masks returned.")
            continue

        # Select a mask that includes the positive point and excludes
        # as many negative points as possible.
        raw_masks = result.masks.data.cpu().numpy()
        best_mask = None
        best_score = -1

        for raw in raw_masks:
            mask = cv2.resize(
                raw.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            score = int(mask[positive[1], positive[0]])
            score -= sum(
                int(mask[y, x])
                for x, y in negatives
            )

            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is None:
            print("  Could not select a mask.")
            continue

        coverage = float(best_mask.mean()) * 100

        overlay = image.copy()
        green = np.array([40, 220, 40], dtype=np.uint8)
        overlay[best_mask] = (
            0.45 * overlay[best_mask] + 0.55 * green
        ).astype(np.uint8)

        cv2.imwrite(
            str(MASK_DIR / f"{category}.png"),
            best_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(OVERLAY_DIR / f"{category}.jpg"),
            overlay,
        )

        print(f"  Mask coverage: {coverage:.1f}%")

    except Exception as exc:
        print(f"  ERROR: {exc}")

print(f"\nFinished. Test results saved in:\n{OUT}")
print("These masks are experimental and need visual inspection.")
