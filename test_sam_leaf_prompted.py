from pathlib import Path
import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Image Data base"
OUT = ROOT / "data" / "vision" / "sam_six_category_test"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "algal leaf in tea",
    "anthracnose in tea",
    "bird eye spot in tea",
    "brown blight in tea",
    "healthy tea leaf",
    "red leaf spot in tea",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

model = SAM(str(ROOT / "sam2.1_b.pt"))


def find_sample(category):
    folder = DATA / category

    if not folder.is_dir():
        print(f"[SKIP] Folder not found: {folder}")
        return None

    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        print(f"[SKIP] No images in: {category}")
        return None

    return images[0]


for category in CATEGORIES:
    image_path = find_sample(category)

    if image_path is None:
        continue

    print(f"\nTesting category: {category}")
    print(f"Image: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("[SKIP] OpenCV could not read this image.")
        continue

    height, width = image.shape[:2]

    # Initial experiment: a positive point at the image centre.
    positive_point = [width // 2, height // 2]

    # Negative points near the corners.
    negative_points = [
        [int(width * 0.05), int(height * 0.05)],
        [int(width * 0.95), int(height * 0.05)],
        [int(width * 0.05), int(height * 0.95)],
        [int(width * 0.95), int(height * 0.95)],
    ]

    points = [positive_point] + negative_points
    labels = [1, 0, 0, 0, 0]

    try:
        results = model.predict(
            source=str(image_path),
            points=[points],
            labels=[labels],
            imgsz=1024,
            device="mps",
            save=False,
            verbose=False,
        )
    except Exception as exc:
        print(f"[ERROR] SAM failed: {exc}")
        continue

    result = results[0]

    if result.masks is None or len(result.masks.data) == 0:
        print("[NO MASK] SAM returned no masks.")
        continue

    masks = result.masks.data.cpu().numpy()
    overlay = image.copy()

    for i, raw_mask in enumerate(masks):
        mask = cv2.resize(
            raw_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        print(f"  Mask {i + 1}: {mask.mean():.1%} of image")

        # Blend predicted mask with the original image.
        green = np.array([40, 220, 40], dtype=np.uint8)
        overlay[mask] = (
            0.45 * overlay[mask] + 0.55 * green
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    cv2.circle(overlay, tuple(positive_point), 8, (255, 0, 0), -1)

    safe_name = category.replace(" ", "_")
    overlay_path = OUT / f"{safe_name}_overlay.jpg"
    mask_path = OUT / f"{safe_name}_mask.png"

    # Save the overlay for visual inspection.
    cv2.imwrite(str(overlay_path), overlay)

    # Save the first predicted mask as a binary image.
    first_mask = cv2.resize(
        masks[0].astype(np.uint8) * 255,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(mask_path), first_mask)

    print(f"  Overlay saved: {overlay_path}")
    print(f"  Binary mask saved: {mask_path}")

print("\nFinished testing the available tea categories.")