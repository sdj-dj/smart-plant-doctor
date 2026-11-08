from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Image Data base"
OUT = ROOT / "data" / "vision" / "tea_opencv_tests"

MASK_DIR = OUT / "masks"
OVERLAY_DIR = OUT / "overlays"

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LIMIT = 20

TEA_CATEGORIES = {
    "algal leaf in tea",
    "anthracnose in tea",
    "bird eye spot in tea",
    "brown blight in tea",
    "healthy tea leaf",
    "red leaf spot in tea",
}

images = sorted(
    p for p in DATA.rglob("*")
    if p.is_file()
    and p.suffix.lower() in EXTENSIONS
    and p.parent.name in TEA_CATEGORIES
)[:LIMIT]

if not images:
    raise RuntimeError(f"No images found in {DATA}")

tiles = []
results = []

for index, path in enumerate(images, start=1):
    image = cv2.imread(str(path))

    if image is None:
        print(f"[{index}] Could not read: {path.name}")
        continue

    original_h, original_w = image.shape[:2]

    # Reduce large images to make the test faster.
    scale = min(1.0, 1000 / max(original_h, original_w))

    if scale < 1:
        work = cv2.resize(
            image,
            (int(original_w * scale), int(original_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = image.copy()

    h, w = work.shape[:2]

    # GrabCut starts with a central rectangle presumed to contain
    # the foreground object. This is a candidate, not a guarantee.
    margin_x = max(2, int(w * 0.02))
    margin_y = max(2, int(h * 0.02))

    rect = (
        margin_x,
        margin_y,
        max(1, w - 2 * margin_x),
        max(1, h - 2 * margin_y),
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            work,
            mask,
            rect,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )

        candidate = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        # Keep the largest connected foreground region.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )

        if count > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            candidate = np.where(labels == largest_label, 255, 0).astype(
                np.uint8
            )

        # Restore the original image dimensions.
        full_mask = cv2.resize(
            candidate,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_path = MASK_DIR / f"{path.stem}_mask.png"
        cv2.imwrite(str(mask_path), full_mask)

        # Draw the candidate mask in green on the original image.
        overlay = image.copy()
        selected = full_mask > 0
        overlay[selected] = (
            0.55 * overlay[selected]
            + 0.45 * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            full_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        preview = cv2.resize(
            overlay, (240, 180), interpolation=cv2.INTER_AREA
        )

        label = f"{path.parent.name[:15]} | {path.name[:18]}"
        cv2.putText(
            preview, label, (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        tiles.append(preview)

        overlay_path = OVERLAY_DIR / f"{path.stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        area_ratio = float(np.count_nonzero(full_mask)) / (
            original_h * original_w
        )

        results.append(
            f"{path.relative_to(DATA)} | "
            f"foreground={area_ratio:.1%} | "
            f"mask={mask_path.relative_to(ROOT)}"
        )

        print(
            f"[{index}/{len(images)}] {path.name}: "
            f"candidate foreground {area_ratio:.1%}"
        )

    except cv2.error as exc:
        print(f"[{index}] GrabCut failed for {path.name}: {exc}")

if tiles:
    columns = 4
    rows = (len(tiles) + columns - 1) // columns

    sheet = np.zeros(
        (rows * 180, columns * 240, 3), dtype=np.uint8
    )

    for i, tile in enumerate(tiles):
        y = (i // columns) * 180
        x = (i % columns) * 240
        sheet[y:y + 180, x:x + 240] = tile

    sheet_path = OUT / "opencv_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"\nContact sheet: {sheet_path}")

(OUT / "results.txt").write_text(
    "\n".join(results) + "\n",
    encoding="utf-8",
)

print(f"Masks: {MASK_DIR}")
print(f"Overlays: {OVERLAY_DIR}")
print("IMPORTANT: These are unverified segmentation candidates.")
print("Review them before using them as training labels.")
