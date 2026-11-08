from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
DUPLICATE_PATH = ROOT / "results" / "cross_split_duplicate_groups.csv"
OUTPUT_PATH = ROOT / "results" / "visual_duplicate_verification.csv"
IMAGE_SIZE = (256, 256)


def resolve_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path

    parts = list(Path(str(value).replace("\\", "/")).parts)
    while parts and parts[0] in {"data", "Image Data base"}:
        parts.pop(0)

    return ROOT / "data" / "Image Data base" / Path(*parts)


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def compare_images(path_a, path_b):
    image_a = load_image(path_a)
    image_b = load_image(path_b)

    ssim = structural_similarity(
        image_a,
        image_b,
        channel_axis=2,
        data_range=255,
    )

    mae = np.abs(
        image_a.astype(np.float32) - image_b.astype(np.float32)
    ).mean()

    return float(ssim), float(mae)


def main():
    print("=" * 65, flush=True)
    print("VISUAL DUPLICATE VERIFICATION", flush=True)
    print("=" * 65, flush=True)

    if not DUPLICATE_PATH.exists():
        raise FileNotFoundError(
            f"Duplicate report not found: {DUPLICATE_PATH}"
        )

    groups = pd.read_csv(DUPLICATE_PATH)
    groups = groups[groups["split_count"] > 1].copy()

    print(f"Cross-split groups: {len(groups):,}", flush=True)

    records = []
    total_pairs = sum(
        len([
            p for p in str(value).split("||") if p.strip()
        ]) * (
            len([p for p in str(value).split("||") if p.strip()]) - 1
        ) // 2
        for value in groups["paths"]
    )

    print(f"Image pairs to compare: {total_pairs:,}", flush=True)

    for group_number, (_, group) in enumerate(groups.iterrows(), start=1):
        paths = [
            resolve_path(p.strip())
            for p in str(group["paths"]).split("||")
            if p.strip()
        ]

        for path_a, path_b in combinations(paths, 2):
            record = {
                "hash": group["hash"],
                "path_a": str(path_a),
                "path_b": str(path_b),
            }

            try:
                if not path_a.is_file() or not path_b.is_file():
                    record.update({
                        "ssim": np.nan,
                        "pixel_mae": np.nan,
                        "status": "missing_file",
                    })
                else:
                    ssim, mae = compare_images(path_a, path_b)

                    if ssim >= 0.98 and mae <= 5:
                        status = "near_pixel_identical"
                    elif ssim >= 0.90:
                        status = "high_visual_similarity"
                    else:
                        status = "needs_manual_review"

                    record.update({
                        "ssim": ssim,
                        "pixel_mae": mae,
                        "status": status,
                    })

            except Exception as error:
                record.update({
                    "ssim": np.nan,
                    "pixel_mae": np.nan,
                    "status": f"error: {error}",
                })

            records.append(record)

        if group_number % 100 == 0:
            print(
                f"Processed {group_number:,}/{len(groups):,} groups",
                flush=True,
            )

    results = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nVerification summary:", flush=True)

    if results.empty:
        print("No image pairs were compared.", flush=True)
    else:
        print(results["status"].value_counts().to_string(), flush=True)

        valid = results.dropna(subset=["ssim", "pixel_mae"])
        if not valid.empty:
            print("\nSSIM summary:", flush=True)
            print(valid["ssim"].describe().to_string(), flush=True)

            print("\nMost similar pairs:", flush=True)
            print(
                valid.sort_values(
                    ["ssim", "pixel_mae"],
                    ascending=[False, True],
                ).head(10).to_string(index=False),
                flush=True,
            )

    print(f"\nReport saved to: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
