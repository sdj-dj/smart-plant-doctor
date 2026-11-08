import argparse
import csv
import json
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Image Data base"
MODEL_PATH = ROOT / "models" / "vision" / "phenoleaf_yolo11_seg.pt"
OUTPUT_DIR = ROOT / "data" / "vision" / "batch_masks"
MANIFEST = OUTPUT_DIR / "manifest.csv"
ERROR_LOG = OUTPUT_DIR / "errors.jsonl"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_id(path):
    return hashlib.sha1(
        str(path.relative_to(DATA_DIR)).encode("utf-8")
    ).hexdigest()


def read_completed():
    completed = set()
    if MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "processed":
                    completed.add(row["image_id"])
    return completed


def append_manifest(row):
    exists = MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "image_path", "status",
            "mask_count", "output_path", "elapsed_seconds"
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log_error(path, exc):
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "image_path": str(path.relative_to(DATA_DIR)),
            "error": repr(exc),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of new images to process; use 0 for all remaining"
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(DATA_DIR)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    completed = read_completed()
    pending = [p for p in images if image_id(p) not in completed]

    if args.limit > 0:
        pending = pending[:args.limit]

    print(f"Images found: {len(images):,}")
    print(f"Already processed: {len(completed):,}")
    print(f"Images in this run: {len(pending):,}")
    print("Loading YOLO segmentation model...")
    model = YOLO(str(MODEL_PATH))

    for index, path in enumerate(pending, start=1):
        started = time.time()
        key = image_id(path)
        output_path = OUTPUT_DIR / f"{key}.npz"

        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("OpenCV could not read image")

            result = model.predict(
                source=image,
                conf=args.conf,
                imgsz=args.imgsz,
                retina_masks=True,
                verbose=False,
            )[0]

            if result.masks is None:
                masks = np.zeros(
                    (0, image.shape[0], image.shape[1]), dtype=np.uint8
                )
                scores = np.zeros((0,), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int32)
            else:
                masks = result.masks.data.cpu().numpy().astype(np.uint8)
                scores = result.boxes.conf.cpu().numpy().astype(np.float32)
                classes = result.boxes.cls.cpu().numpy().astype(np.int32)

                # Keep mask dimensions consistent with the original image.
                resized = []
                for mask in masks:
                    resized.append(cv2.resize(
                        mask,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ))
                masks = np.stack(resized) if resized else np.zeros(
                    (0, image.shape[0], image.shape[1]), dtype=np.uint8
                )

            # Save candidate predictions, not verified ground truth.
            np.savez_compressed(
                output_path,
                masks=masks,
                scores=scores,
                classes=classes,
                image_height=np.array(image.shape[0]),
                image_width=np.array(image.shape[1]),
            )

            elapsed = round(time.time() - started, 3)
            append_manifest({
                "image_id": key,
                "image_path": str(path.relative_to(DATA_DIR)),
                "status": "processed",
                "mask_count": len(masks),
                "output_path": str(output_path.relative_to(ROOT)),
                "elapsed_seconds": elapsed,
            })

        except Exception as exc:
            log_error(path, exc)
            print(f"ERROR: {path.name}: {exc}")
            continue

        if index % 5 == 0 or index == len(pending):
            print(
                f"[{index}/{len(pending)}] "
                f"{path.name} | masks={len(masks)} | {elapsed}s"
            )

    print("\nBatch run finished.")
    print(f"Manifest: {MANIFEST}")
    print(f"Errors: {ERROR_LOG}")
    print("Masks are provisional predictions and require validation.")


if __name__ == "__main__":
    main()
