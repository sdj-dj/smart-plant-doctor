from pathlib import Path
import csv
import hashlib
import json
import time

import cv2
import numpy as np
import torch
from ultralytics import SAM

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Image Data base"
MODEL_PATH = ROOT / "sam2.1_b.pt"
OUT = ROOT / "data" / "vision" / "automatic_leaf_masks"

MASK_DIR = OUT / "masks"
OVERLAY_DIR = OUT / "overlays"
MANIFEST = OUT / "manifest.csv"
ERRORS = OUT / "errors.jsonl"

MAX_SIDE = 1024
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

if not DATA.is_dir():
    raise FileNotFoundError(f"Dataset not found: {DATA}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

completed = set()
if MANIFEST.exists():
    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") in {"saved", "no_masks"}:
                completed.add(row["image"])

new_manifest = not MANIFEST.exists()
manifest = MANIFEST.open("a", newline="", encoding="utf-8")
writer = csv.DictWriter(
    manifest,
    fieldnames=[
        "image", "category", "status",
        "candidate_masks", "message"
    ],
)
if new_manifest:
    writer.writeheader()
    manifest.flush()

try:
    print("Loading SAM model...", flush=True)
    model = SAM(str(MODEL_PATH))

    print("Scanning dataset...", flush=True)
    images = sorted(
        p for p in DATA.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    total = len(images)
    pending = [
        p for p in images
        if str(p.relative_to(DATA)) not in completed
    ]

    print(f"Images discovered: {total:,}", flush=True)
    print(f"Already recorded: {len(completed):,}", flush=True)
    print(f"Images remaining: {len(pending):,}", flush=True)

    started = time.time()

    for index, path in enumerate(pending, 1):
        relative = str(path.relative_to(DATA))
        image = cv2.imread(str(path))

        if image is None:
            writer.writerow({
                "image": relative,
                "category": path.parent.name,
                "status": "error",
                "candidate_masks": 0,
                "message": "Could not read image",
            })
            manifest.flush()
            print(f"[{index}/{len(pending)}] READ ERROR: {relative}", flush=True)
            continue

        h, w = image.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = (
            cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else image
        )

        image_id = hashlib.sha1(relative.encode("utf-8")).hexdigest()

        try:
            # Prompt-free prediction is experimental. Validate the output.
            result = model.predict(
                source=small,
                imgsz=MAX_SIDE,
                device=DEVICE,
                save=False,
                verbose=False,
            )[0]

            if result.masks is None or len(result.masks.data) == 0:
                writer.writerow({
                    "image": relative,
                    "category": path.parent.name,
                    "status": "no_masks",
                    "candidate_masks": 0,
                    "message": "No masks returned without prompts",
                })
                manifest.flush()
                print(f"[{index}/{len(pending)}] NO MASKS: {relative}", flush=True)
                continue

            raw_masks = result.masks.data.cpu().numpy()
            folder = MASK_DIR / image_id
            folder.mkdir(parents=True, exist_ok=True)
            overlay = small.copy()
            saved = 0

            for mask_index, raw_mask in enumerate(raw_masks, 1):
                mask = cv2.resize(
                    raw_mask.astype(np.uint8),
                    (sw, sh),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

                coverage = float(mask.mean())
                if coverage < 0.005 or coverage > 0.95:
                    continue

                cv2.imwrite(
                    str(folder / f"candidate_{mask_index}.png"),
                    mask.astype(np.uint8) * 255,
                )

                green = np.array([40, 220, 40], dtype=np.uint8)
                overlay[mask] = (
                    0.45 * overlay[mask] + 0.55 * green
                ).astype(np.uint8)
                saved += 1

            cv2.imwrite(str(OVERLAY_DIR / f"{image_id}.jpg"), overlay)

            writer.writerow({
                "image": relative,
                "category": path.parent.name,
                "status": "saved",
                "candidate_masks": saved,
                "message": "Experimental candidates; not validated",
            })
            manifest.flush()

            if index % 25 == 0 or index == 1:
                elapsed = (time.time() - started) / 60
                print(
                    f"[{index}/{len(pending)}] "
                    f"{saved} candidates | {elapsed:.1f} min | {relative}",
                    flush=True,
                )

        except KeyboardInterrupt:
            print("\nInterrupted. Progress has been recorded.", flush=True)
            break
        except Exception as exc:
            with ERRORS.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "image": relative,
                    "error": repr(exc),
                }) + "\n")

            writer.writerow({
                "image": relative,
                "category": path.parent.name,
                "status": "error",
                "candidate_masks": 0,
                "message": repr(exc),
            })
            manifest.flush()
            print(f"[ERROR] {relative}: {exc}", flush=True)

    print("\nRun finished.", flush=True)
    print(f"Manifest: {MANIFEST}", flush=True)
    print(f"Masks: {MASK_DIR}", flush=True)
    print(f"Overlays: {OVERLAY_DIR}", flush=True)

finally:
    manifest.close()
