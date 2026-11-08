import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ml.model import load_model, get_image_transform


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "vision" / "live_captures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# General-purpose YOLO model.
# It is NOT a plant-disease detector.
YOLO_MODEL = "yolo11n.pt"

CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
YOLO_EVERY_N_FRAMES = 3

CLASSIFIER_CHECKPOINT = (
    ROOT / "checkpoints" / "experiments" / "leakage_reduced"
    / "condition_balanced_best_v1.pth"
)
SPECIES_MAPPING = ROOT / "audit" / "species_classes.json"
CONDITION_MAPPING = ROOT / "audit" / "condition_classes.json"
CLASSIFIER_EVERY_N_FRAMES = 10


def load_class_names(mapping_path):
    with mapping_path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    return [
        name for name, index in
        sorted(mapping.items(), key=lambda item: int(item[1]))
    ]


def predict_frame(model, transform, frame, device, species_names, condition_names):
    """Run the existing classifier on one camera frame."""
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        species_logits, condition_logits = model(tensor)
        species_scores = torch.softmax(species_logits.float(), dim=1)[0]
        condition_scores = torch.softmax(condition_logits.float(), dim=1)[0]

        species_values, species_indices = torch.topk(
            species_scores, min(3, len(species_names))
        )
        condition_values, condition_indices = torch.topk(
            condition_scores, min(3, len(condition_names))
        )

    species = [
        (species_names[i.item()], score.item())
        for score, i in zip(species_values, species_indices)
    ]
    conditions = [
        (condition_names[i.item()], score.item())
        for score, i in zip(condition_values, condition_indices)
    ]

    return species, conditions

def analyze_leaf_patterns(frame):
    """
    Exploratory visual analysis only.
    Green-region masks and dark-pixel clusters are not disease diagnoses.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Approximate green regions; this can miss yellow, brown, or dark leaves.
    lower_green = np.array([25, 30, 25], dtype=np.uint8)
    upper_green = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter small green regions to reduce noise.
    leaf_regions = [
        c for c in contours
        if cv2.contourArea(c) > 1200
    ]

    annotated = frame.copy()
    for contour in leaf_regions:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(
            annotated, "Green region",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2
        )

    # Exploratory dark-pixel signal inside approximate green regions.
    # Shadows, veins, dirt, and lighting can produce the same signal.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_pixels = cv2.inRange(gray, 0, 65)
    dark_inside_green = cv2.bitwise_and(dark_pixels, mask)

    green_area = int(cv2.countNonZero(mask))
    dark_area = int(cv2.countNonZero(dark_inside_green))
    dark_fraction = dark_area / green_area if green_area else 0.0

    return annotated, len(leaf_regions), green_area, dark_fraction


def main():
    print("Loading general-purpose YOLO model...")
    detector = YOLO(YOLO_MODEL)

    yolo_device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"YOLO inference device: {yolo_device}")

    print("Loading existing ResNet18 plant classifier...")
    classifier, classifier_device = load_model(CLASSIFIER_CHECKPOINT)
    species_names = load_class_names(SPECIES_MAPPING)
    condition_names = load_class_names(CONDITION_MAPPING)
    transform = get_image_transform()

    if len(species_names) != 19 or len(condition_names) != 36:
        raise ValueError("Class mapping sizes do not match the trained model.")

    print(f"Classifier device: {classifier_device}")
    print("Species classes:", len(species_names))
    print("Condition classes:", len(condition_names))

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera. Check macOS camera permission "
            "for Terminal and try CAMERA_INDEX = 1 if another camera is connected."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_count = 0
    last_yolo_result = None
    last_species = []
    last_conditions = []
    fps = 0.0
    previous_time = time.time()

    print("\nLIVE CAMERA STARTED")
    print("Press Q to quit.")
    print("Press S to save a frame.")
    print("Green boxes are approximate color regions, not verified leaf boundaries.")
    print("YOLO labels are general-purpose object detections, not disease predictions.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a camera frame.")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)

            visual, region_count, green_area, dark_fraction = (
                analyze_leaf_patterns(frame)
            )

            # Run general-purpose YOLO periodically to reduce computation.
            if frame_count % YOLO_EVERY_N_FRAMES == 0:
                try:
                    last_yolo_result = detector.predict(
                        source=frame,
                        imgsz=640,
                        conf=0.35,
                        device=yolo_device,
                        verbose=False,
                    )[0]
                except Exception as exc:
                    print(f"YOLO inference warning: {exc}")
                    last_yolo_result = None

            # Run the existing plant classifier periodically.
            if frame_count % CLASSIFIER_EVERY_N_FRAMES == 0:
                try:
                    last_species, last_conditions = predict_frame(
                        classifier,
                        transform,
                        frame,
                        classifier_device,
                        species_names,
                        condition_names,
                    )
                except Exception as exc:
                    print(f"Classifier inference warning: {exc}")
                    last_species, last_conditions = [], []

            # Draw general-purpose YOLO detections when available.
            if last_yolo_result is not None:
                for box in last_yolo_result.boxes:
                    class_id = int(box.cls.item())
                    label = detector.names.get(class_id, str(class_id))
                    confidence = float(box.conf.item())

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].detach().cpu().tolist()
                    )

                    cv2.rectangle(
                        visual, (x1, y1), (x2, y2), (255, 180, 0), 2
                    )
                    cv2.putText(
                        visual,
                        f"YOLO: {label} {confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 180, 0),
                        2,
                    )

            current_time = time.time()
            elapsed = current_time - previous_time
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                fps = instantaneous_fps if fps == 0 else 0.85 * fps + 0.15 * instantaneous_fps
            previous_time = current_time

            lines = [
                f"Live Plant Vision | FPS: {fps:.1f}",
                f"Approx. green regions: {region_count}",
                f"Green pixels: {green_area}",
                f"Dark pixels within green mask: {dark_fraction:.1%}",
                "RESNET18 - preliminary screening, not a diagnosis",
            ]

            if last_species:
                lines.append("Species candidates:")
                for name, score in last_species[:2]:
                    lines.append(f"  {name}: {score:.1%} score")

            if last_conditions:
                lines.append("Condition candidates (uncalibrated scores):")
                for name, score in last_conditions[:2]:
                    lines.append(f"  {name}: {score:.1%} score")

            lines.extend([
                "Scores are not calibrated disease probabilities.",
                "Q: quit | S: save frame",
            ])

            y = 25
            for line in lines:
                cv2.putText(
                    visual, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 3, cv2.LINE_AA
                )
                cv2.putText(
                    visual, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (25, 25, 25), 1, cv2.LINE_AA
                )
                y += 27

            cv2.imshow("Smart Plant Doctor - Live Camera", visual)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = OUTPUT_DIR / f"leaf_frame_{int(time.time())}.jpg"
                if cv2.imwrite(str(filename), frame):
                    print(f"Saved frame: {filename}")
                else:
                    print("Could not save the frame.")

    except KeyboardInterrupt:
        print("\\nCamera interrupted by user.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera stopped cleanly.")


if __name__ == "__main__":
    main()
