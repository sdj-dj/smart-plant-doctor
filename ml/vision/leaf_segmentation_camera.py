import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import SAM

from ml.model import load_model, get_image_transform


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "vision" / "live_captures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAM_MODEL = "sam2.1_b.pt"
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

CLASSIFIER_CHECKPOINT = (
    ROOT / "checkpoints" / "experiments" / "leakage_reduced"
    / "condition_balanced_best_v1.pth"
)
SPECIES_MAPPING = ROOT / "audit" / "species_classes.json"
CONDITION_MAPPING = ROOT / "audit" / "condition_classes.json"

# Mouse clicks are interpreted as positive points inside the desired leaf.
click_points = []
current_mask = None
current_frame = None
last_species = []
last_conditions = []
last_segmented_crop = None


def load_class_names(path):
    with path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    return [
        name for name, index in
        sorted(mapping.items(), key=lambda item: int(item[1]))
    ]


def mouse_callback(event, x, y, flags, param):
    global click_points

    if event == cv2.EVENT_LBUTTONDOWN:
        click_points = [(int(x), int(y))]
        print(f"Segmentation point selected: ({x}, {y})")


def segment_leaf(sam_model, frame, point):
    """
    Segment the clicked object, rejecting masks that are too large
    to be plausible leaf candidates in this camera view.

    This is a safeguard, NOT a leaf-specific classifier.
    """
    x, y = point
    h, w = frame.shape[:2]

    if not (0 <= x < w and 0 <= y < h):
        print("Click was outside the frame. Try again.")
        return None

    results = sam_model.predict(
        source=frame,
        points=[[x, y]],
        labels=[1],
        verbose=False,
    )

    if not results or results[0].masks is None:
        print("SAM returned no segmentation masks.")
        return None

    raw_masks = results[0].masks.data.detach().cpu().numpy()
    valid_masks = []

    for raw_mask in raw_masks:
        mask = cv2.resize(
            raw_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # The selected mask must contain the actual click point.
        if not mask[y, x]:
            continue

        area_ratio = float(mask.sum()) / float(h * w)

        # Reject frame-dominating objects such as a person's body.
        # Adjust this threshold later if a leaf fills much of the frame.
        if area_ratio > 0.30:
            print(
                f"Rejected oversized mask ({area_ratio:.1%} of frame). "
                "Move the leaf closer, keep it away from your body, "
                "and click inside the leaf."
            )
            continue

        # Ignore tiny accidental masks.
        if area_ratio < 0.001:
            continue

        valid_masks.append((area_ratio, mask))

    if not valid_masks:
        print(
            "No suitable mask found. Click clearly inside the leaf "
            "against a plain background."
        )
        return None

    # If several masks contain the click, prefer the smallest valid one.
    valid_masks.sort(key=lambda item: item[0])
    area_ratio, selected_mask = valid_masks[0]

    print(f"Selected mask covers {area_ratio:.1%} of the frame.")
    return selected_mask


def predict_segmented_leaf(model, transform, frame, mask, device,
                           species_names, condition_names):
    """Classify the segmented leaf crop using the existing classifier."""
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return [], [], None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    # Crop around the mask and remove the background outside the leaf.
    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    crop[~crop_mask] = 0

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

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

    return species, conditions, crop


def main():
    global click_points, current_mask, current_frame
    global last_species, last_conditions, last_segmented_crop

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading SAM 2.1 segmentation model: {SAM_MODEL}")
    print("The first run may download the model weights.")
    sam_model = SAM(SAM_MODEL)

    print("Loading existing ResNet18 classifier...")
    classifier, classifier_device = load_model(CLASSIFIER_CHECKPOINT)

    species_names = load_class_names(SPECIES_MAPPING)
    condition_names = load_class_names(CONDITION_MAPPING)
    transform = get_image_transform()

    if len(species_names) != 19 or len(condition_names) != 36:
        raise ValueError("Class mapping sizes do not match the trained model.")

    print(f"SAM device requested: {device}")
    print(f"Classifier device: {classifier_device}")
    print("Species classes:", len(species_names))
    print("Condition classes:", len(condition_names))

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera. Check macOS camera permission."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    window = "Smart Plant Doctor - Leaf Segmentation"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    frame_count = 0
    last_point_processed = None
    last_error = None

    print("\nLIVE LEAF SEGMENTATION STARTED")
    print("Click inside a leaf to segment it.")
    print("Press C to clear the current mask.")
    print("Press S to save the segmented crop.")
    print("Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read a camera frame.")
                break

            frame = cv2.flip(frame, 1)
            current_frame = frame.copy()
            frame_count += 1

            # Run segmentation only when a new click is made.
            if click_points and click_points[0] != last_point_processed:
                point = click_points[0]
                last_point_processed = point

                try:
                    print("Segmenting selected object...")
                    current_mask = segment_leaf(sam_model, frame, point)

                    if current_mask is not None:
                        (
                            last_species,
                            last_conditions,
                            last_segmented_crop,
                        ) = predict_segmented_leaf(
                            classifier,
                            transform,
                            frame,
                            current_mask,
                            classifier_device,
                            species_names,
                            condition_names,
                        )
                        last_error = None
                        print("Segmentation and classification complete.")
                    else:
                        last_error = "No mask returned; try clicking the leaf again."
                        print(last_error)

                except Exception as exc:
                    current_mask = None
                    last_segmented_crop = None
                    last_species, last_conditions = [], []
                    last_error = str(exc)
                    print(f"Segmentation error: {exc}")

            visual = frame.copy()

            if current_mask is not None:
                # Tint segmented pixels and draw the pixel-mask boundary.
                tint = np.zeros_like(visual)
                tint[:, :] = (255, 180, 0)
                visual[current_mask] = (
                    0.55 * visual[current_mask]
                    + 0.45 * tint[current_mask]
                ).astype(np.uint8)

                contours, _ = cv2.findContours(
                    current_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(visual, contours, -1, (0, 255, 255), 2)

            lines = [
                "SAM 2.1 - click inside a leaf to segment",
                "C: clear | S: save segmented crop | Q: quit",
                "ResNet18 output is preliminary screening, not diagnosis",
            ]

            if current_mask is not None:
                lines.append(
                    f"Mask pixels: {int(current_mask.sum())}"
                )

            if last_species:
                lines.append("Species candidates:")
                for name, score in last_species[:2]:
                    lines.append(f"  {name}: {score:.1%} score")

            if last_conditions:
                lines.append("Condition candidates (uncalibrated):")
                for name, score in last_conditions[:2]:
                    lines.append(f"  {name}: {score:.1%} score")

            if last_error:
                lines.append(f"Notice: {last_error[:75]}")

            y = 25
            for line in lines:
                cv2.putText(
                    visual, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    visual, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                y += 25

            cv2.imshow(window, visual)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                click_points = []
                current_mask = None
                last_point_processed = None
                last_species, last_conditions = [], []
                last_segmented_crop = None
                last_error = None

            elif key == ord("s"):
                timestamp = int(time.time())

                if last_segmented_crop is not None:
                    path = OUTPUT_DIR / f"segmented_leaf_{timestamp}.png"
                    if cv2.imwrite(str(path), last_segmented_crop):
                        print(f"Saved segmented leaf crop: {path}")
                    else:
                        print("Could not save the segmented crop.")
                else:
                    path = OUTPUT_DIR / f"leaf_frame_{timestamp}.jpg"
                    if cv2.imwrite(str(path), frame):
                        print(f"Saved unsegmented frame: {path}")
                    else:
                        print("Could not save the frame.")

    except KeyboardInterrupt:
        print("\nCamera interrupted by user.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera stopped cleanly.")


if __name__ == "__main__":
    main()
