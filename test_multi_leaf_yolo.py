from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = Path("models/vision/phenoleaf_yolo11_seg.pt")
IMAGE_PATH = Path(
    "data/Image Data base/anthracnose in tea/Anthracnose_in_tea39.jpg"
)
OUTPUT_DIR = Path("data/vision/multi_leaf_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(IMAGE_PATH)

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise RuntimeError(f"Could not read {IMAGE_PATH}")

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=image,
    conf=0.05,
    imgsz=1024,
    retina_masks=True,
    verbose=False,
)

result = results[0]
print("Model class names:", result.names)

if result.masks is None or result.boxes is None:
    print("No segmentation masks detected.")
    output = image.copy()
else:
    output = result.plot()
    print("Detected instances:", len(result.masks.data))

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        print(
            f"Instance {i}: class={result.names[class_id]}, "
            f"confidence={confidence:.3f}"
        )

output_path = OUTPUT_DIR / "tea_yolo_test.jpg"

if not cv2.imwrite(str(output_path), output):
    raise RuntimeError(f"Could not save {output_path}")

print("Saved visualization:", output_path)
