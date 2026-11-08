from pathlib import Path
import cv2
import numpy as np
from ultralytics import SAM

IMAGE = Path(
    "data/Image Data base/anthracnose in tea/Anthracnose_in_tea39.jpg"
)
OUT = Path("data/vision/sam_prompted_tests")
OUT.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE))
if image is None:
    raise FileNotFoundError(IMAGE)

height, width = image.shape[:2]

# Prompt points: one positive point inside the leaf,
# and four negative points in the surrounding background.
points = [[
    [width // 2, height // 2],
    [int(width * 0.08), int(height * 0.10)],
    [int(width * 0.90), int(height * 0.15)],
    [int(width * 0.10), int(height * 0.85)],
    [int(width * 0.90), int(height * 0.85)],
]]
labels = [[1, 0, 0, 0, 0]]

model = SAM("sam2.1_b.pt")
result = model.predict(
    source=str(IMAGE),
    points=points,
    labels=labels,
    imgsz=1024,
    save=False,
    verbose=False,
)[0]

if result.masks is None or len(result.masks.data) == 0:
    raise RuntimeError("SAM returned no masks.")

raw_mask = result.masks.data[0].cpu().numpy()
mask = cv2.resize(
    raw_mask.astype(np.uint8),
    (width, height),
    interpolation=cv2.INTER_NEAREST,
)

binary = (mask > 0).astype(np.uint8) * 255
output = OUT / "tea_leaf_binary_mask.png"

if not cv2.imwrite(str(output), binary):
    raise RuntimeError(f"Could not save {output}")

print("Saved binary mask:", output)
print(f"Leaf-mask area: {np.mean(binary > 0):.1%}")
print("White = predicted leaf; black = predicted background.")
