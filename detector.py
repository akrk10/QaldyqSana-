import os
import io
import gdown

from PIL import Image, ImageOps
from ultralytics import YOLO

# Model configuration
FILE_ID = "15FVrrLwqYpXi6W1UbEdcmESovQf95y9b"
MODEL_PATH = "best_v3.pt"

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.45

CATEGORIES = [
  "Plastic",
  "Paper/Cardboard",
  "Metal",
  "Glass",
  "Other"
]

# Load YOLO11n V3
_model = None
def load_model():
  global _model
  
  if _model is None:
    if (
        not os.path.exists(MODEL_PATH)
        or os.path.getsize(MODEL_PATH) < 100000
    ):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

    _model = YOLO(MODEL_PATH)
    
  return _model
# Map model classes to waste categories
def to_category(class_name):
  name = str(class_name).lower().replace("_", " ").replace("-", " ")
  
  if "plastic" in name:
    return "Plastic"
    
  if "paper" in name or "cardboard" in name:
    return "Paper/Cardboard"
    
  if "metal" in name or "aluminium" in name or "aluminum" in name:
    return "Metal"
    
  if "glass" in name:
    return "Glass"
    
  return "Other"
# Convert uploaded image to RGB
def _to_pil(image):
  if isinstance(image, Image.Image):
    img = image.copy()
  elif isinstance(image, (bytes, bytearray)):
    img = Image.open(io.BytesIO(image))
  elif hasattr(image, "getvalue"):
    img = Image.open(io.BytesIO(image.getvalue()))
  elif hasattr(image, "read"):
    img = Image.open(io.BytesIO(image.read()))
  else:
    img = Image.open(image)

return ImageOps.exif_transpose(img).convert("RGB")
# Main detection function
def detect(image):
  img = _to_pil(image)
  model = load_model()

result = model.predict(
    source=img,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    verbose=False
)[0]

names = result.names

counts = {category: 0 for category in CATEGORIES}
detections = []

for box in result.boxes:
    class_id = int(box.cls[0])
    class_name = str(names[class_id])
    category = to_category(class_name)

    x1, y1, x2, y2 = [
        round(float(value), 2)
        for value in box.xyxy[0].tolist()
    ]

    confidence = round(float(box.conf[0]), 4)

    counts[category] += 1

    detections.append({
        "class_name": class_name,
        "category": category,
        "confidence": confidence,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    })

return {
    "image": {
        "width": img.width,
        "height": img.height
    },
    "detections": detections,
    "counts": counts,
    "total": len(detections)
}
