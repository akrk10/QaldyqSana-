import os
import gdown
from PIL import Image, ImageOps
from ultralytics import YOLO

FILE_ID = "15FVrrLwqYpXi6W1UbEdcmESovQf95y9b"
MODEL_PATH = "best_v3.pt"

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.45

CATEGORIES = [
    "Plastic",
    "Paper/Cardboard",
    "Metal",
    "Glass",
    "Other",
]

_model = None


def load_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                MODEL_PATH,
                quiet=False,
            )

        _model = YOLO(MODEL_PATH)

    return _model


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


def detect(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = ImageOps.exif_transpose(image).convert("RGB")
    model = load_model()

    result = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )[0]

    counts = {category: 0 for category in CATEGORIES}
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = str(result.names[class_id])
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
                "y2": y2,
            },
        })

    return {
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "detections": detections,
        "counts": counts,
        "total": len(detections),
    }
