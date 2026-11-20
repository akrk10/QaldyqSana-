import io
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from detector import detect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qaldyqsana")

app = FastAPI(title="QaldyqSana API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://akrk10.github.io",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "QaldyqSana API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info("1. Image upload started")

    try:
        contents = await file.read()
        logger.info("2. Image received: %s bytes", len(contents))

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info("3. Image opened: %s", image.size)

        result = detect(image)
        logger.info("4. Detection completed")

        return result

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Image processing failed")
