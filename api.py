from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from detector import detect
import io

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=[""],
  allow_methods=[""],
  allow_headers=["*"],
)

@app.get("/")
def home():
  return {"message": "QaldyqSana API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  image = Image.open(io.BytesIO(await file.read()))
  return detect(image)
