
import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI(title="Road Damage Prediction API")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "road_damage_best.pt"
)

model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Road Damage API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/jpg"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WebP image."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    results = model.predict(
        source=image,
        imgsz=224,
        device="cpu",
        verbose=False
    )

    prediction = results[0]
    class_id = prediction.probs.top1
    confidence = prediction.probs.top1conf.item()

    return {
        "predicted_class": prediction.names[class_id],
        "confidence": round(confidence, 4)
    }
