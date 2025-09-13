# app.py
import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image

# Config via environment
MODEL_PATH = os.getenv("MODEL_PATH", "mymodel.h5")
FRONTEND_URL = os.getenv("FRONTEND_URL")  # set this on Render to your Vercel URL (optional)

# Setup FastAPI
app = FastAPI(title="Waste Classifier API")

# CORS: if FRONTEND_URL set, allow that origin only; otherwise allow all (useful for testing)
if FRONTEND_URL:
    origins = [FRONTEND_URL]
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
print(f"Loading model from: {MODEL_PATH} ...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    # Keep app running but model will be None — endpoint will respond with error
    model = None

# Define classes in same order the model outputs them
CLASS_NAMES = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable']

def preprocess_image(image: Image.Image):
    # Keep consistent with how model was trained: 224x224 and normalized
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file content type quickly
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    start = time.perf_counter()
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    img_array = preprocess_image(image)

    # Predict
    preds = model.predict(img_array)
    # handle case model returns 1D or 2D
    preds = np.asarray(preds).squeeze()
    idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[idx]
    confidence = float(np.max(preds))  # probability 0..1

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "class": predicted_class,
        "confidence": confidence,
        "processingTime": elapsed_ms
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

