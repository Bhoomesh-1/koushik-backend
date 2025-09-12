from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("waste_classifier.h5")

# Manually define your class names in the correct order
CLASS_NAMES = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable']

app = FastAPI()

def preprocess_image(image: Image.Image):
    # Resize to the model's expected input size (224x224 for MobileNetV2)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_array = preprocess_image(image)
    
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
