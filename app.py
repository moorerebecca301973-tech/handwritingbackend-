from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import requests, os, io
from PIL import Image

app = FastAPI()

# --- Allow frontend to connect ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GitHub model file ---
MODEL_URL = "https://github.com/moorerebecca301973-tech/model/releases/download/verson/best_writer_model.keras"
MODEL_PATH = "best_writer_model.keras"

# --- Download model if not present ---
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from GitHub...")
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully.")

# --- Load model ---
print("🧠 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model ready for predictions.")

# --- Constants ---
CLASS_NAMES = ['Rebby', 'Precious',]  # <-- update this
IMG_HEIGHT, IMG_WIDTH = 224, 224
THRESHOLD = 0.60

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.expand_dims(np.array(img) / 255.0, 0)

    preds = model.predict(img_array)
    conf = float(np.max(preds))
    pred_class = CLASS_NAMES[np.argmax(preds)]

    if conf < THRESHOLD:
        result = {"predicted_class": "Unknown", "confidence": conf}
    else:
        result = {"predicted_class": pred_class, "confidence": conf}

    return result
