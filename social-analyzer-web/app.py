from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import pickle
import numpy as np
import os
import io

# ── Initialize the Web API ────────────────────────────────────────
app = FastAPI(title="Smart Social Media Analyzer API")

# ── Load Models & Translators (Happens ONCE at startup) ───────────
MODEL_DIR = "models"

try:
    print("Loading AI Models into memory...")
    
    # 1. Sentiment Analysis
    sentiment_model = load_model(os.path.join(MODEL_DIR, 'sentiment_bilstm_model.h5'), compile=False)
    with open(os.path.join(MODEL_DIR, 'sentiment_tokenizer.pkl'), 'rb') as f:
        sentiment_tokenizer = pickle.load(f)
        
    # 2. Fake Text Detection (Autoencoder)
    fake_text_model = load_model(os.path.join(MODEL_DIR, 'text_autoencoder.h5'), compile=False)
    
    # LOAD TOKENIZER (This was missing!)
    with open(os.path.join(MODEL_DIR, 'autoencoder_tokenizer.pkl'), 'rb') as f:
        fake_text_tokenizer = pickle.load(f)
        
    # LOAD DYNAMIC THRESHOLD
    with open(os.path.join(MODEL_DIR, 'threshold.pkl'), 'rb') as f:
        FAKE_TEXT_THRESHOLD = pickle.load(f)

    # 3. Deepfake Forensics (ResNet50)
    forensics_model = load_model(os.path.join(MODEL_DIR, 'forensics_resnet50.h5'), compile=False)
        
    print("✅ All models loaded successfully! Server is ready.")
except Exception as e:
    print(f"❌ Error loading models. Details: {e}")

# ── Data Schemas ──────────────────────────────────────────────────
class PostRequest(BaseModel):
    text: str

# ── API Endpoints ─────────────────────────────────────────────────

@app.post("/analyze-sentiment")
def analyze_sentiment(request: PostRequest):
    """Predicts if a post is Positive, Neutral, or Negative."""
    try:
        seq = sentiment_tokenizer.texts_to_sequences([request.text])
        pad = pad_sequences(seq, maxlen=50, padding='post')
        
        pred = sentiment_model.predict(pad, verbose=0)
        classes = ['Negative', 'Neutral', 'Positive']
        idx = np.argmax(pred)
        
        return {
            "text": request.text,
            "sentiment": classes[idx],
            "confidence": round(float(pred[0][idx]) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-fake")
def detect_fake_post(request: PostRequest):
    """Uses the LSTM Autoencoder to detect Spam and Out-Of-Vocabulary anomalies."""
    try:
        MAX_LEN = 20
        
        # 1. Tokenize the input
        seq = fake_text_tokenizer.texts_to_sequences([request.text])
        
        # 2. Check for OOV (Out-Of-Vocabulary) Anomaly
        # In Keras, the <OOV> token is always integer 1.
        total_words = len(seq[0])
        oov_count = seq[0].count(1)
        oov_ratio = oov_count / total_words if total_words > 0 else 0
        
        # 3. Calculate Autoencoder Reconstruction Error
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        enc_target = pad.reshape(-1, MAX_LEN, 1)
        pred = fake_text_model.predict(pad, verbose=0)
        
        error = float(np.mean(np.power(enc_target - pred, 2)))
        
        # 4. HYBRID DETECTION LOGIC
        # It is fake IF the reconstruction error is too high OR if too many words are unknown (>30%)
        is_fake = bool(error > FAKE_TEXT_THRESHOLD or oov_ratio > 0.3)
        
        # Determine the exact reason for the UI
        if oov_ratio > 0.3:
            status_message = "FAKE/SPAM (High Unknown Vocabulary)"
        elif error > FAKE_TEXT_THRESHOLD:
            status_message = "FAKE/SPAM (Anomalous Structure)"
        else:
            status_message = "NORMAL"

        return {
            "text": request.text,
            "reconstruction_error": error,
            "oov_ratio": round(oov_ratio * 100, 2), # Send as percentage
            "is_fake": is_fake,
            "status": status_message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """Uses ResNet50 to classify images as Real Photographs or AI-Generated (Deepfakes)."""
    try:
        # Read the uploaded image bytes directly into PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resize to ResNet50's strict 224x224 requirement
        image = image.resize((224, 224))
        
        # Preprocess the pixels mathematically for the model
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict: Result < 0.5 = AI, Result > 0.5 = Real
        prediction = forensics_model.predict(img_array, verbose=0)[0][0]
        
        if prediction < 0.5:
            result = "⚠️ AI-GENERATED CONTENT DETECTED"
            confidence = (1.0 - prediction) * 100
        else:
            result = "✅ REAL PHOTOGRAPH"
            confidence = prediction * 100
            
        return {
            "filename": file.filename,
            "status": result,
            "confidence": f"{confidence:.2f}%",
            "raw_score": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Social Media Analyzer API. All engines running."}
