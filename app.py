from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
import joblib

from fastapi import FastAPI
app = FastAPI()


# Load model & label encoder
model = joblib.load("emotion_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    # Save uploaded file
    temp_file = "temp.wav"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    # Extract MFCC
    y, sr = librosa.load(temp_file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

    # Predict
    pred = model.predict(mfcc_mean)
    emotion = le.inverse_transform(pred)[0]
    return {"emotion": emotion}
