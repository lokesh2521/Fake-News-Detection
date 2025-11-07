import os
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# path setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")


#  loading model 

model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))


#  CREATE FASTAPI APP

app = FastAPI(title="Fake News Classifier API")


# Serve frontend filess  (HTML, CSS,)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


#  Serve index.html when user opens the root URL
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


#  DEFINE INPUT MODEL (for incoming data)
class PredictRequest(BaseModel):
    text: str



#  Predict endpoint
@app.post("/predict")
def predict(req: PredictRequest):

    X = vectorizer.transform([req.text])
    pred = model.predict(X)[0]
    result = {
        "prediction": "FAKE" if pred == 1 else "REAL",
       
    }

    return result
