from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

# ------------------------------------------------------------
# FastAPI setup
# ------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Input data format
# ------------------------------------------------------------
class SequenceRequest(BaseModel):
    sequence: str

# ------------------------------------------------------------
# Root route
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Proxy server running and forwarding to HF Space"}

# ------------------------------------------------------------
# Prediction route
# ------------------------------------------------------------
@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        sequence = req.sequence.strip()

        # HF Space API URL
        hf_space_url = "https://hf.space/embed/neuralbioinfo/prokbert-mini-promoter/api/predict"

        # Forward request to HF Space
        response = requests.post(hf_space_url, json={"data": [sequence]})
        response.raise_for_status()
        hf_result = response.json()

        # HF Space returns: hf_result["data"] = [[label, confidence]]
        label, confidence = hf_result["data"][0]

        # Log
        print("Sequence:", sequence)
        print("Label:", label)
        print("Confidence:", confidence)
        print("-------------------------------")

        return {
            "sequence": sequence,
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
