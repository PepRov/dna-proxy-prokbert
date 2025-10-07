from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SequenceRequest(BaseModel):
    sequence: str

@app.get("/")
def root():
    return {"message": "Proxy server running"}

@app.post("/predict")
def predict(req: SequenceRequest):
    sequence = req.sequence.strip()

    try:
        hf_url = "https://hf.space/embed/neuralbioinfo/prokbert-mini-promoter/api/predict"
        response = requests.post(hf_url, json={"data": [sequence]})
        response.raise_for_status()
        hf_result = response.json()

        # Extract label and confidence
        label, confidence = hf_result["data"][0]

        return {
            "sequence": sequence,
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
