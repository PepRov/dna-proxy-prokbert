from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

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
# Load pipeline once on startup
# ------------------------------------------------------------
pipe = pipeline(
    "text-classification",
    model="neuralbioinfo/prokbert-mini-promoter",
    trust_remote_code=True
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
    return {"message": "ProkBERT promoter classifier running via pipeline"}

# ------------------------------------------------------------
# Prediction route
# ------------------------------------------------------------
@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        sequence = req.sequence.strip().upper()

        # Run the pipeline
        result = pipe(sequence)[0]  # Returns dict with 'label' and 'score'

        label = result["label"]      # e.g., "Promoter" or "Non-promoter"
        confidence = result["score"] # float probability

        # Log for debugging
        print("Sequence:", sequence)
        print("Label:", label)
        print("Confidence:", confidence)
        print("-------------------------------")

        return {
            "sequence": sequence,
            "prediction": label,
            "confidence": f"{confidence:.4f}"
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}

