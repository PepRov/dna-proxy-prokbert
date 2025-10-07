from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from prokbert.models import BertForBinaryClassificationWithPooling
import torch
import torch.nn.functional as F

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
# Load ProkBERT model once on startup
# ------------------------------------------------------------
finetuned_model = "neuralbioinfo/prokbert-mini-promoter"
kmer = 6
shift = 1

tok_params = {'kmer': kmer, 'shift': shift}
tokenizer = ProkBERTTokenizer(tokenization_params=tok_params)
model = BertForBinaryClassificationWithPooling.from_pretrained(finetuned_model)
model.eval()  # Inference mode

# ------------------------------------------------------------
# Request model
# ------------------------------------------------------------
class SequenceRequest(BaseModel):
    sequence: str

# ------------------------------------------------------------
# Root route
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "ProkBERT promoter classifier running"}

# ------------------------------------------------------------
# Prediction route
# ------------------------------------------------------------
@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        sequence = req.sequence.strip().upper()

        # Tokenize
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}  # Add batch dim

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)
        prob_promoter = probs[0, 1].item()
        prob_non_promoter = probs[0, 0].item()

        label = "Promoter" if prob_promoter > prob_non_promoter else "Non-promoter"

        print("Sequence:", sequence)
        print("Label:", label)
        print("Prob promoter:", prob_promoter)
        print("Prob non-promoter:", prob_non_promoter)
        print("-------------------------------")

        return {
            "sequence": sequence,
            "prediction": label,
            "confidence": f"{prob_promoter:.4f}" if label == "Promoter" else f"{prob_non_promoter:.4f}"
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
