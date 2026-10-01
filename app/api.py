import os
import sys
import io
import yaml
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.device import get_device
from src.data.vocabulary import Vocabulary
from src.models.cnn_lstm import CNNLSTMCaptioner
from src.training.checkpoint import load_checkpoint
from src.inference.generate import CaptionGenerator

app = FastAPI(
    title="AI Image Caption Generation API",
    description="REST API for Deep Learning Image Captioning with Attention & Beam Search",
    version="1.0.0"
)

# Global model state
generator = None
vocab = None

class PredictionResponse(BaseModel):
    caption: str
    decoding: str
    score: float = 0.0

@app.on_event("startup")
def startup_event():
    global generator, vocab
    config_path = "configs/config.yaml"
    vocab_path = "data/processed/vocab.json"
    checkpoint_path = "checkpoints/best_model.pth"

    if os.path.exists(config_path) and os.path.exists(vocab_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        device = get_device()
        vocab = Vocabulary.load(vocab_path)
        model = CNNLSTMCaptioner(
            vocab_size=len(vocab),
            backbone=config["model"]["encoder"]["backbone"],
            embed_size=config["model"]["encoder"]["embed_size"],
            hidden_size=config["model"]["decoder"]["hidden_size"],
            attention_dim=config["model"]["decoder"]["attention_dim"],
            dropout=config["model"]["decoder"]["dropout"]
        )

        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, model=model, device=device)

        generator = CaptionGenerator(model=model, vocab=vocab, device=device)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": generator is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_caption(
    file: UploadFile = File(...),
    method: str = Query("beam", enum=["greedy", "beam"]),
    beam_size: int = Query(3, ge=1, le=10)
):
    if generator is None:
        raise HTTPException(status_code=500, detail="Captioning model is not initialized.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        if method == "beam":
            res = generator.generate_beam_search(pil_img, beam_size=beam_size)
            score = res.get("score", 0.0)
        else:
            res = generator.generate_greedy(pil_img)
            score = 0.0

        return PredictionResponse(
            caption=res["caption"],
            decoding=res["decoding"],
            score=round(score, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image captioning failed: {str(e)}")
