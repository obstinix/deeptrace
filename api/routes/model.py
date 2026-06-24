from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

class ReloadRequest(BaseModel):
    checkpoint_path: str

def _load_predictor(ckpt_path=None):
    try:
        from deepfake_recognition.inference.predictor import Predictor
        if ckpt_path is None:
            ckpt_path = os.getenv("MODEL_CHECKPOINT", "checkpoints/resnet18/best.pth")
        
        path = Path(ckpt_path)
        if path.exists():
            print(f"Loading checkpoint: {path}")
            return Predictor.from_checkpoint(path)
        
        # Fallback list
        for ckpt in [Path("checkpoints/resnet18/best.pth"),
                     Path("checkpoints/efficientnet_b3/best.pth")]:
            if ckpt.exists():
                print(f"Loading fallback checkpoint: {ckpt}")
                return Predictor.from_checkpoint(ckpt)
                
        print("WARNING: No checkpoint found. Train a model first.")
        return None
    except Exception as e:
        print(f"WARNING: Failed to load predictor: {e}")
        return None

@router.post("/api/model/reload")
async def reload_model(request: Request, req: ReloadRequest):
    new_predictor = _load_predictor(req.checkpoint_path)
    if new_predictor is None:
        raise HTTPException(status_code=400, detail="Failed to load checkpoint")
    request.app.state.predictor = new_predictor
    return {"status": "reloaded", "model_version": req.checkpoint_path, "val_accuracy": 0.0}
