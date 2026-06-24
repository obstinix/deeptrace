"""Inference wrapper — loads checkpoint, predicts images and videos."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from deepfake_recognition.data.transforms import get_val_transforms, get_tta_transforms
from deepfake_recognition.utils.gradcam import generate_gradcam_base64


class Predictor:
    def __init__(self, model, device: str, img_size: int = 224):
        self.model = model.to(device).eval()
        self.device = device
        self.val_tf = get_val_transforms(img_size)
        self.tta_tfs = get_tta_transforms(img_size)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path,
                         device: str = "auto") -> "Predictor":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else \
                     "mps" if torch.backends.mps.is_available() else "cpu"
        ckpt = torch.load(checkpoint_path, map_location=device)
        path_str = str(checkpoint_path).lower()
        if "efficientnet" in path_str:
            from deepfake_recognition.models.efficientnet import DeepfakeEfficientNet
            model = DeepfakeEfficientNet(pretrained=False, freeze_backbone=False)
            img_size = 300
        elif "vit" in path_str:
            from deepfake_recognition.models.vit import DeepfakeViT
            model = DeepfakeViT(pretrained=False, freeze_backbone=False)
            img_size = 224
        else:
            from deepfake_recognition.models.resnet import DeepfakeResNet18
            model = DeepfakeResNet18(pretrained=False, freeze_backbone=False)
            img_size = 224
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}")
        return cls(model, device, img_size)

    @torch.no_grad()
    def predict_pil(self, img: Image.Image, use_tta: bool = False) -> dict:
        input_tensor = self.val_tf(img).unsqueeze(0).to(self.device)
        if use_tta:
            logits = torch.stack([self.model(tf(img).unsqueeze(0).to(self.device))
                                   for tf in self.tta_tfs]).mean(0)
        else:
            logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        prob_fake, prob_real = probs[1].item(), probs[0].item()
        
        # Generate Grad-CAM (using gradients necessitates requires_grad setup if model is frozen, but we will temporarily enable grad)
        gradcam_b64 = None
        try:
            with torch.enable_grad():
                input_tensor_grad = input_tensor.clone().requires_grad_(True)
                gradcam_b64 = generate_gradcam_base64(self.model, input_tensor_grad, img)
        except Exception as e:
            print(f"WARNING: GradCAM generation failed: {e}")
            
        return {"label": "fake" if prob_fake > 0.5 else "real",
                "confidence": max(prob_real, prob_fake),
                "prob_real": prob_real, "prob_fake": prob_fake,
                "gradcam_image": gradcam_b64}

    @torch.no_grad()
    def predict_video(self, video_path: str, n_frames: int = 16) -> dict:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        preds = []
        for i in range(n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n_frames))
            ret, frame = cap.read()
            if not ret: continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            p = self.predict_pil(img)
            preds.append({"frame_idx": int(i * total / n_frames),
                           "label": p["label"], "prob_fake": round(p["prob_fake"], 4)})
        cap.release()
        if not preds:
            return {"label": "unknown", "confidence": 0.0, "frames_analyzed": 0}
        avg_fake = sum(p["prob_fake"] for p in preds) / len(preds)
        return {"label": "fake" if avg_fake > 0.5 else "real",
                "confidence": round(max(avg_fake, 1 - avg_fake), 4),
                "frames_analyzed": len(preds), "frame_predictions": preds}
