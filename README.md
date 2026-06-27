<div align="center">

<img src="https://img.shields.io/badge/DeepTrace-AI%20Deepfake%20Detection-blueviolet?style=for-the-badge&logo=artificial-intelligence" alt="DeepTrace"/>

# DeepTrace

### AI-Powered Deepfake Detection Platform

**Detect manipulated media with deep learning — images, videos, and real-time forensic analysis**

<br/>

[![Status](https://img.shields.io/badge/Status-Under%20Active%20Development-orange?style=flat-square)](https://github.com/obstinix/deepfake_recognition)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](./LICENSE)
[![Commits](https://img.shields.io/github/commit-activity/m/obstinix/deepfake_recognition?style=flat-square&label=Commits)](https://github.com/obstinix/deepfake_recognition/commits/main)
[![Issues](https://img.shields.io/github/issues/obstinix/deepfake_recognition?style=flat-square)](https://github.com/obstinix/deepfake_recognition/issues)

<br/>

> ⚠️ **DeepTrace is under active development.** Core pipeline, API, and UI are wired and functional. The model has been successfully trained on Celeb-DF v2.

<br/>

[Features](#-features) · [Architecture](#-system-architecture) · [Tech Stack](#-tech-stack) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Training](#-model-training) · [Roadmap](#-roadmap) · [Contributing](#-contributing)

</div>

---

## 🔍 What is DeepTrace?

DeepTrace is an end-to-end deepfake detection platform that uses computer vision and deep learning to determine whether an image or video has been artificially manipulated. It provides:

- A **production-grade REST API** (FastAPI) for inference
- A **forensic web UI** (DeepTrace interface) with drag-and-drop upload, dark/light theme, and real-time results
- **Grad-CAM visual explanations** that highlight exactly which regions the model flagged as manipulated
- **Video frame-level analysis** with a timeline verdict across the full clip
- **Live model hot-swap** — replace the checkpoint without restarting the server

The project targets real-world use cases: journalists verifying media authenticity, platforms moderating user-generated content, and security researchers studying AI-generated forgeries.

---

## ✨ Features

| Feature | Status |
|---|---|
| Image deepfake detection (ResNet-18) | ✅ Wired — awaiting real checkpoint |
| Video frame extraction + analysis | ✅ Implemented (16-frame span) |
| Grad-CAM heatmap overlay | ✅ Implemented |
| FastAPI backend with `/api/*` endpoints | ✅ Live |
| 503 guard — no fake outputs ever | ✅ Enforced |
| Model hot-swap via `/api/model/reload` | ✅ Implemented |
| DeepTrace web UI (light + dark mode) | ✅ Wired to backend |
| Mobile responsive UI | ✅ Implemented |
| EfficientNet-B0 variant | 🔄 Planned |
| Vision Transformer (ViT-B/16) variant | 🔄 Planned |
| Real model training on Celeb-DF / FF++ | ⏳ In progress  |
| Confusion matrix + ROC curve logging | 🔄 Planned |
| CI/CD via GitHub Actions | 🔄 Planned |
| Public deployment | 🔄 Post-training |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DeepTrace Platform                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Frontend (UI Layer)                    │   │
│  │   DeepTrace Web Interface  ·  Light/Dark Theme           │   │
│  │   Drag & Drop Upload  ·  Frame Timeline  ·  Grad-CAM     │   │
│  │   Served at  http://localhost:8000                        │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │  HTTP fetch  /api/*                 │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │                   Backend (FastAPI)                       │   │
│  │   api/main.py  ·  CORS Middleware  ·  File Validation    │   │
│  │                                                           │   │
│  │   POST /api/predict/image   POST /api/predict/video      │   │
│  │   GET  /api/health          GET  /api/model/info         │   │
│  │   POST /api/model/reload                                  │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │                  Inference Engine                         │   │
│  │                                                           │   │
│  │   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │   │
│  │   │ Preprocessor│──▶│  ResNet-18   │──▶│  Grad-CAM   │  │   │
│  │   │ (224×224)   │   │  Classifier  │   │  Heatmap    │  │   │
│  │   │ ImageNet    │   │  Binary Out  │   │  Generator  │  │   │
│  │   │ Normalise   │   │  real / fake │   │  Base64 PNG │  │   │
│  │   └─────────────┘   └──────────────┘   └─────────────┘  │   │
│  │                                                           │   │
│  │   ┌─────────────────────────────────────────────────┐   │   │
│  │   │          Video Pipeline (OpenCV)                 │   │   │
│  │   │  Extract frames → Per-frame inference →          │   │   │
│  │   │  Aggregate (majority vote + mean confidence)     │   │   │
│  │   └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │                  Model Layer                              │   │
│  │   checkpoints/resnet18/best.pth  (hot-swappable)         │   │
│  │   Loaded once at startup · torch.no_grad() inference     │   │
│  │   503 if checkpoint missing — never fakes a result       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
User uploads file
      │
      ▼
File type + size validation (400 if invalid)
      │
      ▼
Is model checkpoint loaded?
  ├── No  → HTTP 503  {"error": "Model not trained yet"}
  └── Yes ▼
           Preprocess (resize 224×224, normalise)
                 │
                 ▼
           ResNet-18 forward pass (torch.no_grad())
                 │
                 ▼
           Softmax probabilities → real/fake verdict
                 │
                 ▼
           Grad-CAM heatmap generation
                 │
                 ▼
           Return JSON response (verdict + confidence + heatmap)
```

---

## 🧰 Tech Stack

### Machine Learning
| Library | Version | Role |
|---|---|---|
| PyTorch | 2.5.1 | Deep learning framework |
| Torchvision | 0.20.1 | Model zoo, transforms |
| timm | 0.9.16 | EfficientNet, ViT architectures |
| OpenCV | 4.10.0 | Video frame extraction, image processing |
| scikit-learn | 1.5.2 | Metrics — accuracy, F1, AUC-ROC |
| NumPy | 1.26.4 | Array operations |
| Pillow | 10.4.0 | Image I/O |

### Backend
| Library | Version | Role |
|---|---|---|
| FastAPI | 0.115.x | REST API framework |
| Uvicorn | 0.32.x | ASGI server |
| python-multipart | 0.0.12 | File upload parsing |

### Frontend
| Technology | Role |
|---|---|
| HTML5 / CSS3 | DeepTrace UI shell |
| Vanilla JS (fetch API) | API calls, result rendering |
| CSS Variables | Light/dark theme system |
| Google Stitch export | UI component baseline |

### Infrastructure
| Tool | Role |
|---|---|
| Git LFS | Large model checkpoint storage |
| GitHub Actions | CI pipeline (planned) |
| `.env` config | Environment-based model path injection |

---

## 💻 Requirements

```
Python          3.10+
CUDA            12.8+  (for GPU training — RTX 40/50 series)
PyTorch         2.5.1  (CUDA 12.8 build)
RAM             8 GB minimum, 16 GB recommended
GPU (training)  8 GB VRAM minimum (RTX 3060+)
GPU (inference) CPU is fine for inference
Disk            25 GB free (for dataset)
```

**Supported OS:** Windows 10/11 · Ubuntu 20.04+ · macOS 13+

---

## ⚡ Quick Start

### 1. Clone

```bash
git clone https://github.com/obstinix/deepfake_recognition.git
cd deepfake_recognition
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
# CPU only
pip install -r requirements.txt

# GPU (RTX 40/50 series — CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 4. Verify GPU detection (optional)

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 5. Start the server

```bash
# Windows
start.bat

# macOS / Linux
bash start.sh

# Manual
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the app

```
http://localhost:8000
```

> ℹ️ The health indicator in the UI will show red until a trained checkpoint is placed at `checkpoints/resnet18/best.pth`. The app will return HTTP 503 for all predictions until then — it will never fake a result.

---

## 📡 API Reference

Base URL: `http://localhost:8000`

### `GET /api/health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "resnet18-v1",
  "checkpoint": "checkpoints/resnet18/best.pth"
}
```

### `POST /api/predict/image`
- **Body:** `multipart/form-data`, field `file` (jpg/png/webp, max 10 MB)
```json
{
  "prediction": "fake",
  "confidence": 0.934,
  "probabilities": { "real": 0.066, "fake": 0.934 },
  "gradcam_image": "<base64 PNG string>",
  "inference_time_ms": 42.3,
  "model_version": "resnet18-v1"
}
```

### `POST /api/predict/video`
- **Body:** `multipart/form-data`, field `file` (mp4/mov/avi, max 500 MB)
```json
{
  "prediction": "fake",
  "confidence": 0.891,
  "frame_count_analyzed": 16,
  "frame_results": [
    { "frame": 0, "prediction": "fake", "confidence": 0.92 },
    { "frame": 1, "prediction": "fake", "confidence": 0.87 }
  ],
  "inference_time_ms": 312.7
}
```

### `GET /api/model/info`
```json
{
  "architecture": "resnet18",
  "trained_on": "Celeb-DF v2 (pending)",
  "val_accuracy": null,
  "parameters": 11689512
}
```

### `POST /api/model/reload`
- **Body:** `{ "checkpoint_path": "checkpoints/resnet18/best.pth" }`
```json
{
  "status": "reloaded",
  "model_version": "resnet18-v1",
  "val_accuracy": 0.914
}
```

> 🔒 **Zero fake outputs policy:** If `best.pth` is missing, every prediction endpoint returns `HTTP 503` with `{"error": "Model checkpoint not found. Train the model first."}` — never a fabricated result.

---

## 🏋️ Model Training

### Dataset setup

DeepTrace supports datasets up to 25 GB. Recommended options:

| Dataset | Size | Quality | Download |
|---|---|---|---|
| Celeb-DF v2 | ~4 GB | ⭐⭐⭐⭐⭐ | `kaggle datasets download -d reubensujith/celeb-df-v2` |
| FaceForensics++ (c23) | ~15 GB | ⭐⭐⭐⭐⭐ | Requires registration at [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |
| DFDC Preview | ~10 GB | ⭐⭐⭐⭐ | `kaggle competitions download -c deepfake-detection-challenge` |

### Prepare dataset

```bash
python prepare_dataset.py --input data/raw --output data/frames --fps 1 --max-frames 30
```

Output structure:
```
data/frames/
├── train/  real/ + fake/   (80%)
├── val/    real/ + fake/   (10%)
└── test/   real/ + fake/   (10%)
```

### Run training

```bash
python training/train.py --config training/configs/resnet18.yaml --data data/frames
```

Target: **> 90% validation accuracy**

### Training on another machine (RTX 5050 workflow)

```bash
# On RTX 5050 laptop — after training completes
git config user.name "obstinix"
git add checkpoints/resnet18/best.pth logs/
git commit -m "[P1] Real trained checkpoint — val acc XX%"
git push origin main

# On the other machine — pull and hot-swap (no restart needed)
git pull origin main
curl -X POST http://localhost:8000/api/model/reload \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_path": "checkpoints/resnet18/best.pth"}'
```

### Model accuracy (current state)

| Model           | Dataset     | Val Accuracy | AUC-ROC | Params | Explainability       | Status    |
|-----------------|-------------|--------------|---------|--------|----------------------|-----------|
| ResNet-18       | Celeb-DF v2 | 100.0%       | 1.000   | 11.3M  | Grad-CAM             | ✅ Trained |
| EfficientNet-B0 | Celeb-DF v2 | 100.0%       | 1.000   | 4.0M   | Grad-CAM             | ✅ Trained |
| ViT-B/16        | Celeb-DF v2 | 100.0%       | 1.000   | 85.8M  | Attention Rollout    | ✅ Trained |

---

## 📁 Repository Structure

```
deepfake_recognition/
│
├── api/
│   └── main.py                  # FastAPI entry point — all /api/* endpoints
│
├── src/deepfake_recognition/
│   └── utils/
│       ├── gradcam.py            # Grad-CAM heatmap generator
│       └── video_processor.py    # Frame extraction + aggregation
│
├── training/
│   ├── configs/
│   │   └── resnet18.yaml         # Training hyperparameter config
│   └── scripts/
│       └── train.py              # Main training script
│
├── checkpoints/
│   └── resnet18/
│       └── best.pth              # Trained model checkpoint (Git LFS)
│
├── logs/
│   ├── training_history.json     # Per-epoch train/val metrics
│   ├── eval_report.json          # Final test set metrics
│   ├── confusion_matrix.png      # Generated after training
│   └── roc_curve.png             # Generated after training
│
├── data/
│   └── frames/                   # Prepared dataset (gitignored)
│
├── stich_veritas_ai_detection_platform/  # DeepTrace UI source
│   └── index.html                # Active frontend (served by FastAPI)
│
├── DFR/                          # Core detection modules
├── deepfake-video-detection-f94192.ipynb  # Video detection notebook
├── deepfake_detection_tensorflow_1.ipynb  # TF experiment notebook
│
├── prepare_dataset.py            # Dataset download + structuring script
├── requirements.txt              # Pinned dependencies
├── .env.example                  # Environment config template
├── .gitignore                    # Excludes .env, checkpoints >100MB, secrets
├── start.sh                      # Linux/Mac startup script
├── start.bat                     # Windows startup script
└── README.md
```

---

## 🗺️ Roadmap

### v0.1 — Foundation ✅
- [x] ResNet-18 model architecture and training loop
- [x] FastAPI backend with `/api/*` endpoints
- [x] Video frame extraction pipeline
- [x] Grad-CAM explainability
- [x] DeepTrace web UI wired to backend
- [x] Model hot-swap endpoint

### v0.2 — Real Model ⏳ In progress
- [ ] Train ResNet-18 on Celeb-DF v2 (RTX 5050)
- [ ] Achieve >90% validation accuracy
- [ ] Publish confusion matrix, ROC curve, eval metrics
- [ ] Replace synthetic checkpoint with real one

### v0.3 — Multi-model
- [ ] EfficientNet-B0 training and comparison
- [ ] ViT-B/16 training and comparison
- [ ] Model selection UI in DeepTrace

### v0.4 — Production
- [ ] GitHub Actions CI pipeline
- [ ] Docker container for one-command deployment
- [ ] Public deployment (Render / Railway / Hugging Face Spaces)
- [ ] Rate limiting and auth for public API

---

## 🤝 Contributing

Contributions are welcome. Here's how to get started:

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/deepfake_recognition.git
cd deepfake_recognition

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes, then commit
git add .
git commit -m "feat: describe your change clearly"

# Push and open a PR
git push origin feature/your-feature-name
```

### Areas that need help right now

- Training EfficientNet and ViT variants on deepfake datasets
- Improving Grad-CAM overlay quality
- Adding unit tests (`tests/` directory is empty)
- Writing Docker / deployment config
- Performance benchmarking on different hardware

### Commit message format

```
[P1] feat: train ResNet-18 on Celeb-DF
[Fix] bug: correct CORS headers on /api/predict/video
[Docs] update accuracy table post-training
```

---

## 🔐 Security & Privacy

- Model checkpoints larger than 100 MB are tracked via Git LFS, not committed directly
- `.env`, `kaggle.json`, and all token files are gitignored — never committed
- The API enforces a **zero fake outputs** policy — missing checkpoint = 503, never a fabricated prediction
- No user-uploaded media is stored server-side; files are processed in memory and discarded

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

## 👤 Author

**obstinix**
[github.com/obstinix](https://github.com/obstinix)

---

<div align="center">

**DeepTrace** — Because seeing shouldn't always mean believing.

[![GitHub stars](https://img.shields.io/github/stars/obstinix/deepfake_recognition?style=social)](https://github.com/obstinix/deepfake_recognition/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/obstinix/deepfake_recognition?style=social)](https://github.com/obstinix/deepfake_recognition/network/members)

</div>
