# DeepTrace GPU Training Run Log

This log documents the training session executed on local GPU, transitioning DeepTrace from a synthetic-data pipeline to real deepfake classification.

## Hardware & Environment
- **GPU:** NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM)
- **CUDA Version:** Driver UMD CUDA 13.3
- **Python Version:** 3.11.0
- **PyTorch Build:** `2.11.0+cu128` (Blackwell sm_120 compatible build)
- **Torchvision Build:** `0.26.0+cu128`

## Dataset: Kaggle 140k Real and Fake Faces
- **Type:** Image-level (Real: Flickr faces, Fake: StyleGAN-generated faces).
- **Distinction:** This dataset evaluates general fake-face/StyleGAN detection, not frame-to-frame temporal artifacts of video-swapping deepfakes (like Celeb-DF).
- **Integrity Checks:**
  - Split Folders: Preserved and verified.
  - De-duplication: Verified by file hash MD5 across splits (0 duplicates found).
- **Splits Summary:**
  - **Train:** 100,000 images (50.0% real / 50.0% fake)
  - **Val:** 20,000 images (50.0% real / 50.0% fake)
  - **Test:** 20,000 images (50.0% real / 50.0% fake)
  - **Total:** 140,000 images

## Training Progress & Status
The entire training pipeline is currently running sequentially in the background using `scripts/train_pipeline.py`.
- **Pipeline Log:** [pipeline.log](file:///C:/Users/Piyush/Documents/antigravity/sharp-brahmagupta/logs/pipeline.log)
- **Active Model Log:** [train.log](file:///C:/Users/Piyush/Documents/antigravity/sharp-brahmagupta/logs/resnet18/train.log) (ResNet18 actively training)

### Performance & Run Metrics (To Be Updated post-run)

| Architecture | Epochs | Batch Size | Wall-Clock Time | Best Val Acc | Test Acc | Test AUC | Overfit Warning? |
|---|---|---|---|---|---|---|---|
| resnet18 | 35 | 64 | *Running* | *TBD* | *TBD* | *TBD* | *TBD* |
| efficientnet_b3 | 50 | 16 | *Pending* | *TBD* | *TBD* | *TBD* | *TBD* |
| vit_base | 30 | 16 | *Pending* | *TBD* | *TBD* | *TBD* | *TBD* |
| efficientnet_b0 | 30 | 64 | *Pending* | *TBD* | *TBD* | *TBD* | *TBD* |
| vit_b16 | 30 | 16 | *Pending* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Ensemble (Learned)** | N/A | N/A | *Pending* | *TBD* | *TBD* | *TBD* | *TBD* |

*Note: Individual plots of training history are saved dynamically to `training/logs/<arch>/loss_curves.png` at the end of each run.*
