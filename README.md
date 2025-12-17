# DR-MoEViT-UNet: Colorectal Cancer Segmentation & Classification

A deep learning model that combines **Vision Transformer (ViT)**, **Mixture of Experts (MoE)**, and **U-Net decoder** for simultaneous segmentation and classification of colorectal histopathology images.

## 🎯 Project Overview

This project addresses automated colorectal cancer detection from histopathology images. The model performs two tasks:
1. **Segmentation** - Identifies WHERE cancerous tissue is located (pixel-level)
2. **Classification** - Determines WHAT TYPE of tissue it is (6 classes)

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 86.72% |
| IoU | 81.21% |
| F1-Score | 89.61% |
| Precision | 88.41% |
| Recall | 90.90% |

## 🏗️ Model Architecture

- **Encoder**: Vision Transformer (ViT) with 12 layers, 12 attention heads
- **Dynamic Routing**: Mixture of Experts (4 experts per layer)
- **Decoder**: U-Net style with skip connections
- **Output**: Binary segmentation mask + 6-class classification

## 📁 Files

| File | Description |
|------|-------------|
| `DR_MoEViT_UNet_DriveVersion (original).ipynb` | Main training notebook (Google Colab) |
| `gradio_app.py` | Web-based GUI for inference |
| `inference_correct_only.py` | Inference script with visualization |
| `latest_checkpoint.pth` | Trained model weights (1.1GB, Git LFS) |
| `3_BSAI204_FYP_CV_Task_Report.md` | Project report |

## 🚀 Quick Start

### Run the GUI
```bash
pip install torch torchvision gradio pillow numpy
python gradio_app.py
```
Then open http://127.0.0.1:7860 in your browser.

### Download Model Weights
The model checkpoint is stored using Git LFS. After cloning:
```bash
git lfs pull
```

## 📚 Dataset

**EBHI-SEG** - Colorectal Cancer Histopathology Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/orvile/ebhi-seg-colorectal-cancer)
- 6 Classes: Adenocarcinoma, High-grade IN, Low-grade IN, Normal, Polyp, Serrated adenoma
- 2226 image-mask pairs

## 🛠️ Training

Trained on Google Colab with:
- GPU: Tesla T4
- Epochs: 50
- Batch Size: 16
- Optimizer: AdamW (lr=1e-4)
- Loss: Dice + Focal + CrossEntropy

## 👤 Author

- **Group**: 3
- **Roll Number**: BSAI204
- **Course**: Computer Vision - Final Year Project

## 📄 License

This project is for educational purposes (FYP submission).
