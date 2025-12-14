# 3_BSAI204_FYP_CV_Task_Report

---

**Project Video Link:** [INSERT YOUR VIDEO LINK HERE]

**GitHub Repository:** [INSERT YOUR GITHUB REPO LINK HERE]

---

**Group Number:** 3  
**Roll Number:** BSAI204  
**Course:** Computer Vision  
**Submission Date:** December 14, 2025

---

## Table of Contents
1. Introduction
2. Literature Review
3. Solution Approach
4. Implementation
5. Results & Discussion
6. Conclusion
7. References

---

## 1. Introduction

### 1.1 Problem Statement
Colorectal cancer (CRC) is the third most common cancer worldwide and the second leading cause of cancer-related deaths. Early detection and accurate classification of colorectal tissue abnormalities are crucial for effective treatment and improved patient outcomes. Manual histopathological examination is time-consuming, subject to inter-observer variability, and requires highly skilled pathologists.

### 1.2 Importance
Automated computer vision systems for colorectal cancer detection can:
- Reduce diagnostic time and workload for pathologists
- Provide consistent and objective analysis
- Enable early detection of precancerous lesions
- Improve patient survival rates through timely intervention

### 1.3 Objective
This project implements **DR-MoEViT-UNet**, a novel deep learning architecture for simultaneous segmentation and classification of colorectal histopathology images. The model combines:
- **Vision Transformer (ViT)** for global feature extraction
- **Dynamic Routing Mixture of Experts (MoE)** for specialized feature learning
- **U-Net decoder** for precise segmentation

---

## 2. Literature Review

### 2.1 Medical Image Segmentation
Medical image segmentation has evolved significantly with deep learning. Key architectures include:

**U-Net (Ronneberger et al., 2015):** The foundational architecture for medical image segmentation, featuring encoder-decoder structure with skip connections for preserving spatial information.

**Attention U-Net (Oktay et al., 2018):** Introduced attention gates to focus on relevant features, improving segmentation accuracy in medical imaging.

### 2.2 Vision Transformers in Medical Imaging
**Vision Transformer (ViT) (Dosovitskiy et al., 2020):** Revolutionized image analysis by applying transformer architecture to image patches, capturing long-range dependencies effectively.

**TransUNet (Chen et al., 2021):** Combined CNNs and Transformers for medical image segmentation, leveraging both local and global feature representations.

**Swin-UNet (Cao et al., 2021):** Applied Swin Transformer with U-shaped architecture for medical image segmentation with hierarchical feature maps.

### 2.3 Mixture of Experts
**Mixture of Experts (MoE) (Jacobs et al., 1991):** A technique that uses multiple specialized neural networks (experts) with a gating mechanism to route inputs to appropriate experts.

**Switch Transformer (Fedus et al., 2022):** Scaled MoE to trillion-parameter models with efficient sparse routing.

### 2.4 Colorectal Cancer Detection
Previous works on EBHI-SEG dataset have achieved varying levels of accuracy using CNNs and hybrid architectures. Our approach aims to leverage the strengths of transformers, MoE, and U-Net for improved performance.

---

## 3. Solution Approach

### 3.1 Model Architecture: DR-MoEViT-UNet

The proposed architecture consists of three main components:

#### 3.1.1 Vision Transformer Encoder
- **Patch Embedding:** Divides input images (256×256) into 16×16 patches
- **Positional Encoding:** Learnable position embeddings
- **Transformer Blocks:** 12 layers with multi-head self-attention (12 heads)
- **Embedding Dimension:** 768

#### 3.1.2 Dynamic Routing Mixture of Experts (DR-MoE)
- **Number of Experts:** 4 per MoE layer
- **Dynamic Router:** Learns to route features to specialized experts based on input characteristics
- **Expert Modules:** Convolutional networks for specialized feature processing
- **Soft Routing:** Weighted combination of expert outputs

#### 3.1.3 U-Net Decoder
- **Decoder Blocks:** 4 upsampling stages with skip connections
- **Channel Dimensions:** 256 → 128 → 64 → 32
- **Output:** Binary segmentation mask + 6-class classification

### 3.2 Multi-Task Learning
The model performs two tasks simultaneously:
1. **Segmentation:** Pixel-wise binary mask of tissue regions
2. **Classification:** 6-class cancer type classification
   - Adenocarcinoma
   - High-grade Intraepithelial Neoplasia (IN)
   - Low-grade Intraepithelial Neoplasia (IN)
   - Normal
   - Polyp
   - Serrated Adenoma

### 3.3 Loss Function
Combined loss for multi-task learning:
```
Total Loss = λ₁ × Dice Loss + λ₂ × BCE Loss + λ₃ × Cross-Entropy Loss
```
Where:
- Dice Loss: For segmentation quality
- BCE Loss: Binary cross-entropy for mask prediction
- Cross-Entropy Loss: For classification

---

## 4. Implementation

### 4.1 Dataset: EBHI-SEG
- **Source:** Kaggle (luisedrt/colorectal-histology-ebhi-seg)
- **Classes:** 6 categories of colorectal tissue
- **Structure:** Each class contains paired images and segmentation masks
- **Image Size:** Resized to 256×256 for training

### 4.2 Data Preprocessing
- Resizing to 256×256 pixels
- Normalization to [0, 1] range
- Train/Validation/Test split: 70%/15%/15%

### 4.3 Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Batch Size | 8 |
| Epochs | 50 |
| Device | CUDA (T4 GPU) |
| Framework | PyTorch |

### 4.4 Training Environment
- Platform: Google Colab
- GPU: Tesla T4 (16GB VRAM)
- Training Time: ~4-5 hours for 50 epochs

### 4.5 Code Structure
```
FYP/
├── DR_MoEViT_UNet_DriveVersion.ipynb  # Main training notebook
├── gradio_app.py                       # Web GUI application
├── inference_correct_only.py           # Inference visualization
├── EBHI-SEG/                           # Dataset
│   ├── Adenocarcinoma/
│   ├── High-grade IN/
│   ├── Low-grade IN/
│   ├── Normal/
│   ├── Polyp/
│   └── Serrated adenoma/
└── GOOGLE-DRIVE-NOTEBOOK-PLUS-RESULTS/
    ├── latest_checkpoint.pth           # Trained model
    ├── confusion_matrices.png
    └── training_history.png
```

---

## 5. Results & Discussion

### 5.1 Final Test Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 86.72% |
| **Precision** | 88.41% |
| **Recall** | 90.90% |
| **F1-Score** | 89.61% |
| **IoU (Intersection over Union)** | 81.21% |
| **Specificity** | 79.47% |

### 5.2 Training Progress
The model was trained for 50 epochs with consistent improvement:
- **Initial Train Loss:** 1.81 → **Final Train Loss:** 0.22
- **Initial Val Loss:** 1.75 → **Final Val Loss:** 1.18
- **Initial IoU:** 0.61 → **Final IoU:** 0.81

### 5.3 Per-Class Performance
The model successfully classifies all 6 tissue types:
1. **Adenocarcinoma:** High confidence detection
2. **High-grade IN:** Accurately distinguished from low-grade
3. **Low-grade IN:** Good segmentation of tissue boundaries
4. **Normal:** Clear differentiation from abnormal tissue
5. **Polyp:** Distinctive gland patterns identified
6. **Serrated Adenoma:** Complex patterns captured correctly

### 5.4 Challenges Faced
1. **GPU Memory:** Large ViT backbone required batch size optimization
2. **Overfitting:** Addressed with dropout and data augmentation
3. **Class Imbalance:** Some classes had fewer samples
4. **Validation Loss Fluctuation:** Expected with complex multi-task learning

### 5.5 GUI Application
A Gradio-based web interface was developed for practical usage:
- Accepts histopathology image uploads
- Displays segmentation mask (white = detected tissue)
- Shows overlay visualization (red = detected regions)
- Provides classification result with confidence percentage

---

## 6. Conclusion

### 6.1 Summary
This project successfully implemented DR-MoEViT-UNet, a novel architecture combining Vision Transformers, Mixture of Experts, and U-Net for colorectal cancer segmentation and classification. Key achievements:

- **81.21% IoU** for tissue segmentation
- **86.72% accuracy** for 6-class classification
- **Functional GUI** for practical deployment
- **End-to-end pipeline** from training to inference

### 6.2 Future Work
1. **Larger Dataset:** Train on more diverse histopathology datasets
2. **Model Optimization:** Quantization and pruning for faster inference
3. **Multi-Scale Processing:** Handle varying image resolutions
4. **Clinical Validation:** Testing with real-world clinical data
5. **Mobile Deployment:** Lightweight version for edge devices

### 6.3 Contributions
This work demonstrates the effectiveness of combining transformer architectures with mixture of experts for medical image analysis, providing a foundation for automated diagnostic assistance in colorectal cancer screening.

---

## 7. References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.

2. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.

3. Chen, J., et al. (2021). TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. arXiv.

4. Cao, H., et al. (2021). Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation. ECCV Workshop.

5. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models. JMLR.

6. EBHI-SEG Dataset: Available at Kaggle (luisedrt/colorectal-histology-ebhi-seg)

---

**End of Report**
