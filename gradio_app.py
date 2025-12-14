"""
DR-MoEViT-UNet Gradio App
Using exact same model architecture as inference_correct_only.py
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Config
CHECKPOINT_PATH = r"c:\Users\PMLS\Desktop\FYP\GOOGLE-DRIVE-NOTEBOOK-PLUS-RESULTS\latest_checkpoint.pth"
IMG_SIZE = 256
CLASS_NAMES = ['Adenocarcinoma', 'High-grade IN', 'Low-grade IN', 'Normal', 'Polyp', 'Serrated adenoma']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL ARCHITECTURE - EXACT COPY FROM WORKING INFERENCE SCRIPT
# ============================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x), attn

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        x_attn, attn = self.attn(self.norm1(x))
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x, attn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) + self.pos_embed
        attentions = []
        for block in self.blocks:
            x, attn = block(x)
            attentions.append(attn)
        return self.norm(x), attentions

class ExpertModule(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)

class DynamicRouter(nn.Module):
    def __init__(self, channels, num_experts):
        super().__init__()
        self.router = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1), nn.BatchNorm2d(channels//2), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels//2, num_experts), nn.Softmax(dim=1)
        )
    def forward(self, x): return self.router(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([ExpertModule(in_ch, mid_ch, out_ch) for _ in range(num_experts)])
        self.router = DynamicRouter(in_ch, num_experts)
    def forward(self, x):
        weights = self.router(x)
        outputs = torch.stack([e(x) for e in self.experts], dim=1)
        weights = weights.view(x.size(0), len(self.experts), 1, 1, 1)
        return (outputs * weights).sum(dim=1), weights.squeeze()

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None: x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_ch, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

class MoEViTUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=6, num_experts=4):
        super().__init__()
        self.vit = VisionTransformer(img_size, patch_size, 3, embed_dim, depth, num_heads)
        self.patch_size, self.embed_dim = patch_size, embed_dim
        dims = [256, 128, 64, 32]
        self.conv_transform = nn.Conv2d(embed_dim, dims[0], 1)
        self.moe1 = MixtureOfExperts(dims[0], dims[0], dims[0], num_experts)
        self.moe2 = MixtureOfExperts(dims[0], dims[0], dims[1], num_experts)
        self.moe3 = MixtureOfExperts(dims[1], dims[1], dims[2], num_experts)
        self.moe4 = MixtureOfExperts(dims[2], dims[2], dims[3], num_experts)
        self.decoder1 = DecoderBlock(dims[0], dims[1])
        self.decoder2 = DecoderBlock(dims[1]*2, dims[2])
        self.decoder3 = DecoderBlock(dims[2]*2, dims[3])
        self.decoder4 = DecoderBlock(dims[3], dims[3])
        self.aux1 = AuxiliaryClassifier(dims[0], num_classes)
        self.aux2 = AuxiliaryClassifier(dims[1], num_classes)
        self.aux3 = AuxiliaryClassifier(dims[2], num_classes)
        self.aux4 = AuxiliaryClassifier(dims[3], num_classes)
        self.final_conv = nn.Conv2d(dims[3], 1, 1)
        self.main_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dims[3], 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        vit_out, attns = self.vit(x)
        feat = vit_out[:, 1:, :]
        h = w = int(np.sqrt(feat.shape[1]))
        feat = feat.reshape(feat.shape[0], h, w, self.embed_dim).permute(0, 3, 1, 2)
        x0 = self.conv_transform(feat)
        x1, w1 = self.moe1(x0)
        x2, w2 = self.moe2(x1)
        x3, w3 = self.moe3(x2)
        x4, w4 = self.moe4(x3)
        d1 = self.decoder1(x1)
        x2_r = F.interpolate(x2, size=d1.shape[2:], mode='bilinear', align_corners=True) if d1.shape[2:] != x2.shape[2:] else x2
        d2 = self.decoder2(torch.cat([d1, x2_r], dim=1))
        x3_r = F.interpolate(x3, size=d2.shape[2:], mode='bilinear', align_corners=True) if d2.shape[2:] != x3.shape[2:] else x3
        d3 = self.decoder3(torch.cat([d2, x3_r], dim=1))
        d4 = self.decoder4(d3)
        d4_up = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=True)
        return {
            'mask': torch.sigmoid(self.final_conv(d4_up)),
            'main_class': self.main_classifier(d4),
        }

# ============================================================================
# LOAD MODEL
# ============================================================================
print("Loading model...")
model = MoEViTUNet(img_size=IMG_SIZE, num_classes=6).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Model loaded! (Device: {DEVICE})")

transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

# ============================================================================
# ANALYZE FUNCTION
# ============================================================================
def analyze_image(image):
    if image is None:
        return None, None, "Please upload an image"
    
    # Convert to PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        mask = outputs['mask'].cpu().squeeze().numpy()
        probs = F.softmax(outputs['main_class'], dim=1).cpu().squeeze().numpy()
        pred_idx = probs.argmax()
    
    # Binary mask
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary_mask)
    
    # Overlay - red on detected areas
    orig_resized = image.resize((IMG_SIZE, IMG_SIZE))
    overlay = np.array(orig_resized).astype(np.float32)
    m = mask > 0.5
    overlay[m, 0] = np.clip(overlay[m, 0] * 0.5 + 180, 0, 255)
    overlay[m, 1] = overlay[m, 1] * 0.6
    overlay[m, 2] = overlay[m, 2] * 0.6
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    
    # Result
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100
    result_text = f"## {pred_class}\n\n**Confidence: {confidence:.1f}%**"
    
    return mask_img, overlay_img, result_text

# ============================================================================
# GRADIO INTERFACE
# ============================================================================
demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(label="📷 Upload Histology Image", type="pil", height=300),
    outputs=[
        gr.Image(label="🎯 Segmentation Mask (White = Cancer tissue)", height=300),
        gr.Image(label="🔴 Detection Overlay (Red = Detected areas)", height=300),
        gr.Markdown(label="🏥 Classification Result")
    ],
    title="🔬 DR-MoEViT-UNet",
    description="**Colorectal Cancer Segmentation & Classification**\n\nUpload a colorectal histology image to detect and classify cancer tissue."
)

if __name__ == "__main__":
    demo.launch(share=False)
