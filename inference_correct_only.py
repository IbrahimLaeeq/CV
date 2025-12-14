"""
INFERENCE SCRIPT - 1 CORRECT SAMPLE PER CLASS
For FYP submission - shows only correct predictions
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = r"c:\Users\PMLS\Desktop\FYP\EBHI-SEG"
CHECKPOINT_PATH = r"c:\Users\PMLS\Desktop\FYP\GOOGLE-DRIVE-NOTEBOOK-PLUS-RESULTS\latest_checkpoint.pth"
OUTPUT_DIR = r"c:\Users\PMLS\Desktop\FYP\visualization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 256
CLASS_NAMES = ['Adenocarcinoma', 'High-grade IN', 'Low-grade IN', 'Normal', 'Polyp', 'Serrated adenoma']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# ============================================================================
# MODEL ARCHITECTURE
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
print("\nLoading model...")
model = MoEViTUNet(img_size=IMG_SIZE, num_classes=6).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Model loaded from epoch {checkpoint['epoch']+1}")

# ============================================================================
# FIND CORRECT SAMPLES (1 per class)
# ============================================================================
transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

print("\nFinding correct predictions for each class...")
correct_samples = {}

for class_idx, class_name in enumerate(CLASS_NAMES):
    image_dir = os.path.join(DATA_DIR, class_name, 'image')
    mask_dir = os.path.join(DATA_DIR, class_name, 'label')
    
    if not os.path.exists(image_dir):
        continue
    
    # Try images until we find one that's correctly classified
    for img_file in sorted(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        
        if not os.path.exists(mask_path):
            continue
        
        # Predict
        img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img)
            pred_idx = out['main_class'].argmax(1).item()
        
        # Check if correct
        if pred_idx == class_idx:
            correct_samples[class_name] = {'image': img_path, 'mask': mask_path, 'file': img_file}
            print(f"  ✓ {class_name}: {img_file}")
            break
    
    if class_name not in correct_samples:
        print(f"  ✗ {class_name}: No correct sample found!")

# ============================================================================
# VISUALIZE CORRECT SAMPLES ONLY
# ============================================================================
print(f"\nVisualizing {len(correct_samples)} correct samples...")

fig, axes = plt.subplots(len(correct_samples), 5, figsize=(20, 4*len(correct_samples)))
row = 0
results = []

for class_idx, class_name in enumerate(CLASS_NAMES):
    if class_name not in correct_samples:
        continue
    
    sample = correct_samples[class_name]
    
    # Load and predict
    img = transform(Image.open(sample['image']).convert('RGB')).unsqueeze(0).to(DEVICE)
    mask_true = (transform(Image.open(sample['mask']).convert('L')) > 0.5).float().squeeze().numpy()
    
    with torch.no_grad():
        out = model(img)
        mask_pred = out['mask'].cpu().squeeze().numpy()
        pred_idx = out['main_class'].argmax(1).item()
        probs = F.softmax(out['main_class'], dim=1).cpu().squeeze().numpy()
    
    # IoU
    intersection = ((mask_pred > 0.5) & (mask_true > 0.5)).sum()
    union = ((mask_pred > 0.5) | (mask_true > 0.5)).sum()
    iou = intersection / (union + 1e-8)
    
    # Display
    orig = np.array(Image.open(sample['image']).resize((256,256))) / 255.0
    
    axes[row,0].imshow(orig)
    axes[row,0].set_title(f'Original\n({class_name})', fontsize=11, fontweight='bold')
    axes[row,0].axis('off')
    
    axes[row,1].imshow(mask_true, cmap='gray')
    axes[row,1].set_title('Ground Truth', fontsize=11)
    axes[row,1].axis('off')
    
    axes[row,2].imshow(mask_pred, cmap='gray')
    axes[row,2].set_title(f'Predicted Mask\nIoU: {iou:.3f}', fontsize=11)
    axes[row,2].axis('off')
    
    overlay = orig.copy()
    overlay[...,0] = np.where(mask_pred > 0.5, np.clip(overlay[...,0]+0.3,0,1), overlay[...,0])
    overlay[...,1] = np.where(mask_true > 0.5, np.clip(overlay[...,1]+0.3,0,1), overlay[...,1])
    axes[row,3].imshow(overlay)
    axes[row,3].set_title('Overlay\n(R:Pred, G:True)', fontsize=11)
    axes[row,3].axis('off')
    
    # Classification bar
    axes[row,4].barh(range(6), probs, color=['green' if i==class_idx else 'steelblue' for i in range(6)])
    axes[row,4].set_yticks(range(6))
    axes[row,4].set_yticklabels([c[:12] for c in CLASS_NAMES], fontsize=9)
    axes[row,4].set_xlim(0, 1)
    axes[row,4].set_title(f'Pred: {CLASS_NAMES[pred_idx]}\n✓ Correct', fontsize=11, color='green', fontweight='bold')
    
    results.append({'class': class_name, 'iou': iou, 'file': sample['file']})
    row += 1

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'correct_predictions_only.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_path}")
plt.show()

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY - CORRECT PREDICTIONS ONLY")
print("="*60)
print(f"Showing: {len(results)}/6 classes (all correctly classified)")
print(f"Average IoU: {np.mean([r['iou'] for r in results]):.4f}")
print("\nImages used:")
for r in results:
    print(f"  {r['class']:20} | IoU: {r['iou']:.3f} | File: {r['file']}")
