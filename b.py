import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary

# -------------------------------
# Model Components (unchanged)
# -------------------------------

class SparseHashing(nn.Module):
    def __init__(self, in_dim, out_dim, hash_size=128, sparsity=0.2):
        super().__init__()
        self.hash_size = hash_size
        self.sparsity = sparsity
        self.hash_matrix = nn.Parameter(torch.randn(in_dim, hash_size) * 0.02)
        self.projection = nn.Parameter(torch.randn(hash_size, out_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x):
        # Compute hash codes
        hash_codes = torch.sigmoid(x @ self.hash_matrix)
        # Apply top-k sparsity to make it highly efficient
        if self.training:
            mask = torch.bernoulli(torch.ones_like(hash_codes) * (1 - self.sparsity))
            hash_codes = hash_codes * mask
        else:
            topk_values, _ = torch.topk(hash_codes, k=int(self.hash_size * (1 - self.sparsity)), dim=-1)
            kth_values = topk_values[:, :, -1].unsqueeze(-1)
            hash_codes = hash_codes * (hash_codes >= kth_values).float()
        return hash_codes @ self.projection + self.bias

class MultiscaleFrequencyFilters(nn.Module):
    """Memory-efficient multiscale frequency domain processing"""
    def __init__(self, channels, scales=3):
        super().__init__()
        self.scales = scales
        self.filters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 8, 1, bias=False, groups=1),
                nn.GELU(),
                nn.Conv2d(8, 2, 1, bias=False, groups=1)
            ) for _ in range(scales)
        ])
        
        def get_groups(ch, max_groups=8):
            ch_div = max(1, ch // 8)
            for i in range(min(max_groups, ch_div), 0, -1):
                if ch_div % i == 0 and ch % i == 0:
                    return i
            return 1
            
        ch_groups = get_groups(channels)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, groups=ch_groups),
            nn.GELU(),
            nn.Conv2d(channels // 8, channels, 1, groups=ch_groups),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x)
        out = torch.zeros_like(x)
        for i in range(min(self.scales, 3)):
            scale_factor = 2 ** i
            H_scale, W_scale = H // scale_factor, W // scale_factor
            if H_scale < 4 or W_scale < 4:
                continue
            batch_size = 4
            for b_start in range(0, B, batch_size):
                b_end = min(b_start + batch_size, B)
                for c_start in range(0, C, 16):
                    c_end = min(c_start + 16, C)
                    x_slice = x[b_start:b_end, c_start:c_end]
                    x_scale = F.interpolate(x_slice, size=(H_scale, W_scale), mode='bilinear', align_corners=False)
                    x_freq_scale = torch.fft.rfft2(x_scale)
                    real_imag = torch.stack([x_freq_scale.real, x_freq_scale.imag], dim=1)
                    real_imag = real_imag.reshape((b_end-b_start)*(c_end-c_start), 2, H_scale, W_scale//2+1)
                    real_imag = self.filters[i](real_imag)
                    real_imag = real_imag.reshape(b_end-b_start, c_end-c_start, 2, H_scale, W_scale//2+1)
                    filtered_freq = torch.complex(real_imag[:,:,0], real_imag[:,:,1])
                    filtered_spatial = torch.fft.irfft2(filtered_freq, s=(H_scale, W_scale))
                    filtered_spatial = F.interpolate(filtered_spatial, size=(H, W), mode='bilinear', align_corners=False)
                    out[b_start:b_end, c_start:c_end] = out[b_start:b_end, c_start:c_end] + filtered_spatial
        out = out * self.channel_attn(out)
        return out

class EfficientAttention(nn.Module):
    """Memory-efficient attention mechanism using linear complexity approximation"""
    def __init__(self, dim, heads=4, dim_head=24, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_q = SparseHashing(dim, inner_dim, hash_size=inner_dim//2)
        self.to_k = SparseHashing(dim, inner_dim, hash_size=inner_dim//2)
        self.to_v = SparseHashing(dim, inner_dim, hash_size=inner_dim//2)
        self.to_out = SparseHashing(inner_dim, dim, hash_size=inner_dim//2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        orig_H, orig_W = H, W
        if H * W > 32*32:
            scale_factor = min(1.0, np.sqrt(32*32 / (H * W)))
            H, W = int(H * scale_factor), int(W * scale_factor)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x_flat = x.flatten(2).transpose(1, 2)  # B, H*W, C
        q = self.to_q(x_flat)
        k = self.to_k(x_flat)
        v = self.to_v(x_flat)
        q = q.reshape(B, H*W, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.reshape(B, H*W, self.heads, self.dim_head).permute(0, 2, 3, 1)
        v = v.reshape(B, H*W, self.heads, self.dim_head).permute(0, 2, 1, 3)
        chunk_size = min(H*W, (H*W) // 4 + 1)
        out = torch.zeros_like(v)
        for i in range(0, H*W, chunk_size):
            end_idx = min(i + chunk_size, H*W)
            attn = torch.matmul(q[:, :, i:end_idx], k) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out[:, :, i:end_idx] = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, H*W, -1)
        out = self.to_out(out)
        out = out.transpose(1, 2).reshape(B, -1, H, W)
        if H != orig_H or W != orig_W:
            out = F.interpolate(out, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        return out

class FrequencyBiasedMLP(nn.Module):
    """Memory-efficient MLP with frequency bias"""
    def __init__(self, dim, expansion_factor=2):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        def get_groups(ch, max_groups=8):
            for i in range(max_groups, 0, -1):
                if ch % i == 0:
                    return i
            return 1
        conv_groups = get_groups(dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, groups=conv_groups),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, groups=conv_groups)
        )
        self.register_buffer('freq_bias', None)
        self.register_buffer('freq_scale', None)
        
    def forward(self, x):
        out = self.net(x)
        B, C, H, W = x.shape
        if self.freq_bias is None or self.freq_bias.shape[0] != C:
            self.freq_bias = nn.Parameter(torch.zeros(C, 1, 1)).to(x.device)
            self.freq_scale = nn.Parameter(torch.ones(C, 1, 1)).to(x.device)
        x_freq = torch.fft.rfft2(x)
        x_freq = x_freq * self.freq_scale.unsqueeze(-1) + self.freq_bias.unsqueeze(-1)
        x_freq = torch.fft.irfft2(x_freq, s=(x.shape[2], x.shape[3]))
        return out + x_freq * 0.1

class FeatureEnhancementBlock(nn.Module):
    """Memory-efficient feature enhancement block"""
    def __init__(self, channels, expansion=2):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels // 4), channels)
        if channels <= 64:
            self.path = MultiscaleFrequencyFilters(channels)
            self.use_dual_path = False
        else:
            self.path = EfficientAttention(channels, heads=min(4, channels//32))
            self.use_dual_path = False
        def get_groups(ch, max_groups=16):
            for i in range(max_groups, 0, -1):
                if ch % i == 0:
                    return i
            return 1
        conv_groups = get_groups(channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=conv_groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, groups=conv_groups)
        )
        self.drop_path_rate = 0.1
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        out = self.path(x)
        out = self.fusion(out)
        if self.training and torch.rand(1).item() < self.drop_path_rate:
            return residual
        else:
            return residual + out

class EnhancedFrequencyHashNetwork(nn.Module):
    def __init__(self, num_classes=1000, img_size=224, base_channels=24, 
                 depth=[1, 2, 4, 1], expansion=2):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(4)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(channels[0] // 4, channels[0]),
            nn.GELU()
        )
        self.stages = nn.ModuleList()
        stage1 = [FeatureEnhancementBlock(channels[0]) for _ in range(depth[0])]
        self.stages.append(nn.Sequential(*stage1))
        for i in range(1, 4):
            downsample = nn.Sequential(
                nn.Conv2d(channels[i-1], channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(channels[i] // 4, channels[i])
            )
            blocks = [downsample]
            blocks.extend([FeatureEnhancementBlock(channels[i]) for _ in range(depth[i])])
            self.stages.append(nn.Sequential(*blocks))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.classifier(x)
        return x

# -------------------------------
# Model Creation
# -------------------------------

def create_FHN_small(num_classes=1000):
    return EnhancedFrequencyHashNetwork(
        num_classes=num_classes,
        base_channels=24,
        depth=[1, 2, 4, 1]
    )

def create_FHN_base(num_classes=1000):
    return EnhancedFrequencyHashNetwork(
        num_classes=num_classes,
        base_channels=32,
        depth=[2, 3, 6, 2]
    )

def create_FHN_large(num_classes=1000):
    return EnhancedFrequencyHashNetwork(
        num_classes=num_classes,
        base_channels=40,
        depth=[3, 4, 9, 3]
    )

# Create the base model and print its summary
model = create_FHN_base(num_classes=100)
summary(model, (1, 3, 32, 32))

def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 * 1024)

print(f"Model size: {count_parameters_in_MB(model):.2f} MB")

# -------------------------------
# Training Setup
# -------------------------------

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

num_epochs = 100
criterion = nn.CrossEntropyLoss()

# Use AdamW optimizer and CosineAnnealingWarmRestarts scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Data augmentation for CIFAR-100
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

best_val_acc = 0.0

for epoch in range(1, num_epochs+1):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Optional: add resonance regularization loss here if desired.
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()
        pbar.set_postfix(loss=train_loss/total_train, acc=100.*correct_train/total_train)
    
    scheduler.step()
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    pbar_val = tqdm(test_loader, desc=f"Val Epoch {epoch}")
    with torch.no_grad():
        for inputs, targets in pbar_val:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_val += targets.size(0)
            correct_val += predicted.eq(targets).sum().item()
            pbar_val.set_postfix(loss=val_loss/total_val, acc=100.*correct_val/total_val)
    
    val_acc = 100. * correct_val / total_val
    print(f"Epoch {epoch}: Train Loss={train_loss/total_train:.4f}, Train Acc={100.*correct_train/total_train:.2f}%, Val Loss={val_loss/total_val:.4f}, Val Acc={val_acc:.2f}%")
    
    # Save the best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved best model at epoch {epoch} with Val Acc: {val_acc:.2f}%")

# -------------------------------
# Generate Classification Report and Save as PDF
# -------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
         inputs = inputs.to(device)
         outputs = model(inputs)
         preds = outputs.argmax(dim=1).cpu().numpy()
         all_preds.extend(preds)
         all_targets.extend(targets.numpy())

report = classification_report(all_targets, all_preds, digits=4)
print("Classification Report:")
print(report)

# Save classification report to a PDF file using matplotlib
plt.figure(figsize=(8.5, 11))
plt.text(0.01, 0.99, report, {'fontsize': 10, 'family': 'monospace'},
         verticalalignment='top', transform=plt.gca().transAxes)
plt.axis('off')
plt.savefig("classification_report.pdf", format="pdf")
plt.close()
