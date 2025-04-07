# =============================
# Blender Dataset to PyTorch Incremental Training Pipeline
# Includes preprocessing, model, training loop, checkpoints, and visualization
# =============================

import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---- Dataset Definition ----
class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None, is_blender=True, depth_scale=1.0):
        if (is_blender == True):
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        else:
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        self.transform = transform
        self.is_blender = is_blender
        self.depth_scale = depth_scale  # Scale factor for normalization

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.is_blender:
            # depth = Image.open(self.depth_paths[idx]).convert('L')
            depth = Image.open(self.depth_paths[idx])
            depth = np.array(depth).astype(np.float32) / 65535.0  # normalize 16-bit depth
            depth = torch.from_numpy(depth).unsqueeze(0)  # shape: [1, H, W]
        else:
            depth_img = Image.open(self.depth_paths[idx])
            if depth_img.mode == 'RGB':
                depth = self.decode_rgb_depth(depth_img)
            else:
                depth = depth_img.convert('L')
        
        if self.transform:
            image = self.transform(image)
            # Apply the same resize transformation to depth images
            depth = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)(depth)
            if (self.is_blender == False):
                depth = transforms.ToTensor()(depth).float()
            # depth = transforms.ToTensor()(depth).float()
            
            # # Normalize Blender depth to [0, 1] if using 8-bit PNG
            # if self.is_blender:
            #     depth = depth / 255.0  # <-- FIXED normalization
            # else:
            depth = depth * self.depth_scale  # e.g., 0.1 for NYUv2
            
        return image, depth

    def decode_rgb_depth(self, depth_img):
        """Decode RGB depth where depth = R + G/256 + B/(256*256)"""
        depth_np = np.array(depth_img).astype(np.float32)
        r = depth_np[:, :, 0]
        g = depth_np[:, :, 1]
        b = depth_np[:, :, 2]
        decoded = r + g / 256.0 + b / (256.0 * 256.0)
        return Image.fromarray(decoded)

# ---- Lightweight CNN ----
class TinyDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Decoder blocks with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1),  # 64 = 32 + 32 (skip)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, 3, stride=2, padding=1, output_padding=1),   # 32 = 16 + 16 (skip)
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        # Final depth prediction layer
        self.final = nn.Sequential(
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Attention module for better feature focusing
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Apply attention to bottleneck features
        att = self.attention(e3)
        e3 = e3 * att
        
        # Decoder with skip connections
        d1 = self.dec1(e3)
        d1_cat = torch.cat([d1, e2], dim=1)  # Skip connection from encoder
        
        d2 = self.dec2(d1_cat)
        d2_cat = torch.cat([d2, e1], dim=1)  # Skip connection from encoder
        
        d3 = self.dec3(d2_cat)
        
        # Final depth prediction
        depth = self.final(d3)
        
        return depth

# ---- Accuracy Metrics ----
def abs_relative_error(pred, target):
    """Calculate absolute relative error"""
    return torch.mean(torch.abs(pred - target) / (target + 1e-6)).item()

def mean_absolute_error(pred, target):
    """Calculate mean absolute error in pixels"""
    return torch.mean(torch.abs(pred - target)).item()

def compute_metrics(pred, target):
    """Compute all depth metrics"""
    metrics = {
        'abs_rel': abs_relative_error(pred, target),
        'mae': mean_absolute_error(pred, target)
    }
    return metrics

# ---- Training Function ----
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    metrics_sum = {'abs_rel': 0, 'mae': 0}
    
    for images, depths in dataloader:
        images, depths = images.to(device), depths.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, depths)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_metrics = compute_metrics(outputs, depths)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
    
    # Average metrics over all batches
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    return total_loss / len(dataloader), avg_metrics

# ---- Visualization Function ----
def visualize_predictions(model, dataloader, device, num_samples=3, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    images, depths = next(iter(dataloader))
    images, depths = images.to(device), depths.to(device)
    with torch.no_grad():
        outputs = model(images)
    
    for i in range(min(num_samples, len(images))):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axs[0, 0].imshow(images[i].permute(1, 2, 0).cpu())
        axs[0, 0].set_title('Input Image')
        
        # Ground truth depth
        depth_vis = depths[i, 0].cpu()
        im1 = axs[0, 1].imshow(depth_vis, cmap='gray')
        axs[0, 1].set_title('Ground Truth')
        fig.colorbar(im1, ax=axs[0, 1], label='Depth Value')
        
        # Predicted depth
        pred_vis = outputs[i, 0].cpu()
        im2 = axs[1, 0].imshow(pred_vis, cmap='gray')
        axs[1, 0].set_title('Predicted Depth')
        fig.colorbar(im2, ax=axs[1, 0], label='Depth Value')
        
        # Error map (absolute difference)
        error_map = torch.abs(pred_vis - depth_vis)
        im3 = axs[1, 1].imshow(error_map, cmap='hot')
        axs[1, 1].set_title('Absolute Error')
        fig.colorbar(im3, ax=axs[1, 1], label='Error Magnitude')
        
        for ax in axs.flatten():
            ax.axis('off')
        
        # Add metrics as text
        abs_rel = abs_relative_error(pred_vis.unsqueeze(0).unsqueeze(0), 
                                     depth_vis.unsqueeze(0).unsqueeze(0))
        mae = mean_absolute_error(pred_vis.unsqueeze(0).unsqueeze(0), 
                                  depth_vis.unsqueeze(0).unsqueeze(0))
        
        plt.figtext(0.5, 0.01, f'Abs Rel Error: {abs_rel:.4f} | Mean Abs Error: {mae:.4f}', 
                   ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
        plt.close()

# ---- Training Script ----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def run_stage(image_dir, depth_dir, model, stage_name, lr=1e-3, batch_size=8, epochs=5, ckpt_path=None, is_blender=True):
    print(f"\n--- Training {stage_name} ---")
    
    # Set appropriate depth scale based on dataset type
    depth_scale = 1.0  # Default scale for Blender
    if not is_blender:
        depth_scale = 0.1  # Adjust for NYUv2/real-world datasets
    
    full_dataset = DepthDataset(image_dir, depth_dir, transform=transform, 
                               is_blender=is_blender, depth_scale=depth_scale)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded checkpoint from {ckpt_path}")

    for epoch in range(epochs):
        loss, metrics = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, AbsRelError={metrics['abs_rel']:.4f}, MAE={metrics['mae']:.4f}")

    torch.save(model.state_dict(), f"depthnet_{stage_name}.pth")
    print(f"Saved checkpoint: depthnet_{stage_name}.pth")

    # Evaluate final model on test set
    print("\n--- Evaluation on Test Set ---")
    model.eval()
    test_metrics_sum = {'abs_rel': 0, 'mae': 0}
    with torch.no_grad():
        for images, depths in test_loader:
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            batch_metrics = compute_metrics(outputs, depths)
            for k in test_metrics_sum:
                test_metrics_sum[k] += batch_metrics[k]
    
    # Average metrics over all test batches
    avg_test_metrics = {k: v / len(test_loader) for k, v in test_metrics_sum.items()}
    print(f"Test Metrics:")
    print(f"  Abs Relative Error: {avg_test_metrics['abs_rel']:.4f}")
    print(f"  Mean Absolute Error: {avg_test_metrics['mae']:.4f} (pixel units)")

    # Visualize predictions
    visualize_predictions(model, test_loader, device, save_dir=f"visualizations_{stage_name}")

# ---- Incremental Training ----
model = TinyDepthNet()

# Stage 1: Single object (Blender)
run_stage("E:/Caterpillar_Latest_Solution/depth_dataset/stage1/images", "E:/Caterpillar_Latest_Solution/depth_dataset/stage1/depths", model, "stage1", is_blender=True)

# Stage 2: Two objects (Blender)
run_stage("E:/Caterpillar_Latest_Solution/depth_dataset/stage2/images", "E:/Caterpillar_Latest_Solution/depth_dataset/stage2/depths", model, "stage2", ckpt_path="depthnet_stage1.pth", is_blender=True)

# Stage 3: Multiple objects + backgrounds (Blender)
run_stage("E:/Caterpillar_Latest_Solution/depth_dataset/stage3/images", "E:/Caterpillar_Latest_Solution/depth_dataset/stage3/depths", model, "stage3", ckpt_path="depthnet_stage2.pth", is_blender=True)

# Stage 4: NYUv2 / DIODE dataset (Real-world RGB depth maps)
run_stage("E:/Caterpillar_Latest_Solution/depth_dataset/stage4_realworld/images", "E:/Caterpillar_Latest_Solution/depth_dataset/stage4_realworld/depths", model, "stage4_realworld", ckpt_path="depthnet_stage3.pth", lr=1e-4, epochs=10, is_blender=False)