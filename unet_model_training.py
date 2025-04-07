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
import torch.nn.functional as F

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
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128]):
        super().__init__()

        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in features:
            self.downs.append(self.conv_block(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1]*2)

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.dec_blocks.append(self.conv_block(feature*2, feature))

        # Output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for up, dec, skip in zip(self.ups, self.dec_blocks, skip_connections):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])  # handle any mismatch
            x = torch.cat((skip, x), dim=1)
            x = dec(x)

        return self.final_conv(x)

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
        # outputs = torch.clamp(outputs, min=1e-3, max=10.0)
        # depths = torch.clamp(depths, min=1e-3, max=10.0)
        # loss = scale_invariant_loss(outputs, depths)
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
model = UNetSmall()

# Stage 1: Single object (Blender)
run_stage("E:/Caterpillar_Latest_Solution/depth_dataset/stage1/images", "E:/Caterpillar_Latest_Solution/depth_dataset/stage1/depths", model, "stage1", is_blender=True)

# Stage 2: Two objects (Blender)
run_stage("E:/Caterpillar_Latest_Solution/depth_dataset/stage2/images", "E:/Caterpillar_Latest_Solution/depth_dataset/stage2/depths", model, "stage2", ckpt_path="depthnet_stage1.pth", is_blender=True)

# Stage 3: Multiple objects + backgrounds (Blender)
run_stage("E:/Caterpillar_Latest_Solution/depth_dataset/stage3/images", "E:/Caterpillar_Latest_Solution/depth_dataset/stage3/depths", model, "stage3", ckpt_path="depthnet_stage2.pth", is_blender=True)

# Stage 4: NYUv2 / DIODE dataset (Real-world RGB depth maps)
run_stage("E:/Caterpillar_Latest_Solution/depth_dataset/stage4_realworld/images", "E:/Caterpillar_Latest_Solution/depth_dataset/stage4_realworld/depths", model, "stage4_realworld", ckpt_path="depthnet_stage3.pth", lr=1e-4, epochs=10, is_blender=False)