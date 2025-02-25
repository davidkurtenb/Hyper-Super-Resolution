# utils.py
import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import datetime
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# SRCNN Model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# Dataset Class
class Set14Dataset(Dataset):
    def __init__(self, root_dir, scale_factor=2, transform=None, image_size=(256, 256)):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale_factor (int): Factor to downscale and upscale images.
            transform (callable, optional): Optional transform to be applied to images.
            image_size (tuple): Size to resize all images to (width, height).
        """
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.transform = transform
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Resize all images to the same size
        image = image.resize(self.image_size, Image.BICUBIC)
        
        # Create low-resolution image
        low_res = image.resize((self.image_size[0]//self.scale_factor, 
                              self.image_size[1]//self.scale_factor), 
                              Image.BICUBIC)
        low_res = low_res.resize(self.image_size, Image.BICUBIC)
        
        if self.transform:
            high_res = self.transform(image)
            low_res = self.transform(low_res)
            
        return low_res, high_res

# Utility Functions
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def save_checkpoint(state, is_best, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_path)
        
    metadata = {
        'epoch': state['epoch'],
        'val_loss': state['val_loss'],
        'timestamp': str(datetime.datetime.now())
    }
    metadata_path = os.path.join(save_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def train_model(model, train_loader, val_loader, num_epochs, device, save_dir=r'C:\Users\dk412\Desktop\David\Python Projects\HyperSuperResolution\outputs\checkpoints'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for low_res, high_res in pbar:
                low_res, high_res = low_res.to(device), high_res.to(device)
                
                optimizer.zero_grad()
                output = model(low_res)
                loss = criterion(output, high_res)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'train_loss': train_loss/len(train_loader)})
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                output = model(low_res)
                val_loss += criterion(output, high_res).item()
        
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        save_checkpoint(checkpoint, val_loss < best_val_loss, save_dir)
        best_val_loss = min(val_loss, best_val_loss)

"""
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))
"""

