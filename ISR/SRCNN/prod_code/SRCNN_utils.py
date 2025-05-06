import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import argparse
import sys

###########################################################
#                 TRAINING
###########################################################

class SRDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_factor=2, patch_size=96, augment=True):

        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        w, h = image.size
        if w > self.patch_size and h > self.patch_size:
            left = np.random.randint(0, w - self.patch_size)
            top = np.random.randint(0, h - self.patch_size)
            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
        else:
            image = image.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            
        #Data augmentation
        if self.augment:
            #rotation
            if np.random.rand() > 0.5:
                angle = np.random.choice([90, 180, 270])
                image = image.rotate(angle)
                
            #horizontal flip
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
            #vertical flip
            if np.random.rand() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        hr_image = image
        
        lr_size = (hr_image.width // self.scale_factor, hr_image.height // self.scale_factor)
        lr_image = hr_image.resize(lr_size, Image.BICUBIC)
        lr_upscaled = lr_image.resize((hr_image.width, hr_image.height), Image.BICUBIC)
        
        to_tensor = transforms.ToTensor()
        hr_tensor = to_tensor(hr_image)
        lr_tensor = to_tensor(lr_upscaled)
        
        return lr_tensor, hr_tensor

#SRCNN model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)  
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def save_checkpoint(state, is_best, save_dir, scale_factor):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'checkpoint_latest_sf{scale_factor}.pth')
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(save_dir, f'model_best_sf{scale_factor}.pth')
        torch.save(state, best_path)
        print(f"Saved best model to {best_path}")
        
    metadata = {
        'epoch': state['epoch'],
        'val_loss': state['val_loss'],
        'val_psnr': state.get('val_psnr', 0),
        'scale_factor': scale_factor,
        'timestamp': str(datetime.datetime.now())
    }
    metadata_path = os.path.join(save_dir, f'training_metadata_sf{scale_factor}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def plot_and_save_metrics(train_losses, val_losses, train_psnrs, val_psnrs, save_path):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('SRCNN Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_psnrs, 'g-', label='Training PSNR')
    plt.plot(epochs, val_psnrs, 'm-', label='Validation PSNR')  # 'm' for magenta
    plt.title('SRCNN Training and Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {save_path}")

def train_model(model, train_loader, val_loader, num_epochs, device, save_dir, plot_dir, scale_factor, lr=0.0001):
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    time_log_path = os.path.join(save_dir, f'training_time_log_sf{scale_factor}.txt')
    with open(time_log_path, 'w') as f:
        f.write("Epoch,Elapsed_Time,Epoch_Duration\n")
    
    training_start_time = datetime.datetime.now()
    epoch_start_time = training_start_time

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    train_psnrs = []
    val_psnrs = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_psnr = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for low_res, high_res in pbar:
                low_res, high_res = low_res.to(device), high_res.to(device)
                
                optimizer.zero_grad()
                output = model(low_res)
                loss = criterion(output, high_res)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                with torch.no_grad():
                    batch_psnr = 0
                    for i in range(output.size(0)):
                        img1 = output[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                        img2 = high_res[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                        batch_psnr += calculate_psnr(img1, img2)
                    batch_psnr /= output.size(0)
                    train_psnr += batch_psnr
                
                train_loss += loss.item()
                pbar.set_postfix({'train_loss': train_loss/len(train_loader), 'psnr': batch_psnr})
        
        epoch_train_loss = train_loss/len(train_loader)
        epoch_train_psnr = train_psnr/len(train_loader)
        train_losses.append(epoch_train_loss)
        train_psnrs.append(epoch_train_psnr)

        model.eval()
        val_loss = 0
        val_psnr = 0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                output = model(low_res)
                loss = criterion(output, high_res)
                val_loss += loss.item()
                
                for i in range(output.size(0)):
                    img1 = output[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    img2 = high_res[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    val_psnr += calculate_psnr(img1, img2)
        
        epoch_val_loss = val_loss/len(val_loader)
        epoch_val_psnr = val_psnr/len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_psnrs.append(epoch_val_psnr)

        scheduler.step(epoch_val_loss)

        current_time = datetime.datetime.now()
        time_since_start = current_time - training_start_time
        epoch_duration = current_time - epoch_start_time

        with open(time_log_path, 'a') as f:
            f.write(f"{epoch+1},{time_since_start},{epoch_duration}\n")

        epoch_start_time = current_time
        
        # Print time information
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Train PSNR: {epoch_train_psnr:.2f} dB, Val PSNR: {epoch_val_psnr:.2f} dB')
        print(f'Time elapsed: {time_since_start}, Epoch duration: {epoch_duration}')

        #print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        #print(f'Train PSNR: {epoch_train_psnr:.2f} dB, Val PSNR: {epoch_val_psnr:.2f} dB')
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_psnr': epoch_train_psnr,
            'val_psnr': epoch_val_psnr,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_psnrs': train_psnrs,
            'val_psnrs': val_psnrs,
            'scale_factor': scale_factor
        }
        
        is_best = epoch_val_loss < best_val_loss
        save_checkpoint(checkpoint, is_best, save_dir, scale_factor)
        
        if is_best:
            best_val_loss = epoch_val_loss
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            plot_and_save_metrics(train_losses, val_losses, train_psnrs, val_psnrs, 
                                 os.path.join(plot_dir, f'metrics_epoch_{epoch+1}_sf{scale_factor}.png'))
    
    total_training_time = datetime.datetime.now() - training_start_time
    with open(time_log_path, 'a') as f:
        f.write(f"Total,{total_training_time},--\n")
    
    print(f"Total training time: {total_training_time}")

    plot_and_save_metrics(train_losses, val_losses, train_psnrs, val_psnrs, 
                         os.path.join(plot_dir, f'metrics_final_sf{scale_factor}.png'))
    
    metrics_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_psnrs': train_psnrs,
        'val_psnrs': val_psnrs,
        'epochs': list(range(1, num_epochs + 1))
    }
    
    with open(os.path.join(plot_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    return train_losses, val_losses, train_psnrs, val_psnrs


###########################################################
#                 INFERFENCE
###########################################################

def load_model(model_path, device):
    model = SRCNN().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    if 'val_psnr' in checkpoint:
        print(f"Validation PSNR: {checkpoint['val_psnr']:.2f} dB")
    
    return model

def enhance_image(model, image_path, scale_factor=2, device='cpu', output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(output_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    
    img = Image.open(image_path).convert('RGB')
    
    w, h = img.size
    
    low_res = img.resize((w//scale_factor, h//scale_factor), Image.BICUBIC)
    
    bicubic_upscaled = low_res.resize((w, h), Image.BICUBIC)
    
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(bicubic_upscaled).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    output_img = output.squeeze().cpu().clamp(0, 1)
    output_img = transforms.ToPILImage()(output_img)
    
    original_output_path = os.path.join(save_dir, f"{filename}_original.png")
    lowres_output_path = os.path.join(save_dir, f"{filename}_lowres.png")
    bicubic_output_path = os.path.join(save_dir, f"{filename}_bicubic.png")
    sr_output_path = os.path.join(save_dir, f"{filename}_superres.png")
    
    img.save(original_output_path)
    low_res.save(lowres_output_path)
    bicubic_upscaled.save(bicubic_output_path)
    output_img.save(sr_output_path)
    
    #PSNR
    img_np = np.array(img)
    bicubic_np = np.array(bicubic_upscaled)
    sr_np = np.array(output_img)
    
    bicubic_psnr = calculate_psnr(img_np, bicubic_np)
    sr_psnr = calculate_psnr(img_np, sr_np)
    
    print(f"PSNR for Bicubic upscaling: {bicubic_psnr:.2f} dB")
    print(f"PSNR for SRCNN: {sr_psnr:.2f} dB")
    print(f"PSNR improvement: {sr_psnr - bicubic_psnr:.2f} dB")
    
    #visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flat
    
    axes[0].imshow(np.array(img))
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(np.array(low_res))
    axes[1].set_title(f"Low-res ({w//scale_factor}x{h//scale_factor})")
    axes[1].axis("off")
    
    axes[2].imshow(np.array(bicubic_upscaled))
    axes[2].set_title(f"Bicubic Upscaling\nPSNR: {bicubic_psnr:.2f} dB")
    axes[2].axis("off")
    
    axes[3].imshow(np.array(output_img))
    axes[3].set_title(f"SRCNN Super-Resolution\nPSNR: {sr_psnr:.2f} dB")
    axes[3].axis("off")
    
    zoomed_fig = plt.figure(figsize=(18, 8))
    
    crop_w, crop_h = w//4, h//4
    crop_x, crop_y = w//3, h//3 
    
    #crops
    original_crop = np.array(img)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    bicubic_crop = np.array(bicubic_upscaled)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    sr_crop = np.array(output_img)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    ax1 = zoomed_fig.add_subplot(131)
    ax1.imshow(original_crop)
    ax1.set_title("Original (zoomed)")
    ax1.axis("off")
    
    ax2 = zoomed_fig.add_subplot(132)
    ax2.imshow(bicubic_crop)
    ax2.set_title(f"Bicubic (zoomed)\nPSNR: {bicubic_psnr:.2f} dB")
    ax2.axis("off")
    
    ax3 = zoomed_fig.add_subplot(133)
    ax3.imshow(sr_crop)
    ax3.set_title(f"SRCNN (zoomed)\nPSNR: {sr_psnr:.2f} dB")
    ax3.axis("off")
    
    comparison_path = os.path.join(save_dir, f"{filename}_comparison.png")
    zoomed_comparison_path = os.path.join(save_dir, f"{filename}_zoomed_comparison.png")
    
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300)
    
    plt.figure(zoomed_fig.number)
    plt.tight_layout()
    plt.savefig(zoomed_comparison_path, dpi=300)
    
    plt.close('all')
    
    return {
        "original": original_output_path,
        "lowres": lowres_output_path,
        "bicubic": bicubic_output_path,
        "superres": sr_output_path,
        "comparison": comparison_path,
        "zoomed_comparison": zoomed_comparison_path,
        "bicubic_psnr": bicubic_psnr,
        "sr_psnr": sr_psnr,
        "psnr_improvement": sr_psnr - bicubic_psnr
    }

def batch_evaluate(model, image_dir, scale_factor=2, device='cpu', output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    bicubic_psnrs = []
    sr_psnrs = []
    improvements = []
    image_names = []
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        result = enhance_image(model, image_path, scale_factor, device, output_dir)
        
        bicubic_psnrs.append(result["bicubic_psnr"])
        sr_psnrs.append(result["sr_psnr"])
        improvements.append(result["psnr_improvement"])
        image_names.append(image_file)
    
    avg_bicubic_psnr = sum(bicubic_psnrs) / len(bicubic_psnrs)
    avg_sr_psnr = sum(sr_psnrs) / len(sr_psnrs)
    avg_improvement = sum(improvements) / len(improvements)
    
    print("\n===== EVALUATION RESULTS =====")
    print(f"Number of test images: {len(image_files)}")
    print(f"Average Bicubic PSNR: {avg_bicubic_psnr:.2f} dB")
    print(f"Average SRCNN PSNR: {avg_sr_psnr:.2f} dB")
    print(f"Average PSNR improvement: {avg_improvement:.2f} dB")
    
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("===== EVALUATION RESULTS =====\n")
        f.write(f"Number of test images: {len(image_files)}\n")
        f.write(f"Average Bicubic PSNR: {avg_bicubic_psnr:.2f} dB\n")
        f.write(f"Average SRCNN PSNR: {avg_sr_psnr:.2f} dB\n")
        f.write(f"Average PSNR improvement: {avg_improvement:.2f} dB\n\n")
        
        f.write("Individual Image Results:\n")
        for i, image_file in enumerate(image_files):
            f.write(f"{image_file}:\n")
            f.write(f"  Bicubic PSNR: {bicubic_psnrs[i]:.2f} dB\n")
            f.write(f"  SRCNN PSNR: {sr_psnrs[i]:.2f} dB\n")
            f.write(f"  Improvement: {improvements[i]:.2f} dB\n\n")
    
    plt.figure(figsize=(12, 8))
    
    sorted_indices = np.argsort(improvements)[::-1]  
    sorted_names = [image_names[i] for i in sorted_indices]
    sorted_bicubic = [bicubic_psnrs[i] for i in sorted_indices]
    sorted_sr = [sr_psnrs[i] for i in sorted_indices]
    sorted_improvements = [improvements[i] for i in sorted_indices]
    
    x = np.arange(len(sorted_names))
    width = 0.35
    
    plt.bar(x - width/2, sorted_bicubic, width, label='Bicubic PSNR')
    plt.bar(x + width/2, sorted_sr, width, label='SRCNN PSNR')
    
    plt.xlabel('Images')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Comparison: Bicubic vs SRCNN')
    plt.xticks(x, [name[:10] for name in sorted_names], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psnr_comparison.png"), dpi=300)
    
    plt.figure(figsize=(12, 6))
    colors = ['g' if imp > 0 else 'r' for imp in sorted_improvements]
    plt.bar(x, sorted_improvements, color=colors)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Images')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('SRCNN PSNR Improvement over Bicubic Upscaling')
    plt.xticks(x, [name[:10] for name in sorted_names], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psnr_improvement.png"), dpi=300)
    
    plt.close('all')
    
    results_dict = {
        "avg_bicubic_psnr": avg_bicubic_psnr,
        "avg_sr_psnr": avg_sr_psnr,
        "avg_improvement": avg_improvement,
        "images": {
            name: {
                "bicubic_psnr": bicubic,
                "sr_psnr": sr,
                "improvement": imp
            } for name, bicubic, sr, imp in zip(image_names, bicubic_psnrs, sr_psnrs, improvements)
        }
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    return results_dict