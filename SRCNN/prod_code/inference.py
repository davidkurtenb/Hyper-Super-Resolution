import torch
import sys
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\HyperSuperResolution\prod_code')
from utils import SRCNN, calculate_psnr #, preprocess_image, postprocess_image

def load_model(model_path, device):

    model = SRCNN().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model

def enhance_image(model, image_path, scale_factor=2, device='cpu', output_dir=r"C:\Users\dk412\Desktop\David\Python Projects\HyperSuperResolution\outputs\results"):

    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(image_path).convert('RGB')
    
    w, h = img.size
    low_res = img.resize((w//scale_factor, h//scale_factor), Image.BICUBIC)
    bicubic_upscaled = low_res.resize((w, h), Image.BICUBIC)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(bicubic_upscaled).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    output_img = output.squeeze().cpu()
    output_img = output_img * std + mean
    output_img = torch.clamp(output_img, 0, 1)
    output_img = transforms.ToPILImage()(output_img)
    
    original_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(original_filename)
    
    original_output_path = os.path.join(output_dir, f"{name}_original{ext}")
    lowres_output_path = os.path.join(output_dir, f"{name}_lowres{ext}")
    bicubic_output_path = os.path.join(output_dir, f"{name}_bicubic{ext}")
    sr_output_path = os.path.join(output_dir, f"{name}_superres{ext}")
    
    img.save(original_output_path)
    low_res.save(lowres_output_path)
    bicubic_upscaled.save(bicubic_output_path)
    output_img.save(sr_output_path)
    
 
    img_np = np.array(img)
    bicubic_np = np.array(bicubic_upscaled)
    sr_np = np.array(output_img)
    
    # Calculate PSNR
    bicubic_psnr = calculate_psnr(img_np, bicubic_np)
    sr_psnr = calculate_psnr(img_np, sr_np)
    
    print(f"PSNR for Bicubic upscaling: {bicubic_psnr:.2f} dB")
    print(f"PSNR for SRCNN: {sr_psnr:.2f} dB")
    print(f"PSNR improvement: {sr_psnr - bicubic_psnr:.2f} dB")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
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
    
    comparison_path = os.path.join(output_dir, f"{name}_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300)
    
    return {
        "original": original_output_path,
        "lowres": lowres_output_path,
        "bicubic": bicubic_output_path,
        "superres": sr_output_path,
        "comparison": comparison_path,
        "bicubic_psnr": bicubic_psnr,
        "sr_psnr": sr_psnr,
        "psnr_improvement": sr_psnr - bicubic_psnr
    }

def batch_evaluate(model, image_dir, scale_factor=2, device='cpu', output_dir="results"):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    bicubic_psnrs = []
    sr_psnrs = []
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        result = enhance_image(model, image_path, scale_factor, device, output_dir)
        
        bicubic_psnrs.append(result["bicubic_psnr"])
        sr_psnrs.append(result["sr_psnr"])
    
    avg_bicubic_psnr = sum(bicubic_psnrs) / len(bicubic_psnrs)
    avg_sr_psnr = sum(sr_psnrs) / len(sr_psnrs)
    avg_improvement = avg_sr_psnr - avg_bicubic_psnr
    
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
            f.write(f"  Improvement: {sr_psnrs[i] - bicubic_psnrs[i]:.2f} dB\n\n")
    
    return {
        "avg_bicubic_psnr": avg_bicubic_psnr,
        "avg_sr_psnr": avg_sr_psnr,
        "avg_improvement": avg_improvement
    }