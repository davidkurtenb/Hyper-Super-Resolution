import sys
import os

sys.path.append('/homes/dkurtenb/projects/hypersuperresolution/prod_code')

from SRCNN_utils import *

def inference_main(model_path, image_dir, output_dir, scale_factor=2, use_cuda=True):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(model_path, device)
    
    results = batch_evaluate(model, image_dir, scale_factor, device, output_dir)
    
    print(f"Results saved to: {output_dir}")
    for key, value in results.items():
        if key != "images":  
            print(f"  {key}: {value}")
            
    return results

if __name__ == "__main__":
    inference_main(
        model_path= '/homes/dkurtenb/projects/hypersuperresolution/outputs/checkpoints/model_best.pth',
        image_dir= '/homes/dkurtenb/projects/hypersuperresolution/data/beemachine/images/small_test',
        output_dir= '/homes/dkurtenb/projects/hypersuperresolution/outputs/results',
        scale_factor=2,
        use_cuda=True
    )