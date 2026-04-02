import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict
from torchvision.models import resnet50, ResNet50_Weights

# Disable pyvips import
class MockPyvips:
    pass

sys.modules['pyvips'] = MockPyvips()

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import model architecture
from main import AttGenerator_ResNet50_SkipCon

# Remove the mock after importing
del sys.modules['pyvips']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate FLOPs for hyperspectral image inference')
    parser.add_argument('--h', type=int, default=None,
                       help='Height of the sample (e.g., 35000)')
    parser.add_argument('--w', type=int, default=None,
                       help='Width of the sample (e.g., 35000)')
    parser.add_argument('--model', type=int, default=0, choices=[0, 1],
                       help='Model type: 0 for 10-channel (default), 1 for 13-channel with VIS')
    parser.add_argument('--patch_size', type=int, default=1024,
                       help='Patch size for processing (default: 1024)')
    parser.add_argument('--scaling_factor', type=float, default=0.69/2,
                       help='IR to VIS scaling factor (default: 0.345)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model filename for reference (not loaded, just for display)')
    return parser.parse_args()


def count_flops_with_fvcore(model, input_shape):
    """Count FLOPs using fvcore library"""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        
        # Print detailed table
        print("\n=== Detailed FLOPs Analysis (fvcore) ===")
        print(flop_count_table(flops, max_depth=3))
        
        return total_flops
        
    except ImportError:
        print("Warning: fvcore not installed. Install with: pip install fvcore")
        return None


def count_flops_with_thop(model, input_shape):
    """Count FLOPs using thop library"""
    try:
        from thop import profile, clever_format
        
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # MACs to FLOPs (multiply-accumulate = 2 operations)
        flops = macs * 2
        
        macs_str, params_str = clever_format([macs, params], "%.3f")
        flops_str, _ = clever_format([flops, 0], "%.3f")
        
        print("\n=== Model Statistics (thop) ===")
        print(f"Parameters: {params_str}")
        print(f"MACs: {macs_str}")
        print(f"FLOPs: {flops_str}")
        
        return flops
        
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return None


def count_flops_with_ptflops(model, input_shape):
    """Count FLOPs using ptflops library"""
    try:
        from ptflops import get_model_complexity_info
        
        # ptflops expects input as (C, H, W) without batch dimension
        input_res = input_shape[2:]  # (H, W)
        in_channels = input_shape[1]
        
        macs, params = get_model_complexity_info(
            model, 
            (in_channels, *input_res),
            as_strings=False,
            print_per_layer_stat=True,
            verbose=True
        )
        
        # MACs to FLOPs
        flops = macs * 2
        
        print("\n=== Model Statistics (ptflops) ===")
        print(f"Parameters: {params:,}")
        print(f"MACs: {macs:,}")
        print(f"FLOPs: {flops:,}")
        
        return flops
        
    except ImportError:
        print("Warning: ptflops not installed. Install with: pip install ptflops")
        return None


def calculate_total_flops_for_sample(model, height, width, model_type, patch_size=1024, scaling_factor=0.69/2):
    """Calculate total FLOPs for processing an entire sample with patches"""
    
    # For model_type 0 (IR only), dimensions are scaled
    # For model_type 1 (IR+VIS), height and width are VIS dimensions
    if model_type == 0:
        # height and width are VIS-equivalent dimensions
        # IR dimensions are smaller
        ir_h = int(height * scaling_factor)
        ir_w = int(width * scaling_factor)
        r, c = height, width
    else:
        # height and width are already VIS dimensions
        r, c = height, width
    
    # Calculate number of patches
    rn = int(np.ceil(r / patch_size))
    cn = int(np.ceil(c / patch_size))
    total_patches = rn * cn
    
    print(f"\n=== Sample Processing Analysis ===")
    print(f"Model Type: {model_type} ({'IR+VIS 13-channel' if model_type == 1 else 'IR-only 10-channel'})")
    print(f"Output dimensions (VIS equivalent): {r} x {c}")
    if model_type == 0:
        print(f"IR dimensions: {ir_h} x {ir_w}")
    print(f"Patch size: {patch_size}")
    print(f"Number of patches (rows x cols): {rn} x {cn} = {total_patches}")
    
    # Calculate FLOPs for one patch
    channels = 13 if model_type == 1 else 10
    single_patch_shape = (1, channels, patch_size, patch_size)
    
    print(f"\n=== Calculating FLOPs per Patch ===")
    print(f"Patch input shape: {single_patch_shape}")
    
    # Try different libraries in order of preference
    patch_flops = None
    
    # Try fvcore first (most detailed)
    if patch_flops is None:
        patch_flops = count_flops_with_fvcore(model, single_patch_shape)
    
    # Try thop second
    if patch_flops is None:
        patch_flops = count_flops_with_thop(model, single_patch_shape)
    
    # Try ptflops third
    if patch_flops is None:
        patch_flops = count_flops_with_ptflops(model, single_patch_shape)
    
    if patch_flops is None:
        print("\nERROR: No FLOPs counting library available!")
        print("Please install one of: fvcore, thop, or ptflops")
        print("  pip install fvcore")
        print("  pip install thop")
        print("  pip install ptflops")
        return None, None, None
    
    print(f"\n=== Per-Patch FLOPs ===")
    print(f"FLOPs per patch: {patch_flops:,}")
    print(f"FLOPs per patch (GFLOPs): {patch_flops / 1e9:.3f}")
    
    # Calculate total FLOPs for entire sample
    total_sample_flops = patch_flops * total_patches
    
    print(f"\n=== Total Sample FLOPs ===")
    print(f"Total patches: {total_patches}")
    print(f"Total FLOPs: {total_sample_flops:,}")
    print(f"Total FLOPs (GFLOPs): {total_sample_flops / 1e9:.3f}")
    print(f"Total FLOPs (TFLOPs): {total_sample_flops / 1e12:.3f}")
    
    # Calculate average FLOPs per output pixel
    total_output_pixels = r * c
    flops_per_pixel = total_sample_flops / total_output_pixels
    
    print(f"\n=== Per-Pixel Statistics ===")
    print(f"Total output pixels: {total_output_pixels:,}")
    print(f"FLOPs per output pixel: {flops_per_pixel:.2f}")
    
    return total_sample_flops, patch_flops, total_patches


def main():
    args = parse_arguments()
    
    # Validate inputs
    if args.h is None or args.w is None:
        print("ERROR: Both --h and --w must be specified")
        print("Example: python flops_calculator.py --h=35000 --w=35000 --model=1")
        return
    
    print(f"=== FLOPs Calculation ===")
    if args.model_name:
        print(f"Model reference: {args.model_name}")
    print(f"Model type: {args.model} ({'13-channel with VIS' if args.model == 1 else '10-channel IR only'})")
    print(f"Sample dimensions: {args.h} x {args.w}")
    print(f"Patch size: {args.patch_size}")
    print(f"Scaling factor: {args.scaling_factor}")
    
    # Initialize model
    print(f"\nInitializing model architecture...")
    model = AttGenerator_ResNet50_SkipCon(input_channels=13 if args.model == 1 else 10)
    
    # Move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    # Calculate FLOPs
    with torch.no_grad():
        total_flops, patch_flops, num_patches = calculate_total_flops_for_sample(
            model, args.h, args.w, args.model, args.patch_size, args.scaling_factor
        )
    
    if total_flops is None:
        return
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Sample: {args.h} x {args.w}")
    print(f"Model type: {args.model}")
    print(f"Total FLOPs: {total_flops:,} ({total_flops/1e12:.3f} TFLOPs)")
    print(f"FLOPs per patch: {patch_flops:,} ({patch_flops/1e9:.3f} GFLOPs)")
    print(f"Number of patches: {num_patches}")
    
    # Additional context
    print(f"\n=== Performance Context ===")
    print(f"On a GPU with 100 TFLOPs/s compute:")
    print(f"  Theoretical minimum time: {total_flops / (100e12):.2f} seconds")
    print(f"On a GPU with 50 TFLOPs/s compute:")
    print(f"  Theoretical minimum time: {total_flops / (50e12):.2f} seconds")
    print(f"On a GPU with 20 TFLOPs/s compute:")
    print(f"  Theoretical minimum time: {total_flops / (20e12):.2f} seconds")


if __name__ == '__main__':
    main()