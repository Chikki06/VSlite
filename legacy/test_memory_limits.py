#!/usr/bin/env python3
"""
Memory limit testing script
This helps determine the maximum batch size and image size your GPU can handle
"""

import torch
import torch.nn as nn
import numpy as np

def test_memory_limit():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = torch.device('cuda:0')
    print(f"Testing on {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test different configurations
    configs = [
        {"batch_size": 1, "img_size": 256, "channels": 64},
        {"batch_size": 1, "img_size": 384, "channels": 64},
        {"batch_size": 1, "img_size": 512, "channels": 64},
        {"batch_size": 2, "img_size": 256, "channels": 64},
        {"batch_size": 2, "img_size": 384, "channels": 64},
        {"batch_size": 4, "img_size": 256, "channels": 64},
        {"batch_size": 8, "img_size": 256, "channels": 64},
    ]
    
    for config in configs:
        try:
            torch.cuda.empty_cache()
            
            # Create test tensors similar to your model
            batch_size = config["batch_size"]
            img_size = config["img_size"]
            channels = config["channels"]
            
            # Input tensors
            x = torch.randn(batch_size, 3, img_size, img_size, device=device)
            y = torch.randn(batch_size, 3, img_size, img_size, device=device)
            
            # Feature tensors (similar to generator features)
            features = torch.randn(batch_size, channels, img_size//4, img_size//4, device=device)
            
            # Discriminator features (fix dimension mismatch)
            d_features = torch.randn(batch_size, channels, img_size//8, img_size//8, device=device)
            
            # Test forward pass simulation
            loss = torch.sum(features * d_features.mean(dim=[2,3], keepdim=True))
            loss.backward()
            
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"✓ Config {config}: {memory_used:.2f}GB")
            
        except torch.cuda.OutOfMemoryError:
            print(f"✗ Config {config}: OUT OF MEMORY")
        except Exception as e:
            print(f"✗ Config {config}: Error - {e}")
        finally:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_memory_limit()
