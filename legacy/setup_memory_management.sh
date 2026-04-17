#!/bin/bash

# Memory management configuration script for VSlite
# This script sets up optimal memory settings for PyTorch training

echo "Setting up memory management configuration for VSlite..."

# 1. PyTorch CUDA memory configuration
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
echo "✓ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# 2. Additional memory optimization environment variables
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6"  # Optimize for common architectures
export CUDA_LAUNCH_BLOCKING=0  # Async execution for better performance
export CUBLAS_WORKSPACE_CONFIG=":4096:8"  # Deterministic operations with memory efficiency

# 3. Python memory optimization
export MALLOC_TRIM_THRESHOLD_=0  # Reduce memory fragmentation
export MALLOC_MMAP_THRESHOLD_=131072

echo "✓ Set additional memory optimization variables"

# 4. Create a memory monitoring script
cat > monitor_memory.py << 'EOF'
#!/usr/bin/env python3
"""
Memory monitoring script for VSlite training
Run this in a separate terminal to monitor GPU memory usage
"""

import time
import subprocess
import re

def get_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                               '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for i, line in enumerate(lines):
            used, total = map(int, line.split(', '))
            gpu_info.append((i, used, total))
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return []

def main():
    print("GPU Memory Monitor - Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while True:
            gpu_info = get_gpu_memory()
            if gpu_info:
                for gpu_id, used, total in gpu_info:
                    usage_percent = (used / total) * 100
                    print(f"GPU {gpu_id}: {used:5d}MB / {total:5d}MB ({usage_percent:5.1f}%)")
                print("-" * 50)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
EOF

chmod +x monitor_memory.py
echo "✓ Created memory monitoring script (monitor_memory.py)"

# 5. Create a memory test script
cat > test_memory_limits.py << 'EOF'
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
            
            # Discriminator features
            d_features = torch.randn(batch_size, channels*2, img_size//8, img_size//8, device=device)
            
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
EOF

chmod +x test_memory_limits.py
echo "✓ Created memory limit testing script (test_memory_limits.py)"

# 6. Create quick start training commands
cat > quick_start_training.sh << 'EOF'
#!/bin/bash

# Quick start training commands with different memory profiles

echo "VSlite Memory-Optimized Training Quick Start"
echo "============================================="
echo ""
echo "1. Test memory limits first:"
echo "   python test_memory_limits.py"
echo ""
echo "2. Monitor GPU memory (in separate terminal):"
echo "   python monitor_memory.py"
echo ""
echo "3. Start training with appropriate profile:"
echo ""
echo "   Light profile (lowest memory):"
echo "   ./train_memory_profiles.sh light"
echo ""
echo "   Medium profile (balanced):"
echo "   ./train_memory_profiles.sh medium"
echo ""
echo "   Heavy profile (high quality):"
echo "   ./train_memory_profiles.sh heavy"
echo ""
echo "4. If you still get OOM errors, try:"
echo "   - Use light profile"
echo "   - Reduce batch_size to 1"
echo "   - Reduce crop_size to 256"
echo "   - Use single GPU (--gpu_ids 0)"
echo ""
EOF

chmod +x quick_start_training.sh
echo "✓ Created quick start guide (quick_start_training.sh)"

# 7. Add persistent environment variables to bashrc
echo "" >> ~/.bashrc
echo "# VSlite Memory Optimization Settings" >> ~/.bashrc
echo "export PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:True\"" >> ~/.bashrc
echo "export CUDA_LAUNCH_BLOCKING=0" >> ~/.bashrc
echo "export MALLOC_TRIM_THRESHOLD_=0" >> ~/.bashrc

echo "✓ Added persistent environment variables to ~/.bashrc"

echo ""
echo "Memory management setup complete!"
echo ""
echo "Next steps:"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Test memory limits: python test_memory_limits.py"
echo "3. Start training: ./train_memory_profiles.sh light"
echo ""
echo "Files created:"
echo "- train_memory_optimized.py (memory-efficient training script)"
echo "- train_memory_profiles.sh (training with different memory profiles)"
echo "- trainers/memory_efficient_trainer.py (gradient accumulation trainer)"
echo "- monitor_memory.py (GPU memory monitoring)"
echo "- test_memory_limits.py (memory limit testing)"
echo "- quick_start_training.sh (quick start guide)"