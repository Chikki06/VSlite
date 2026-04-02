# TensorRT FP8 Inference Guide

Complete guide for using native TensorRT with FP8 quantization for hyperspectral image inference.

## Overview

This implementation provides FP8 (8-bit floating point) inference using native TensorRT, completely independent of PyTorch at inference time. This offers:

- **4x memory reduction** compared to FP32
- **2x memory reduction** compared to FP16
- **Faster inference** on Hopper (H100) and Ada (RTX 40xx) GPUs
- **No PyTorch dependency** at inference time

## Requirements

### Hardware
- NVIDIA GPU with compute capability 8.9+ (Hopper H100, Ada RTX 4090/4080, etc.)
- For older GPUs, use FP16 or FP32 precision

### Software
```bash
# Core dependencies
pip install tensorrt>=9.0.0
pip install pycuda
pip install numpy
pip install onnx

# For data loading and image processing
pip install spectral-python
pip install pillow
pip install pyvips
pip install pandas
```

## Two-Step Process

### Step 1: Build TensorRT Engine (One-Time Setup)

This step converts your PyTorch model to an optimized TensorRT engine with FP8 quantization.

#### Basic Usage

```bash
# Build FP8 engine for IR-only model (10 channels)
python src/tensorrt_fp8_builder.py --model 0 --precision fp8 --batch_size 4

# Build FP8 engine for IR+VIS model (13 channels)
python src/tensorrt_fp8_builder.py --model 1 --precision fp8 --batch_size 4

# Build FP16 engine (fallback for older GPUs)
python src/tensorrt_fp8_builder.py --model 0 --precision fp16 --batch_size 4

# Build FP32 engine (full precision)
python src/tensorrt_fp8_builder.py --model 0 --precision fp32 --batch_size 4
```

#### Advanced Options

```bash
python src/tensorrt_fp8_builder.py \
    --model 0 \
    --precision fp8 \
    --batch_size 4 \
    --calibration_samples 500 \
    --calibration_dir /mnt/VSlite/dataset/data_same_batch_final_1024 \
    --workspace_size 8192 \
    --model_name 20_net_G.pth
```

**Parameters:**

- `--model`: Model type (0 = IR-only/10-channel, 1 = IR+VIS/13-channel)
- `--precision`: Quantization precision (fp8, fp16, fp32)
- `--batch_size`: Batch size for inference (must match at inference time)
- `--calibration_samples`: Number of samples for FP8 calibration (100-1000 recommended)
- `--calibration_dir`: Directory with calibration data (.npy files)
- `--workspace_size`: TensorRT workspace memory in MB (default: 4096)
- `--model_name`: Custom PyTorch model filename (default: 20_net_G.pth or 15_net_G.pth)
- `--skip_onnx`: Skip ONNX export if .onnx file already exists

#### What This Creates

```
saved_models/
├── 20_net_G.pth                              # Original PyTorch weights
├── 20_net_G_bs4_model0.onnx                  # ONNX intermediate format
├── 20_net_G_bs4_model0_fp8.engine            # TensorRT FP8 engine
├── 20_net_G_bs4_model0_fp8_calibration.cache # Calibration cache
├── 20_net_G_bs4_model0_fp16.engine           # TensorRT FP16 engine
└── 20_net_G_bs4_model0_fp32.engine           # TensorRT FP32 engine
```

#### Build Time Expectations

- **FP32/FP16**: 2-5 minutes
- **FP8 with calibration**: 5-15 minutes (depends on calibration samples)

### Step 2: Run Inference

Once the engine is built, use it for fast inference without PyTorch.

#### Basic Usage

```bash
# Process single folder
python src/tensorrt_inference_server.py \
    --engine 20_net_G_bs4_model0_fp8.engine \
    --folder FOLDER_NAME \
    --model 0

# Process multiple folders (batch processing)
python src/tensorrt_inference_server.py \
    --engine 20_net_G_bs4_model0_fp8.engine \
    --parent_folder /path/to/parent/directory \
    --model 0

# Save as PNG instead of SZI
python src/tensorrt_inference_server.py \
    --engine 20_net_G_bs4_model0_fp8.engine \
    --folder FOLDER_NAME \
    --model 0 \
    --save_png

# Custom output filename
python src/tensorrt_inference_server.py \
    --engine 20_net_G_bs4_model0_fp8.engine \
    --folder FOLDER_NAME \
    --model 0 \
    --output_name my_custom_name
```

#### Advanced Options

```bash
python src/tensorrt_inference_server.py \
    --engine 20_net_G_bs4_model0_fp8.engine \
    --parent_folder /mnt/VSlite/files \
    --model 0 \
    --batch_size 4 \
    --save_png \
    --output_name output
```

**Parameters:**

- `--engine`: TensorRT engine filename (must be in saved_models/ directory)
- `--folder`: Single folder name to process
- `--parent_folder`: Parent directory containing multiple folders
- `--model`: Model type (must match engine: 0 = IR-only, 1 = IR+VIS)
- `--batch_size`: Batch size (must match engine build batch size)
- `--save_png`: Save output as PNG instead of SZI format
- `--output_name`: Custom output filename (without extension)

## Comparison with main.py

### Similarities (Command-Line Interface)

Both scripts have similar command-line options:

| Feature | main.py | tensorrt_inference_server.py |
|---------|---------|------------------------------|
| Single folder | `--folder NAME` | `--folder NAME` |
| Batch processing | `--parent_folder PATH` | `--parent_folder PATH` |
| Model type | `--model 0/1` | `--model 0/1` |
| PNG output | `--save_png` | `--save_png` |
| Custom name | `--output_name NAME` | `--output_name NAME` |

### Key Differences

| Aspect | main.py | tensorrt_inference_server.py |
|--------|---------|------------------------------|
| **Backend** | PyTorch + torch-tensorrt | Native TensorRT |
| **Precision** | FP16/FP32 only | FP8/FP16/FP32 |
| **Setup** | Load .pth directly | Build .engine first (one-time) |
| **Memory** | Higher | Lower (especially FP8) |
| **Speed** | Fast | Faster (especially FP8) |
| **Dependencies** | Requires PyTorch | No PyTorch at inference |
| **Flexibility** | Runtime precision change | Fixed at build time |

### Migration from main.py

**Before (main.py):**
```bash
python src/main.py --folder FOLDER_NAME --model 0 --use_tensorrt --tensorrt_precision fp16
```

**After (tensorrt_inference_server.py):**
```bash
# One-time: Build engine
python src/tensorrt_fp8_builder.py --model 0 --precision fp8 --batch_size 4

# Run inference
python src/tensorrt_inference_server.py --engine 20_net_G_bs4_model0_fp8.engine --folder FOLDER_NAME --model 0
```

## Calibration Data

FP8 quantization benefits from calibration to maintain accuracy. The calibration process:

1. **Loads samples** from your training dataset
2. **Runs inference** on these samples
3. **Collects statistics** on activation ranges
4. **Optimizes quantization** to minimize accuracy loss

### Preparing Calibration Data

Your calibration data should be in the same format as your training data:

```
/mnt/VSlite/dataset/data_same_batch_final_1024/
├── sample_001.npy  # Shape: (10, H, W) for IR-only
├── sample_002.npy  # Shape: (13, H, W) for IR+VIS
├── sample_003.npy
└── ...
```

**Requirements:**
- 100-1000 samples (more = better calibration, longer build time)
- Same preprocessing as training data
- Representative of your inference distribution

### Without Calibration Data

If you don't have calibration data:

```bash
# FP8 without calibration (may have quality degradation)
python src/tensorrt_fp8_builder.py --model 0 --precision fp8 --calibration_samples 0

# Or use FP16 instead (no calibration needed)
python src/tensorrt_fp8_builder.py --model 0 --precision fp16
```

## Performance Benchmarking

### Measuring Performance

Both scripts log performance metrics to CSV files:

```bash
# Original main.py logs to:
runtime_logs.csv
run_summary.csv

# TensorRT inference server logs to:
tensorrt_runtime_logs.csv
```

### Expected Speedups

Based on typical scenarios:

| Configuration | Relative Speed | Memory Usage |
|---------------|----------------|--------------|
| PyTorch FP32 | 1.0x (baseline) | 1.0x (baseline) |
| PyTorch FP16 | 1.5-2x | 0.5x |
| TensorRT FP16 | 2-3x | 0.5x |
| TensorRT FP8 | 3-4x | 0.25x |

*Note: Actual performance depends on GPU architecture, batch size, and model complexity.*

### Comparing Runs

```bash
# Run with original main.py
python src/main.py --folder TEST --model 0 --use_tensorrt --tensorrt_precision fp16

# Run with TensorRT FP8
python src/tensorrt_inference_server.py --engine 20_net_G_bs4_model0_fp8.engine --folder TEST --model 0

# Compare results in CSV files
```

## Quality Validation

### Comparing FP8 vs FP32 Output

To ensure FP8 maintains acceptable quality:

```bash
# Generate FP32 reference
python src/tensorrt_fp8_builder.py --model 0 --precision fp32 --batch_size 1
python src/tensorrt_inference_server.py --engine 20_net_G_bs1_model0_fp32.engine --folder TEST --model 0 --output_name test_fp32

# Generate FP8 output
python src/tensorrt_fp8_builder.py --model 0 --precision fp8 --batch_size 1
python src/tensorrt_inference_server.py --engine 20_net_G_bs1_model0_fp8.engine --folder TEST --model 0 --output_name test_fp8

# Visually compare or use metrics:
# - PSNR (Peak Signal-to-Noise Ratio)
# - SSIM (Structural Similarity Index)
# - Visual inspection
```

## Troubleshooting

### "FP8 not supported"

**Problem:** Your GPU doesn't support FP8
**Solution:** Use FP16 instead:
```bash
python src/tensorrt_fp8_builder.py --model 0 --precision fp16
```

### "Failed to load calibration data"

**Problem:** Calibration directory is empty or invalid
**Solutions:**
1. Check path: `/mnt/VSlite/dataset/data_same_batch_final_1024`
2. Verify .npy files exist
3. Skip calibration: `--calibration_samples 0`
4. Use FP16: `--precision fp16`

### "Batch size mismatch"

**Problem:** Inference batch size doesn't match engine build batch size
**Solution:** Use the same batch size:
```bash
# Build with batch size 4
python src/tensorrt_fp8_builder.py --batch_size 4 ...

# Inference with batch size 4
python src/tensorrt_inference_server.py --batch_size 4 ...
```

### "Engine build takes too long"

**Problem:** Building engine is slow
**Solutions:**
1. Reduce calibration samples: `--calibration_samples 100`
2. Reduce workspace: `--workspace_size 2048`
3. Use existing ONNX: `--skip_onnx`

### "Output quality degraded"

**Problem:** FP8 output has artifacts or lower quality
**Solutions:**
1. Increase calibration samples: `--calibration_samples 500`
2. Use better calibration data (more representative)
3. Use FP16 instead: `--precision fp16`

## Batch Size Considerations

### Choosing Batch Size

The batch size must be chosen at engine build time and cannot be changed at inference.

**Trade-offs:**

| Batch Size | Pros | Cons |
|------------|------|------|
| 1 | Lower memory, flexible | Slower throughput |
| 4 | Balanced | Moderate memory |
| 8+ | Best throughput | High memory, less flexible |

**Recommendation:** Use batch_size=4 for 1024x1024 tiles

### Multiple Engines

You can build multiple engines with different batch sizes:

```bash
# Build engines with different batch sizes
python src/tensorrt_fp8_builder.py --model 0 --precision fp8 --batch_size 1
python src/tensorrt_fp8_builder.py --model 0 --precision fp8 --batch_size 4
python src/tensorrt_fp8_builder.py --model 0 --precision fp8 --batch_size 8

# Use appropriate engine for your workload
python src/tensorrt_inference_server.py --engine 20_net_G_bs1_model0_fp8.engine --batch_size 1 ...
python src/tensorrt_inference_server.py --engine 20_net_G_bs4_model0_fp8.engine --batch_size 4 ...
```

## Production Deployment

### Recommended Workflow

1. **Development:** Use main.py with PyTorch for flexibility
2. **Optimization:** Build TensorRT engines with various precisions
3. **Validation:** Compare FP8/FP16 outputs vs FP32 reference
4. **Deployment:** Use tensorrt_inference_server.py in production

### Deployment Checklist

- [ ] Build TensorRT engine(s) for target hardware
- [ ] Validate output quality (FP8 vs FP32)
- [ ] Benchmark performance on target system
- [ ] Test with representative data
- [ ] Document engine configuration (precision, batch size)
- [ ] Set up monitoring/logging
- [ ] Create backup engines (FP16 fallback)

### Docker Deployment

For containerized deployment, ensure your Docker image includes:

```dockerfile
FROM nvidia/cuda:12.0-cudnn8-runtime-ubuntu22.04

# Install TensorRT
RUN apt-get update && apt-get install -y \
    python3-libnvinfer \
    python3-libnvinfer-dev

# Install Python dependencies
RUN pip install tensorrt pycuda numpy spectral-python pillow pyvips pandas

# Copy engine files
COPY saved_models/*.engine /app/saved_models/

# Copy inference script
COPY src/tensorrt_inference_server.py /app/

# Run inference
CMD ["python3", "/app/tensorrt_inference_server.py", "--engine", "20_net_G_bs4_model0_fp8.engine"]
```

## Summary

**Quick Start (3 commands):**

```bash
# 1. Build FP8 engine (one-time, 5-15 minutes)
python src/tensorrt_fp8_builder.py --model 0 --precision fp8 --batch_size 4

# 2. Run inference on single folder
python src/tensorrt_inference_server.py --engine 20_net_G_bs4_model0_fp8.engine --folder FOLDER_NAME --model 0

# 3. Run inference on multiple folders
python src/tensorrt_inference_server.py --engine 20_net_G_bs4_model0_fp8.engine --parent_folder /path/to/folders --model 0
```

**Key Benefits:**
- ✅ FP8 quantization (4x memory reduction)
- ✅ No PyTorch dependency at inference
- ✅ Faster inference on modern GPUs
- ✅ Same CLI as main.py
- ✅ Production-ready deployment

**Next Steps:**
1. Build your first engine with the Quick Start commands
2. Compare performance vs main.py
3. Validate output quality
4. Deploy to production

For questions or issues, refer to the Troubleshooting section or check the TensorRT documentation.
