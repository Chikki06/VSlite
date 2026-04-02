"""
TensorRT FP8 Engine Builder
Exports PyTorch model to ONNX and builds TensorRT engine with FP8 quantization.

Usage:
    python tensorrt_fp8_builder.py --model 0 --precision fp8 --calibration_samples 500
    python tensorrt_fp8_builder.py --model 1 --precision fp16
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List
import time

# Add parent directory to path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("ERROR: TensorRT or PyCUDA not available. Please install:")
    print("  pip install tensorrt")
    print("  pip install pycuda")
    sys.exit(1)

# Import model architecture (copy from main.py)
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return x, out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Replace AdaptiveMaxPool2d with combination of MaxPool2d and AdaptiveAvgPool2d
        # This is more TensorRT-friendly
        self.use_max_pool = True  # Flag to enable/disable max pooling branch
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels//reduction_ratio, input_channels))

    def forward(self, x):
        avg_values = self.avg_pool(x)
        # Use global max pooling instead of AdaptiveMaxPool2d
        # This is equivalent but more compatible with ONNX/TensorRT
        max_values = torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        out = self.mlp(avg_values) + self.mlp(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.bn(self.conv(out))
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(Up, self).__init__()
        activation = nn.ReLU(True)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(out_channels),
            activation
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            activation
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = self.conv(torch.cat([x2, x1], dim=1))
        return x


class AttGenerator_ResNet50_SkipCon(nn.Module):
    def __init__(self, input_channels, norm_layer=nn.BatchNorm2d, fpn_feature='decoder'):
        super(AttGenerator_ResNet50_SkipCon, self).__init__()
        
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        new_conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1 = new_conv1
        self.encoder = IntermediateLayerGetter(backbone, return_layers={
            "conv1": 'feat', "layer1": 'feat0', "layer2": 'feat1', 
            "layer3": 'feat2', "layer4": 'feat3'
        })

        assert fpn_feature == 'encoder' or fpn_feature == 'decoder'
        self.fpn_feature = fpn_feature
        self.ngf = 64
        self.n_downsampling = 5
        self.output_nc = 3
        for i in range(self.n_downsampling - 1):
            mult = 2**(self.n_downsampling - i)
            setattr(self, 'upsample_'+str(i), Up(self.ngf * mult, int(self.ngf * mult / 2) if i!=self.n_downsampling-2 else self.ngf))
            setattr(self, 'cbam_'+str(i), CBAM(self.ngf * mult))
        setattr(self, 'cbam_'+str(self.n_downsampling-1), CBAM(self.ngf))
        
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.ngf // 2, kernel_size=3, stride=2, padding=1, output_padding=1), 
            norm_layer(self.ngf // 2), 
            nn.ReLU(True), 
            nn.ReflectionPad2d(3), 
            nn.Conv2d(self.ngf // 2, self.output_nc, kernel_size=7, padding=0), 
            nn.Tanh()
        )
            
    def forward(self, input):
        x, features = self.encoder(input)
        cbam = getattr(self, 'cbam_0')
        x = cbam(x)
        for i in range(self.n_downsampling - 1):
            model = getattr(self, 'upsample_'+str(i))
            feature_key = 'feat' + (str(self.n_downsampling-i-3) if self.n_downsampling-i-3>=0 else '')
            cbam = getattr(self, 'cbam_'+str(i+1))
            feat = cbam(features[feature_key])
            x = model(x, feat)
            if self.fpn_feature == 'decoder':
                features[feature_key] = x
            elif self.fpn_feature == 'encoder':
                features[feature_key] = feat
        x = self.final_upsample(x)
        del features['feat']
        return x


class CalibrationDataLoader:
    """Data loader for INT8/FP8 calibration"""
    
    def __init__(self, calibration_dir, model_type=0, num_samples=100, batch_size=1):
        self.calibration_dir = calibration_dir
        self.model_type = model_type
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.current_idx = 0
        
        # Load calibration data
        self.data = self._load_calibration_data()
        
    def _load_calibration_data(self):
        """Load calibration samples from VS dataset structure (ir/, vis/, he/ folders)
        
        Note: Accepts patches of any size - will resize/pad to 1024x1024 for calibration
        """
        print(f"Loading {self.num_samples} calibration samples from {self.calibration_dir}...")
        
        if not os.path.exists(self.calibration_dir):
            raise ValueError(f"Calibration directory not found: {self.calibration_dir}")
        
        # Check for VS dataset structure (ir/, vis/, he/ subdirectories)
        ir_dir = os.path.join(self.calibration_dir, 'ir')
        vis_dir = os.path.join(self.calibration_dir, 'vis')
        
        if not os.path.exists(ir_dir):
            raise ValueError(f"IR directory not found: {ir_dir}")
        
        # Get list of IR files
        ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith('.npy')])
        
        if len(ir_files) == 0:
            raise ValueError(f"No .npy files found in {ir_dir}")
        
        # Load samples
        data_samples = []
        num_to_load = min(self.num_samples, len(ir_files))
        skipped_count = 0
        skip_reasons = {}
        
        for i, ir_file in enumerate(ir_files):
            if len(data_samples) >= num_to_load:
                break
                
            try:
                # Load IR data (10 channels)
                ir_path = os.path.join(ir_dir, ir_file)
                ir_data = np.load(ir_path).astype(np.float32)
                
                # Validate IR data shape
                if ir_data.ndim != 3:
                    skip_reasons['IR ndim != 3'] = skip_reasons.get('IR ndim != 3', 0) + 1
                    skipped_count += 1
                    continue
                    
                if ir_data.shape[0] != 10:
                    skip_reasons[f'IR channels != 10 (got {ir_data.shape[0]})'] = skip_reasons.get(f'IR channels != 10 (got {ir_data.shape[0]})', 0) + 1
                    skipped_count += 1
                    continue
                
                # Normalize IR data (same as training)
                ir_data = np.clip(ir_data, 0, 5) / 1.0
                
                if self.model_type == 0:
                    # IR-only model (10 channels)
                    sample = ir_data
                elif self.model_type == 1:
                    # IR+VIS model (13 channels)
                    # Load corresponding VIS data (3 channels)
                    vis_path = os.path.join(vis_dir, ir_file)
                    if not os.path.exists(vis_path):
                        skip_reasons['VIS file not found'] = skip_reasons.get('VIS file not found', 0) + 1
                        skipped_count += 1
                        continue
                    
                    vis_data = np.load(vis_path).astype(np.float32)
                    
                    # Validate VIS data shape
                    if vis_data.ndim != 3:
                        skip_reasons['VIS ndim != 3'] = skip_reasons.get('VIS ndim != 3', 0) + 1
                        skipped_count += 1
                        continue
                        
                    if vis_data.shape[0] != 3:
                        skip_reasons[f'VIS channels != 3 (got {vis_data.shape[0]})'] = skip_reasons.get(f'VIS channels != 3 (got {vis_data.shape[0]})', 0) + 1
                        skipped_count += 1
                        continue
                    
                    # Check that IR and VIS have same spatial dimensions
                    if ir_data.shape[1:] != vis_data.shape[1:]:
                        skip_reasons[f'Shape mismatch: IR {ir_data.shape[1:]} vs VIS {vis_data.shape[1:]}'] = skip_reasons.get(f'Shape mismatch: IR {ir_data.shape[1:]} vs VIS {vis_data.shape[1:]}', 0) + 1
                        skipped_count += 1
                        continue
                    
                    vis_data = vis_data / 256.0  # Normalize VIS
                    sample = np.concatenate([ir_data, vis_data], axis=0)
                else:
                    raise ValueError(f"Invalid model_type: {self.model_type}")
                
                # Resize/pad patches to 1024x1024 for calibration
                c, h, w = sample.shape
                if h != 1024 or w != 1024:
                    # Create a padded/cropped version to 1024x1024
                    target_sample = np.zeros((c, 1024, 1024), dtype=np.float32)
                    
                    # Copy what fits (center crop or pad)
                    h_start = max(0, (h - 1024) // 2)
                    w_start = max(0, (w - 1024) // 2)
                    h_end = min(h, h_start + 1024)
                    w_end = min(w, w_start + 1024)
                    
                    target_h_start = max(0, (1024 - h) // 2)
                    target_w_start = max(0, (1024 - w) // 2)
                    target_h_end = target_h_start + (h_end - h_start)
                    target_w_end = target_w_start + (w_end - w_start)
                    
                    target_sample[:, target_h_start:target_h_end, target_w_start:target_w_end] = \
                        sample[:, h_start:h_end, w_start:w_end]
                    
                    sample = target_sample
                
                data_samples.append(sample)
                
                if len(data_samples) % 50 == 0:
                    print(f"  Loaded {len(data_samples)}/{num_to_load} samples (skipped {skipped_count})")
                    
            except Exception as e:
                skip_reasons[f'Exception: {str(e)[:50]}'] = skip_reasons.get(f'Exception: {str(e)[:50]}', 0) + 1
                skipped_count += 1
                continue
        
        if len(data_samples) == 0:
            print(f"\nSkip reasons:")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} samples")
            raise ValueError(f"No valid calibration samples loaded (skipped {skipped_count})")
        
        print(f"Calibration data loaded: {len(data_samples)} samples (skipped {skipped_count} invalid samples)")
        if skip_reasons:
            print(f"Skip reasons summary:")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {reason}: {count} samples")
        return data_samples
    
    def get_batch(self):
        """Get next batch for calibration"""
        if self.current_idx < len(self.data):
            batch = self.data[self.current_idx:self.current_idx + self.batch_size]
            self.current_idx += self.batch_size
            
            # Stack batch
            batch_array = np.stack(batch, axis=0)
            return batch_array
        return None
    
    def reset(self):
        """Reset iterator"""
        self.current_idx = 0


class FP8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    Custom calibrator for FP8 quantization
    Note: TensorRT uses INT8 calibrator interface for FP8 as well
    """
    
    def __init__(self, calibration_loader, cache_file='calibration_cache.bin'):
        super().__init__()
        self.calibration_loader = calibration_loader
        self.cache_file = cache_file
        self.device_input = None
        self.batch_size = calibration_loader.batch_size
        
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """Get next batch of calibration data"""
        batch = self.calibration_loader.get_batch()
        
        if batch is None:
            return None
        
        # Allocate device memory if not already done
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch.nbytes)
        
        # Copy data to device
        cuda.memcpy_htod(self.device_input, batch)
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        """Read calibration cache if it exists"""
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to disk"""
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def export_to_onnx(model, model_path, model_type, batch_size=1):
    """Export PyTorch model to ONNX format"""
    print("\n" + "="*80)
    print("STEP 1: Exporting PyTorch model to ONNX")
    print("="*80)
    
    channels = 13 if model_type == 1 else 10
    input_shape = (batch_size, channels, 1024, 1024)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).cuda()
    
    # Set model to eval mode and disable gradient computation
    model.eval()
    
    # Create ONNX filename
    onnx_path = model_path.replace('.pth', f'_bs{batch_size}_model{model_type}.onnx')
    
    print(f"Input shape: {input_shape}")
    print(f"Output path: {onnx_path}")
    print(f"Exporting model to ONNX...")
    
    # Export to ONNX
    start_time = time.time()
    
    try:
        # Export directly from the model (no tracing) with fixed batch size
        # This avoids issues with traced models and dynamic_axes in PyTorch 2.9+
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,  # Use opset 18 (PyTorch 2.9+ requirement)
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False,
                # Don't use dynamic_axes for fixed batch size - avoids PyTorch 2.9+ issues
            )
        
        # Post-process ONNX to fix bias issues
        print("Post-processing ONNX model to fix bias tensors...")
        import onnx
        from onnx import numpy_helper
        
        onnx_model = onnx.load(onnx_path)
        graph = onnx_model.graph
        
        # Collect all bias names that need to be removed
        biases_to_remove = set()
        nodes_fixed = 0
        
        # First pass: identify nodes with mismatched biases
        for node in graph.node:
            if node.op_type == 'Conv' and len(node.input) == 3:
                # Conv node has bias (third input)
                bias_name = node.input[2]
                weight_name = node.input[1]
                
                # Find weight and bias initializers
                weight_init = None
                bias_init = None
                for init in graph.initializer:
                    if init.name == weight_name:
                        weight_init = init
                    if init.name == bias_name:
                        bias_init = init
                
                if weight_init and bias_init:
                    # Check if bias shape matches weight output channels
                    weight_shape = [dim for dim in weight_init.dims]
                    bias_shape = [dim for dim in bias_init.dims]
                    
                    if len(weight_shape) > 0 and len(bias_shape) > 0:
                        if weight_shape[0] != bias_shape[0]:
                            # Bias shape mismatch - mark for removal
                            print(f"  Removing mismatched bias from node {node.name}: weight channels={weight_shape[0]}, bias shape={bias_shape[0]}")
                            biases_to_remove.add(bias_name)
                            nodes_fixed += 1
        
        # Second pass: remove bias inputs from Conv nodes
        for node in graph.node:
            if node.op_type == 'Conv' and len(node.input) == 3:
                bias_name = node.input[2]
                if bias_name in biases_to_remove:
                    # Remove bias input from node (keep only input and weight)
                    del node.input[2]
        
        # Third pass: remove bias initializers and inputs
        # Remove from initializers
        to_remove_indices = []
        for i, init in enumerate(graph.initializer):
            if init.name in biases_to_remove:
                to_remove_indices.append(i)
        for i in reversed(to_remove_indices):
            del graph.initializer[i]
        
        # Remove from inputs
        to_remove_indices = []
        for i, inp in enumerate(graph.input):
            if inp.name in biases_to_remove:
                to_remove_indices.append(i)
        for i in reversed(to_remove_indices):
            del graph.input[i]
        
        if nodes_fixed > 0:
            print(f"  Fixed {nodes_fixed} nodes with mismatched biases")
            # Save corrected model
            onnx.save(onnx_model, onnx_path)
            print(f"  Corrected ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"ERROR during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    export_time = time.time() - start_time
    print(f"ONNX export completed in {export_time:.2f} seconds")
    print(f"ONNX model saved to: {onnx_path}")
    
    # Verify the ONNX model
    try:
        import onnx
        print("Verifying ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except Exception as e:
        print(f"Warning: ONNX model verification failed: {e}")
    
    return onnx_path


def build_tensorrt_engine(onnx_path, engine_path, precision='fp16', 
                          calibration_loader=None, workspace_size=4096):
    """Build TensorRT engine from ONNX model"""
    print("\n" + "="*80)
    print(f"STEP 2: Building TensorRT engine with {precision.upper()} precision")
    print("="*80)
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print(f"Parsing ONNX model: {onnx_path}")
    
    # Change to the directory containing the ONNX file so TensorRT can find .data file
    original_dir = os.getcwd()
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    onnx_filename = os.path.basename(onnx_path)
    
    try:
        os.chdir(onnx_dir)
        print(f"Changed directory to: {onnx_dir}")
        
        with open(onnx_filename, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("ERROR: Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
    finally:
        os.chdir(original_dir)
    
    print(f"ONNX model parsed successfully")
    print(f"Network inputs: {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * 1024 * 1024)
    print(f"Workspace size: {workspace_size} MB")
    
    # Set precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        else:
            print("WARNING: FP16 not supported on this platform, using FP32")
    
    elif precision == 'fp8':
        # FP8 requires TensorRT 9.0+ and Hopper/Ada GPU
        if hasattr(trt.BuilderFlag, 'FP8'):
            config.set_flag(trt.BuilderFlag.FP8)
            print("FP8 mode enabled")
            
            # FP8 quantization requires calibration
            if calibration_loader is not None:
                print("Setting up FP8 calibration...")
                cache_file = engine_path.replace('.engine', '_calibration.cache')
                calibrator = FP8Calibrator(calibration_loader, cache_file)
                config.int8_calibrator = calibrator
                config.set_flag(trt.BuilderFlag.INT8)  # FP8 uses INT8 calibration
                print("Calibration configured")
            else:
                print("WARNING: No calibration data provided for FP8. Quality may be degraded.")
        else:
            print("ERROR: FP8 not supported. TensorRT 9.0+ required with Hopper/Ada GPU")
            print("Falling back to FP16")
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    print(f"Building TensorRT engine (this may take several minutes)...")
    start_time = time.time()
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return None
    
    build_time = time.time() - start_time
    print(f"Engine built successfully in {build_time:.2f} seconds")
    
    # Save engine
    print(f"Saving engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved successfully")
    print(f"Engine size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
    
    return engine_path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Build TensorRT FP8 engine from PyTorch model')
    
    parser.add_argument('--model', type=int, default=0, choices=[0, 1],
                       help='Model type: 0 for 10-channel IR (default), 1 for 13-channel IR+VIS')
    
    parser.add_argument('--model_name', type=str, default=None,
                       help='Custom model filename (located in saved_models/). Default: 15_net_G.pth or 15_net_G.pth')
    
    parser.add_argument('--precision', type=str, default='fp16', 
                       choices=['fp32', 'fp16', 'fp8'],
                       help='TensorRT precision mode (default: fp16)')
    
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference (default: 1)')
    
    parser.add_argument('--workspace_size', type=int, default=4096,
                       help='TensorRT workspace size in MB (default: 4096)')
    
    parser.add_argument('--calibration_samples', type=int, default=100,
                       help='Number of samples for FP8 calibration (default: 100)')
    
    parser.add_argument('--calibration_dir', type=str, 
                       default='/mnt/VSlite/dataset/data_same_batch_final_1024',
                       help='Directory containing calibration data')
    
    parser.add_argument('--skip_onnx', action='store_true',
                       help='Skip ONNX export if ONNX file already exists')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("="*80)
    print("TensorRT FP8 Engine Builder")
    print("="*80)
    print(f"Model type: {'IR+VIS (13 channels)' if args.model == 1 else 'IR only (10 channels)'}")
    print(f"Precision: {args.precision.upper()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workspace size: {args.workspace_size} MB")
    
    if args.precision == 'fp8':
        print(f"Calibration samples: {args.calibration_samples}")
        print(f"Calibration directory: {args.calibration_dir}")
    
    # Determine model path
    if args.model_name:
        model_path = os.path.join(PARENT_DIR, 'saved_models', args.model_name)
    else:
        default_model = '15_net_G.pth' if args.model == 1 else '20_net_G.pth'
        model_path = os.path.join(PARENT_DIR, 'saved_models', default_model)
    
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    channels = 13 if args.model == 1 else 10
    model = AttGenerator_ResNet50_SkipCon(input_channels=channels)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    print("Model loaded successfully")
    
    # Export to ONNX
    onnx_path = model_path.replace('.pth', f'_bs{args.batch_size}_model{args.model}.onnx')
    
    if args.skip_onnx and os.path.exists(onnx_path):
        print(f"\nSkipping ONNX export, using existing file: {onnx_path}")
    else:
        onnx_path = export_to_onnx(model, model_path, args.model, args.batch_size)
    
    # Prepare calibration data for FP8
    calibration_loader = None
    if args.precision == 'fp8':
        try:
            calibration_loader = CalibrationDataLoader(
                args.calibration_dir,
                model_type=args.model,
                num_samples=args.calibration_samples,
                batch_size=args.batch_size
            )
        except Exception as e:
            print(f"WARNING: Failed to load calibration data: {e}")
            print("Continuing without calibration (may affect FP8 quality)")
    
    # Build TensorRT engine
    engine_path = model_path.replace('.pth', f'_bs{args.batch_size}_model{args.model}_{args.precision}.engine')
    
    engine_path = build_tensorrt_engine(
        onnx_path,
        engine_path,
        precision=args.precision,
        calibration_loader=calibration_loader,
        workspace_size=args.workspace_size
    )
    
    if engine_path:
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"TensorRT engine created: {engine_path}")
        print(f"\nTo use this engine, run:")
        print(f"  python src/tensorrt_inference_server.py --engine {os.path.basename(engine_path)} --model {args.model}")
    else:
        print("\n" + "="*80)
        print("FAILED!")
        print("="*80)
        print("TensorRT engine building failed. Check errors above.")


if __name__ == '__main__':
    main()
