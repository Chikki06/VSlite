"""
TensorRT utilities for model optimization.
This module is only imported when TensorRT functionality is needed.
"""

import os
import torch
import time

# Import TensorRT libraries only when this module is loaded
try:
    import torch_tensorrt
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT libraries not available. TensorRT optimization will not work.")


def check_tensorrt_available():
    """Check if TensorRT is available"""
    return TENSORRT_AVAILABLE


def convert_to_torchscript(model, input_shape):
    """Convert PyTorch model to TorchScript without precision changes"""
    start = time.time()
    print("Converting model to TorchScript without precision changes...")
    
    # Create dummy input for tracing
    dummy_input = torch.randn(input_shape).cuda()
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    print(f"TorchScript conversion completed successfully in {time.time() - start:.2f} seconds!")
    return traced_model


def convert_to_tensorrt(model, input_shape, precision='fp16', workspace_size=1024):
    """Convert PyTorch model to TensorRT optimized model"""
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT libraries are not available. Cannot perform TensorRT conversion.")
    
    start = time.time()
    print(f"Converting model to TensorRT with {precision} precision...")
    
    # Create dummy input for tracing
    dummy_input = torch.randn(input_shape).cuda()
    
    # Trace the model first
    print("Tracing model...")
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Validate traced model
    print("Validating traced model...")
    with torch.no_grad():
        original_output = model(dummy_input)
        traced_output = traced_model(dummy_input)
        diff = torch.abs(original_output - traced_output).max()
        print(f"Max difference between original and traced: {diff.item():.6f}")
        
        if diff > 1e-4:
            print("Warning: Large difference detected in tracing!")
            # If this is a problem, consider using torch.jit.script instead of trace.
            # traced_model = torch.jit.script(model)
    
    # Set precision
    if precision == 'fp16':
        enabled_precisions = {torch.float, torch.half}
    else:
        enabled_precisions = {torch.float}

    # Only use supported compile args and set ir="torchscript"
    compile_args = {
        "inputs": [torch_tensorrt.Input(
            min_shape=input_shape,
            opt_shape=input_shape,
            max_shape=input_shape,
            dtype=torch.float32
        )],
        "enabled_precisions": enabled_precisions,
        "workspace_size": workspace_size * 1024 * 1024,
        "ir": "torchscript"
    }
    print("Compiling with TensorRT...")
    try:
        trt_model = torch_tensorrt.compile(traced_model, **compile_args)
        
        # Validate TensorRT model
        print("Validating TensorRT model...")
        with torch.no_grad():
            trt_output = trt_model(dummy_input)
            diff = torch.abs(original_output - trt_output).max()
            print(f"Max difference between original and TensorRT: {diff.item():.6f}")
            if trt_output.abs().max() < 1e-6:
                print("ERROR: TensorRT model producing near-zero output!")
                print(f"Output stats - Min: {trt_output.min()}, Max: {trt_output.max()}, Mean: {trt_output.mean()}")
                return None
    except Exception as e:
        print(f"TensorRT compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"TensorRT conversion completed successfully in {time.time() - start:.2f} seconds!")
    return trt_model


def save_tensorrt_engine(trt_model, model_path, model_type, precision):
    """Save TensorRT engine to disk"""
    engine_path = model_path.replace('.pth', f'_trt_{model_type}_{precision}.ts')
    torch.jit.save(trt_model, engine_path)
    print(f"TensorRT engine saved to: {engine_path}")
    return engine_path


def load_tensorrt_engine(engine_path):
    """Load TensorRT engine from disk"""
    if os.path.exists(engine_path):
        print(f"Loading TensorRT engine from: {engine_path}")
        return torch.jit.load(engine_path)
    return None


def get_engine_path(model_path, model_type, precision, batch_size):
    """Generate engine path with batch size in filename"""
    if precision:
        model_suffix = f'_trt_{model_type}_{precision}_bs{batch_size}.ts'
    else:
        model_suffix = f'_torchscript_{model_type}_bs{batch_size}.ts'
    return model_path.replace('.pth', model_suffix)


def save_optimized_model(model, model_path, model_type, optimization_type, precision=None, batch_size=None):
    """Save optimized model to disk with batch size in filename"""
    engine_path = get_engine_path(model_path, model_type, precision, batch_size)
    torch.jit.save(model, engine_path)
    print(f"{optimization_type} model saved to: {engine_path}")
    return engine_path


def load_optimized_model(model_path, model_type, use_tensorrt, precision=None, batch_size=None):
    """Load optimized model from disk with batch size validation"""
    try:
        engine_path = get_engine_path(model_path, model_type, precision, batch_size)
        optimization_type = f"TensorRT ({precision})" if use_tensorrt and precision else "TorchScript"
        
        if os.path.exists(engine_path):
            print(f"Loading {optimization_type} model from: {engine_path}")
            optimized_model = torch.jit.load(engine_path)
            return optimized_model
        
        return None
    except Exception as e:
        print(f"Error loading optimized model: {e}")
        return None
