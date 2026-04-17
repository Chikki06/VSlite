"""
TensorRT FP8 Inference Server
Runs inference using TensorRT engine with FP8 precision.
Similar to main.py but uses native TensorRT runtime (no PyTorch dependency at inference).

Usage:
    # Single folder
    python src/tensorrt_inference_server.py --engine 20_net_G_bs4_model0_fp8.engine --folder FOLDER_NAME
    
    # Batch processing
    python src/tensorrt_inference_server.py --engine 20_net_G_bs4_model0_fp8.engine --parent_folder /path/to/folders
    
    # With different batch size
    python src/tensorrt_inference_server.py --engine 20_net_G_bs1_model0_fp16.engine --folder FOLDER_NAME --batch_size 1
"""

import os
import sys
import argparse
import numpy as np
import time
from datetime import datetime
from typing import List, Dict
import json
import pandas as pd
import torch

# Add parent directory to path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

# Import dataset and collate function from main.py
try:
    # Try importing from parent directory (main.py is in src/)
    from src.main import SingleFolderDataset, custom_collate_fn
except ImportError:
    try:
        # Try importing directly if main.py is in same directory
        from main import SingleFolderDataset, custom_collate_fn
    except ImportError:
        print("ERROR: Could not import SingleFolderDataset and custom_collate_fn from main.py")
        print("Please ensure main.py is in the correct location.")
        sys.exit(1)

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

try:
    from PIL import Image
    import pyvips
    from torch.utils.data import DataLoader
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False
    print("ERROR: Required imaging libraries not available. Please install:")
    print("  pip install pillow pyvips")
    sys.exit(1)

Image.MAX_IMAGE_PIXELS = None


class TensorRTInferenceEngine:
    """TensorRT inference engine wrapper"""
    
    def __init__(self, engine_path, batch_size=1):
        """
        Initialize TensorRT engine
        
        Args:
            engine_path: Path to .engine file
            batch_size: Batch size used during engine building
        """
        self.engine_path = engine_path
        self.batch_size = batch_size
        self.logger = trt.Logger(trt.Logger.ERROR)
        
        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        self.engine = self._load_engine()
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        print(f"TensorRT engine loaded successfully")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Number of bindings: {len(self.bindings)}")
        
    def _load_engine(self):
        """Load TensorRT engine from file"""
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
        
        return engine
    
    def _allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        # Use new TensorRT API (TensorRT 8.5+)
        num_io_tensors = self.engine.num_io_tensors
        
        for i in range(num_io_tensors):
            # Get tensor name
            tensor_name = self.engine.get_tensor_name(i)
            
            # Get tensor shape and dtype
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # Calculate size
            size = trt.volume(shape)
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the appropriate list
            bindings.append(int(device_mem))
            
            # Check if input or output
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({
                    'host': host_mem, 
                    'device': device_mem, 
                    'shape': shape, 
                    'dtype': dtype,
                    'name': tensor_name
                })
                print(f"Input tensor: {tensor_name}, shape: {shape}, dtype: {dtype}")
            else:
                outputs.append({
                    'host': host_mem, 
                    'device': device_mem, 
                    'shape': shape, 
                    'dtype': dtype,
                    'name': tensor_name
                })
                print(f"Output tensor: {tensor_name}, shape: {shape}, dtype: {dtype}")
        
        return inputs, outputs, bindings, stream
    
    def infer(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data: numpy array of shape (batch_size, channels, height, width)
            
        Returns:
            Output numpy array
        """
        # Ensure input data is in correct format
        input_data = np.asarray(input_data, dtype=self.inputs[0]['dtype'])
        
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Set tensor addresses for new API
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        
        # Run inference with new API
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Transfer predictions back from device
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchronize the stream
        self.stream.synchronize()
        
        # Reshape output
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        
        return output
    
    def __del__(self):
        """Cleanup"""
        # Free CUDA memory
        for inp in self.inputs:
            if 'device' in inp:
                inp['device'].free()
        for out in self.outputs:
            if 'device' in out:
                out['device'].free()


def process_folder(engine, folder_name, files_dir, model_type=0, save_png=False, output_name=None):
    """
    Process a single folder with TensorRT engine using dataset from main.py
    
    MEMORY-EFFICIENT DESIGN (matching main.py exactly):
    - Full images kept on CPU (IR: ~10GB stays in system RAM, not GPU)
    - Only batch buffer on GPU (4 tiles × 52MB = ~200MB GPU memory)
    - Tiles extracted on CPU, moved to GPU one batch at a time
    - Outputs written directly to CPU array (no GPU accumulation)
    - Direct buffer indexing (no list accumulation)
    - Explicit GPU memory cleanup after processing
    
    Args:
        engine: TensorRTInferenceEngine instance
        folder_name: Name of folder to process
        files_dir: Parent directory containing folders
        model_type: 0 for IR-only, 1 for IR+VIS
        save_png: Save as PNG instead of SZI
        output_name: Custom output filename
    
    Returns:
        Dictionary with timing information
    """
    print(f"\nProcessing folder: {folder_name}")
    folder_start_time = time.time()
    
    # Create dataset using SingleFolderDataset from main.py
    try:
        dataset = SingleFolderDataset(files_dir, folder_name, model_type)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    except FileNotFoundError as e:
        print(f"  ERROR: {str(e)}")
        raise
    
    # Process the batch (there's only one item in single folder dataset)
    for batch in dataloader:
        bsq_load_time = batch['bsq_load_time'].item()
        print(f"  BSQ load time: {bsq_load_time:.2f}s")
        
        inference_start = time.time()
        
        # Keep IR data on CPU (like main.py) - only move tiles to GPU
        ir = batch['ir']  # Shape: [1, bands, height, width] - STAYS ON CPU
        
        # Process VIS data if needed
        if model_type == 1:
            vis_raw = batch['vis_raw'][0]
            vis_pixel_size = batch['vis_pixel_size'][0]
            
            if vis_raw is not None and vis_pixel_size is not None:
                # Scale VIS to target pixel size (matching main.py logic)
                target_pixel_size = 0.69
                scale_factor = vis_pixel_size / target_pixel_size
                
                _, original_h, original_w = vis_raw.shape
                new_h = int(original_h * scale_factor)
                new_w = int(original_w * scale_factor)
                
                print(f"  Scaling VIS from {original_h}x{original_w} to {new_h}x{new_w} (scale: {scale_factor:.3f})")
                
                # Keep on CPU for now (like main.py) - will move tiles to GPU later
                img_V = torch.nn.functional.interpolate(
                    vis_raw.unsqueeze(0),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                img_V = torch.clamp(img_V, 0, 1)
            else:
                # Create dummy VIS data
                _, bands, ir_h, ir_w = ir.shape
                ir_pixel_size = 2.0
                vis_target_pixel_size = 0.69
                scale_factor = ir_pixel_size / vis_target_pixel_size
                
                vis_h = int(ir_h * scale_factor)
                vis_w = int(ir_w * scale_factor)
                
                print(f"  Creating dummy VIS data: {vis_h}x{vis_w}")
                img_V = torch.zeros((3, vis_h, vis_w))  # Keep on CPU
            
            vis = img_V.unsqueeze(0)  # Add batch dimension (stays on CPU)
        else:
            vis = None
        
        # Get dimensions
        b, bands, _, _ = ir.shape
        if vis is not None:
            _, _, r, c = vis.shape
        else:
            _, _, r, c = ir.shape
        
        print(f"  IR shape: {ir.shape}")
        if vis is not None:
            print(f"  VIS shape: {vis.shape}")
        
        # Process tiles (matching main.py logic)
        fact = 1024
        channels = 13 if model_type == 1 else 10
        batch_size = engine.batch_size
        scaling_factor = 0.69/2  # VIS to IR pixel size ratio
        
        rn = int(np.ceil(r / fact))
        cn = int(np.ceil(c / fact))
        total_tiles = rn * cn
        
        print(f"  Image size: {r}x{c}")
        print(f"  Tiles: {rn}x{cn} = {total_tiles} tiles")
        print(f"  Batch size: {batch_size}")
        
        # Initialize output array on CPU (matching main.py - only processed tiles go to GPU)
        fhr = np.zeros((b, 3, int(rn)*fact+1, int(cn)*fact+1), dtype=np.float32)
        
        # Prepare GPU input data tensor for batch processing (ONLY this buffer on GPU)
        DATA_gpu = torch.zeros((batch_size, channels, fact, fact), dtype=torch.float32, device='cuda')
        
        # Create tile indices (matching main.py)
        I = list(range(0, int(rn)*fact+1, fact))
        J = list(range(0, int(cn)*fact+1, fact))
        
        # Process tiles in batches for efficiency (matching main.py memory-efficient approach)
        tiles_processed = 0
        patch_count = 0  # Use counter instead of list (like main.py)
        tile_coords = [None] * batch_size  # Pre-allocate fixed-size array
        
        for ii in range(len(I)-1):
            for jj in range(len(J)-1):
                # Extract tiles from CPU tensors (matching main.py)
                # Extract IR patch with scaling (VIS coordinates -> IR coordinates)
                ir_start_row = int(I[ii] * scaling_factor)
                ir_end_row = int(I[ii+1] * scaling_factor)
                ir_start_col = int(J[jj] * scaling_factor)
                ir_end_col = int(J[jj+1] * scaling_factor)
                
                xx = ir[:, :, ir_start_row:ir_end_row, ir_start_col:ir_end_col]
                
                # Pad if necessary (CPU operation for now)
                expected_ir_size = int(fact * scaling_factor)
                pad_rows = expected_ir_size - xx.shape[2] if xx.shape[2] < expected_ir_size else 0
                pad_cols = expected_ir_size - xx.shape[3] if xx.shape[3] < expected_ir_size else 0
                xx = torch.nn.functional.pad(xx, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                
                # Interpolate IR to fact x fact (CPU operation)
                xx = torch.nn.functional.interpolate(xx, size=(fact, fact), mode='bicubic', align_corners=True)
                
                # Extract VIS patch if needed (CPU operation)
                if vis is not None:
                    vv = vis[:, :, I[ii]:I[ii+1], J[jj]:J[jj+1]]
                    
                    # Pad VIS if necessary (CPU operation)
                    pad_rows = fact - vv.shape[2] if vv.shape[2] < fact else 0
                    pad_cols = fact - vv.shape[3] if vv.shape[3] < fact else 0
                    vv = torch.nn.functional.pad(vv, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                    
                    # Concatenate IR and VIS (CPU operation)
                    input_patch = torch.cat([xx, vv], dim=1)
                else:
                    input_patch = xx
                
                # Move tile to GPU and store in batch buffer (matching main.py)
                DATA_gpu[patch_count] = input_patch[0].cuda()  # Move single tile to GPU
                tile_coords[patch_count] = (ii, jj)
                patch_count += 1
                
                # Process batch when full or at end
                if patch_count == batch_size or (ii == len(I)-2 and jj == len(J)-2):
                    # Zero-pad remaining slots efficiently if needed
                    if patch_count < batch_size:
                        DATA_gpu[patch_count:] = 0
                    
                    # Convert to numpy only once for TensorRT (single CPU-GPU transfer)
                    DATA_np = DATA_gpu.cpu().numpy()
                    
                    # Run inference on batch
                    output = engine.infer(DATA_np)
                    
                    # Store outputs directly in CPU array (no GPU tensor needed)
                    for k in range(patch_count):
                        tile_ii, tile_jj = tile_coords[k]
                        tile_output = output[k]  # Shape: (3, fact, fact) - already numpy
                        fhr[0, :, I[tile_ii]:I[tile_ii+1], J[tile_jj]:J[tile_jj+1]] = tile_output
                    
                    tiles_processed += patch_count
                    if tiles_processed % (10 * batch_size) == 0 or tiles_processed % 10 < batch_size:
                        print(f"  Progress: {tiles_processed}/{total_tiles} tiles ({tiles_processed*100//total_tiles}%)")
                    
                    # Reset counter (no list to clear - memory stays allocated in DATA_gpu buffer)
                    patch_count = 0
        
        inference_time = time.time() - inference_start
        print(f"  Inference time: {inference_time:.2f}s ({inference_time/total_tiles:.3f}s per tile)")
        
        # Post-process output (already on CPU)
        write_start = time.time()
        
        # Explicitly free GPU memory (force cleanup)
        del DATA_gpu
        torch.cuda.empty_cache()
        
        # Crop to actual image size
        fhr = fhr[:, :, :r, :c]
        
        # Convert to uint8
        fhr = (fhr + 1.0) / 2.0  # Scale from [-1, 1] to [0, 1]
        fhr = np.clip(fhr * 255, 0, 255).astype(np.uint8)
        
        # Transpose to HWC format
        output_array = np.transpose(fhr[0], (1, 2, 0))  # CHW -> HWC
        
        # Save output
        output_filename = output_name if output_name else folder_name
        
        if save_png:
            save_png_output(output_array, output_filename, model_type)
        else:
            create_szi(output_array, output_filename, model_type)
            update_json(output_filename, r, c, model_type)
        
        write_time = time.time() - write_start
        folder_time = time.time() - folder_start_time
        
        print(f"  Write time: {write_time:.2f}s")
        print(f"  Total time: {folder_time:.2f}s")
        
        return {
            'folder_name': folder_name,
            'bsq_load_time': bsq_load_time,
            'inference_time': inference_time,
            'write_time': write_time,
            'total_time': folder_time,
            'num_tiles': total_tiles,
            'image_size': f"{c}x{r}"
        }


def save_png_output(img_array, filename, model_type=0):
    """Save output as PNG file"""
    subdir = 'IR' if model_type == 0 else 'IR+VIS'
    output_dir = os.path.join(PARENT_DIR, 'imagesT', subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{filename}.png')
    
    img = Image.fromarray(img_array)
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    img.save(output_path, 'PNG')
    print(f"  Saved PNG: {output_path}")


def create_szi(img_array, filename, model_type=0):
    """Create .szi file and thumbnail using pyvips"""
    subdir = 'IR' if model_type == 0 else 'IR+VIS'
    viewer_files_dir = os.path.join(PARENT_DIR, '..', 'viewer', 'files', subdir)
    viewer_thumbnail_dir = os.path.join(PARENT_DIR, '..', 'viewer', 'thumbnail', subdir)
    os.makedirs(viewer_files_dir, exist_ok=True)
    os.makedirs(viewer_thumbnail_dir, exist_ok=True)
    
    # Convert numpy array to vips image
    height, width, bands = img_array.shape
    linear = img_array.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands, 'uchar')
    
    # Save as deep zoom (.szi)
    szi_path = os.path.join(viewer_files_dir, f'{filename}.szi')
    
    if os.path.exists(szi_path):
        os.remove(szi_path)
    
    vi.dzsave(szi_path, overlap=0, container='zip', suffix='.jpg',
              compression=0, tile_size=1024)
    
    # Save thumbnail
    thumbnail = vi.thumbnail_image(400)
    thumbnail_path = os.path.join(viewer_thumbnail_dir, f"{filename}.jpg")
    thumbnail.jpegsave(thumbnail_path, Q=75)
    
    print(f"  Saved SZI: {szi_path}")


def update_json(filename, height, width, model_type):
    """Update JSON list with image dimensions"""
    json_path = os.path.join(PARENT_DIR, '..', 'viewer', 'static', 'available.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    resolution_str = f"{width}x{height}"
    
    # Update or add entry
    for entry in data:
        if entry.get('name') == filename and entry.get('model_type') == model_type:
            entry['resolution'] = resolution_str
            break
    else:
        data.append({'name': filename, 'resolution': resolution_str, 'model_type': model_type})
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def log_results_to_csv(results, run_config):
    """Log run results to CSV file"""
    csv_path = os.path.join(PARENT_DIR, 'tensorrt_runtime_logs.csv')
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare data
    rows = []
    for result in results:
        row = {
            'timestamp': timestamp,
            'engine': run_config['engine_name'],
            'model_type': run_config['model_type'],
            'precision': run_config['precision'],
            'batch_size': run_config['batch_size'],
            'folder_name': result['folder_name'],
            'image_size': result['image_size'],
            'num_tiles': result['num_tiles'],
            'bsq_load_time': result['bsq_load_time'],
            'inference_time': result['inference_time'],
            'write_time': result['write_time'],
            'total_time': result['total_time']
        }
        rows.append(row)
    
    # Create DataFrame
    df_new = pd.DataFrame(rows)
    
    # Append to existing CSV or create new
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_merged = df_new
    
    df_merged.to_csv(csv_path, index=False)
    print(f"\nResults logged to: {csv_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='TensorRT FP8 inference server for hyperspectral images')
    
    parser.add_argument('--engine', type=str, required=True,
                       help='TensorRT engine filename (in saved_models/ directory)')
    
    parser.add_argument('--folder', type=str, default=None,
                       help='Specific folder name to process (optional)')
    
    parser.add_argument('--parent_folder', type=str, default=None,
                       help='Parent directory containing multiple folders (optional)')
    
    parser.add_argument('--model', type=int, default=0, choices=[0, 1],
                       help='Model type: 0 for IR (10-channel), 1 for IR+VIS (13-channel)')
    
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (must match engine build batch size, default: 4)')
    
    parser.add_argument('--save_png', action='store_true', default=False,
                       help='Save output as PNG instead of SZI')
    
    parser.add_argument('--output_name', type=str, default=None,
                       help='Custom output filename (without extension)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("="*80)
    print("TensorRT FP8 Inference Server")
    print("="*80)
    
    # Construct engine path
    engine_path = os.path.join(PARENT_DIR, 'saved_models', args.engine)
    
    if not os.path.exists(engine_path):
        print(f"ERROR: Engine file not found: {engine_path}")
        print("\nPlease build the engine first using:")
        print(f"  python src/tensorrt_fp8_builder.py --model {args.model} --precision fp8")
        return
    
    # Extract precision from engine filename
    if 'fp8' in args.engine:
        precision = 'fp8'
    elif 'fp16' in args.engine:
        precision = 'fp16'
    else:
        precision = 'fp32'
    
    print(f"Engine: {args.engine}")
    print(f"Model type: {'IR+VIS (13 channels)' if args.model == 1 else 'IR only (10 channels)'}")
    print(f"Precision: {precision.upper()}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize TensorRT engine
    print("\nInitializing TensorRT engine...")
    engine = TensorRTInferenceEngine(engine_path, batch_size=args.batch_size)
    
    # Determine folders to process
    files_dir = os.path.join(PARENT_DIR, 'files')
    
    if args.folder:
        # Handle full path or folder name
        folder_arg = args.folder.strip('"').strip("'")
        
        # Check if it's a full path or just a folder name
        if os.path.exists(folder_arg) and os.path.isdir(folder_arg):
            # It's a full path - split into parent dir and folder name
            files_dir = os.path.dirname(folder_arg)
            folder_name = os.path.basename(folder_arg)
            folder_list = [folder_name]
            print(f"Processing specific folder: {folder_arg}")
        else:
            # It's just a folder name - use default files_dir
            folder_list = [folder_arg]
            print(f"Processing folder: {folder_arg} (in default files directory)")
            
    elif args.parent_folder:
        if not os.path.exists(args.parent_folder):
            print(f"ERROR: Parent folder not found: {args.parent_folder}")
            return
        folder_list = [f for f in os.listdir(args.parent_folder) 
                      if os.path.isdir(os.path.join(args.parent_folder, f))]
        files_dir = args.parent_folder
    else:
        if not os.path.exists(files_dir):
            print(f"ERROR: Default files directory not found: {files_dir}")
            return
        folder_list = [f for f in os.listdir(files_dir) 
                      if os.path.isdir(os.path.join(files_dir, f))]
    
    print(f"\nProcessing {len(folder_list)} folder(s)")
    
    # Store run configuration
    run_config = {
        'engine_name': args.engine,
        'model_type': args.model,
        'precision': precision,
        'batch_size': args.batch_size
    }
    
    # Process folders
    results = []
    start_time = time.time()
    
    for folder_name in folder_list:
        try:
            result = process_folder(
                engine,
                folder_name,
                files_dir,
                model_type=args.model,
                save_png=args.save_png,
                output_name=args.output_name
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR processing {folder_name}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Print summary
    if len(results) > 1:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        avg_bsq = np.mean([r['bsq_load_time'] for r in results])
        avg_inference = np.mean([r['inference_time'] for r in results])
        avg_write = np.mean([r['write_time'] for r in results])
        avg_total = np.mean([r['total_time'] for r in results])
        
        print(f"Folders processed: {len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"\nAverage times per folder:")
        print(f"  BSQ load: {avg_bsq:.2f}s")
        print(f"  Inference: {avg_inference:.2f}s")
        print(f"  Write: {avg_write:.2f}s")
        print(f"  Total: {avg_total:.2f}s")
        
        # Log to CSV
        log_results_to_csv(results, run_config)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == '__main__':
    main()
