import os
import pyvips
import torch
import onnxruntime as ort
import numpy as np
from torch.utils.data import Dataset
import spectral
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import argparse
import onnxruntime as ort

print("Available providers:", ort.get_available_providers())


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run ONNX inference on hyperspectral images')
    parser.add_argument('--folder', type=str, default=None, 
                       help='Specific folder name to process (optional)')
    parser.add_argument('--parent_folder', type=str, default=None,
                       help='Parent directory containing multiple folders to process in batch (optional)')
    parser.add_argument('--model', type=int, default=0, choices=[0, 1],
                       help='Model type: 0 for 10-channel (default), 1 for 13-channel with VIS')
    return parser.parse_args()

class SingleFolderDataset(Dataset):
    def __init__(self, root, folder_name, model_type=0):
        self.parent = root
        self.folder_name = folder_name
        self.model_type = model_type
        self.is_subfolder_structure = self._check_subfolder_structure()
        
    def _check_subfolder_structure(self):
        """Check if this is a subfolder structure (IR/VIS subfolders) or direct folder"""
        folder_path = os.path.join(self.parent, self.folder_name)
        
        # Check for IR subfolder
        ir_path = os.path.join(folder_path, 'IR')
        if os.path.exists(ir_path) and os.path.isdir(ir_path):
            return True
        return False

    def __getitem__(self, index):
        start = time.time()
        
        if self.is_subfolder_structure:
            # Handle subfolder structure (IR/VIS subfolders)
            ir_folder_path = os.path.join(self.parent, self.folder_name, 'IR')
            if not os.path.exists(ir_folder_path):
                raise FileNotFoundError(f"IR subfolder not found: {ir_folder_path}")
            ir_files = os.listdir(ir_folder_path)
            
            # Find HDR file in IR subfolder
            hdr_files = [f for f in ir_files if f.endswith('.hdr')]
            if not hdr_files:
                raise FileNotFoundError(f"No HDR files found in IR subfolder: {ir_folder_path}")
            hdr_file = hdr_files[0]
            
            # Get base name and find data file
            base_name = hdr_file.replace('.hdr', '')
            ir_data_file = self._find_data_file(base_name, ir_files)
            
            if ir_data_file is None:
                raise FileNotFoundError(f"No supported data file found for {base_name} in {ir_folder_path}")
            
            # Load IR data
            image = spectral.envi.open(os.path.join(ir_folder_path, hdr_file),
                                    os.path.join(ir_folder_path, ir_data_file))
            img = np.array(image.load())
            img = np.transpose(img, (2, 0, 1))
            img = img[:,:,:].copy()
            img[img>5] = 5
            img[img<-5] = -5
            img = img/1
            
            # Handle VIS data for subfolder structure
            vis_data = None
            vis_pixel_size = None
            
            if self.model_type == 1:
                vis_folder_path = os.path.join(self.parent, self.folder_name, 'VIS')
                if os.path.exists(vis_folder_path) and os.path.isdir(vis_folder_path):
                    vis_files = os.listdir(vis_folder_path)
                    
                    # Find any HDR file in VIS subfolder
                    vis_hdr_files = [f for f in vis_files if f.endswith('.hdr')]
                    if not vis_hdr_files:
                        raise FileNotFoundError(f"No HDR files found in VIS subfolder: {vis_folder_path}")
                    
                    vis_hdr_file = vis_hdr_files[0]
                    vis_base_name = vis_hdr_file.replace('.hdr', '')
                    vis_hdr_path = os.path.join(vis_folder_path, vis_hdr_file)
                    
                    vis_data_file = self._find_data_file(vis_base_name, vis_files)
                    
                    if vis_data_file is not None:
                        vis_data_path = os.path.join(vis_folder_path, vis_data_file)
                        vis_image = spectral.envi.open(vis_hdr_path, vis_data_path)
                        vis_data = np.array(vis_image.load())
                        vis_data = vis_data / 256.0
                        vis_data = vis_data.astype(np.float32)
                        vis_data = np.transpose(vis_data, (2, 0, 1))
                        
                        pixel_size_info = vis_image.metadata.get('pixel size')
                        if pixel_size_info:
                            if isinstance(pixel_size_info, (list, tuple)):
                                vis_pixel_size = float(pixel_size_info[0]) * 1e6
                            else:
                                vis_pixel_size = float(pixel_size_info) * 1e6
                        else:
                            vis_pixel_size = 2.0
                    else:
                        raise FileNotFoundError(f"No supported VIS data file found for {vis_base_name}")
                else:
                    raise FileNotFoundError(f"Required VIS subfolder not found: {vis_folder_path}")
        else:
            # Handle direct folder structure (original code)
            folder_path = os.path.join(self.parent, self.folder_name)
            files = os.listdir(folder_path)
            
            # Find HDR file
            hdr_files = [f for f in files if f.endswith('.hdr')]
            if not hdr_files:
                raise FileNotFoundError(f"No HDR files found in folder: {folder_path}")
            hdr_file = hdr_files[0]
            
            # Get base name and find data file
            base_name = hdr_file.replace('.hdr', '')
            ir_data_file = self._find_data_file(base_name, files)
            
            if ir_data_file is None:
                raise FileNotFoundError(f"No supported data file found for {base_name} in {folder_path}")
            
            # Load IR data
            image = spectral.envi.open(os.path.join(folder_path, hdr_file),
                                    os.path.join(folder_path, ir_data_file))
            img = np.array(image.load())
            img = np.transpose(img, (2, 0, 1))
            img = img[:,:,:].copy()
            img[img>5] = 5
            img[img<-5] = -5
            img = img/1

            # Handle VIS data for direct folder structure
            vis_data = None
            vis_pixel_size = None
            
            if self.model_type == 1:
                vis_folder_name = self.folder_name + '_VIS'
                vis_folder_path = os.path.join(self.parent, vis_folder_name)
                
                if os.path.exists(vis_folder_path):
                    vis_files = os.listdir(vis_folder_path)
                    
                    # Find any HDR file in VIS folder
                    vis_hdr_files = [f for f in vis_files if f.endswith('.hdr')]
                    if not vis_hdr_files:
                        raise FileNotFoundError(f"No HDR files found in VIS folder: {vis_folder_path}")
                    
                    vis_hdr_file = vis_hdr_files[0]
                    vis_base_name = vis_hdr_file.replace('.hdr', '')
                    vis_hdr_path = os.path.join(vis_folder_path, vis_hdr_file)
                    
                    vis_data_file = self._find_data_file(vis_base_name, vis_files)
                    
                    if vis_data_file is not None:
                        vis_data_path = os.path.join(vis_folder_path, vis_data_file)
                        vis_image = spectral.envi.open(vis_hdr_path, vis_data_path)
                        vis_data = np.array(vis_image.load())
                        vis_data = vis_data / 256.0
                        vis_data = vis_data.astype(np.float32)
                        vis_data = np.transpose(vis_data, (2, 0, 1))
                        
                        pixel_size_info = vis_image.metadata.get('pixel size')
                        if pixel_size_info:
                            if isinstance(pixel_size_info, (list, tuple)):
                                vis_pixel_size = float(pixel_size_info[0]) * 1e6
                            else:
                                vis_pixel_size = float(pixel_size_info) * 1e6
                        else:
                            vis_pixel_size = 2.0
                    else:
                        raise FileNotFoundError(f"No supported VIS data file found for {vis_base_name}")
                else:
                    raise FileNotFoundError(f"Required VIS folder not found: {vis_folder_path}")

        img = torch.from_numpy(img).float()
        bsq_load_time = time.time() - start

        return {
            'ir': img[:,:,:],
            'vis_raw': vis_data if vis_data is not None else None,
            'vis_pixel_size': vis_pixel_size,
            'name': self.folder_name,
            'bsq_load_time': torch.tensor(bsq_load_time),
            'is_subfolder_structure': self.is_subfolder_structure
        }
    
    def _find_data_file(self, base_name, folder_files):
        """Find data file with supported extensions: bsq, bip, bil, dat, img"""
        supported_extensions = ['.bsq', '.bip', '.bil', '.dat', '.img']
        for ext in supported_extensions:
            data_file = base_name + ext
            if data_file in folder_files:
                return data_file
        return None
    
    def __len__(self):
        return 1

# Custom collate function to preserve numpy arrays
def custom_collate_fn(batch):
    """Custom collate function that preserves numpy arrays for vis_raw"""
    collated = {}
    for key in batch[0].keys():
        if key in ['vis_raw', 'vis_pixel_size', 'name', 'is_subfolder_structure']:
            # Keep as list of values
            collated[key] = [item[key] for item in batch]
        else:
            # Use default collation for other keys (ir, bsq_load_time)
            collated[key] = torch.stack([item[key] for item in batch])
    return collated

def validate_folder_structure(files_dir, folder_list, model_type):
    """Validate folder structure and return valid folders"""
    valid_folders = []
    
    for folder_name in folder_list:
        folder_path = os.path.join(files_dir, folder_name)
        
        # Check for HDR files
        files_in_folder = os.listdir(folder_path)
        if not any(f.endswith('.hdr') for f in files_in_folder):
            print(f"Warning: No .hdr files found in {folder_name}")
            continue
            
        # For model_type 1, check VIS subfolder
        if model_type == 1:
            vis_folder_name = folder_name + '_VIS'
            vis_folder_path = os.path.join(files_dir, vis_folder_name)
            if not (os.path.exists(vis_folder_path) and os.path.isdir(vis_folder_path)):
                print(f"Warning: Skipping {folder_name} - no VIS subfolder found")
                continue
                
        valid_folders.append(folder_name)
    
    return valid_folders

def main(opt, target_folder_path=None, parent_folder_path=None, model_type=0): 
    # Use PARENT_DIR for default paths
    files_dir = os.path.join(PARENT_DIR, 'files')
    model_path = os.path.join(PARENT_DIR, 'saved_models', 
                             '15_net_G.onnx' if model_type == 1 else '20_net_G.onnx')

    # Handle target folder path (single folder processing)
    if target_folder_path:
        target_folder_path = target_folder_path.strip('"').strip("'")
        if os.path.exists(target_folder_path) and os.path.isdir(target_folder_path):
            files_dir = os.path.dirname(target_folder_path)
            folder_name = os.path.basename(target_folder_path)
            folder_list = [folder_name]
            print(f"Processing specific folder: {target_folder_path}")
        else:
            print(f"Error: Folder path does not exist: {target_folder_path}")
            return [], 0, 0
    # Handle parent folder path (batch processing from custom parent directory)
    elif parent_folder_path:
        parent_folder_path = parent_folder_path.strip('"').strip("'")
        if os.path.exists(parent_folder_path) and os.path.isdir(parent_folder_path):
            files_dir = parent_folder_path
            folder_list = [
                dir_name for dir_name in os.listdir(files_dir)
                if os.path.isdir(os.path.join(files_dir, dir_name)) and not dir_name.endswith('_VIS')
            ]
            print(f"Processing all folders in parent directory: {parent_folder_path}")
            
            if not folder_list:
                print("Error: No folders found in parent directory")
                return [], 0, 0
        else:
            print(f"Error: Parent folder path does not exist: {parent_folder_path}")
            return [], 0, 0
    else:
        # Get all folders from default files directory
        folder_list = [
            dir_name for dir_name in os.listdir(files_dir)
            if os.path.isdir(os.path.join(files_dir, dir_name)) and not dir_name.endswith('_VIS')
        ]
        
        if not folder_list:
            print("Error: No folders found")
            return [], 0, 0

    # Initialize ONNX Runtime session
    if not os.path.exists(model_path):
        print(f"Error: ONNX model not found at {model_path}")
        return [], 0, 0

    # Set up ONNX Runtime providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Get model input/output info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Loaded ONNX model: {model_path}")
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")

    BATCH_SIZE = opt['batch_size']
    
    print("BATCH_SIZE:", opt['batch_size'])
    print("NUM_WORKERS:", opt['n_cpu']) 
    print("Using ONNX Runtime with providers:", session.get_providers())
    
    results = []
    bsq_load_times = []
    inference_times = []
    folder_times = []
    
    # Process each folder individually
    for folder_idx, folder_name in enumerate(folder_list):
        print(f"\nProcessing folder {folder_idx + 1}/{len(folder_list)}: {folder_name}")
        
        try:
            # Create single-folder dataset with model_type - let it handle validation
            single_folder_dataset = SingleFolderDataset(files_dir, folder_name, model_type)
            dataloader = DataLoader(single_folder_dataset, batch_size=opt['batch_size'], 
                                  shuffle=False, num_workers=opt['n_cpu'], collate_fn=custom_collate_fn)
            
            folder_start_time = time.time()
            
            for batch in dataloader:
                bsq_load_times.append(batch['bsq_load_time'].item())
                inference_start = time.time()
                
                # Check if this is a single folder processing and we have an error
                if target_folder_path and batch['vis_raw'][0] is None and model_type == 1:
                    print(f"Error: Required VIS data not found for model type 1")
                    return [], 0, 0
                
                # Process VIS data
                vis_raw = batch['vis_raw'][0]
                vis_pixel_size = batch['vis_pixel_size'][0]
                
                if vis_raw is not None and vis_pixel_size is not None:
                    # Convert to torch tensor and scale
                    img_V = torch.from_numpy(vis_raw).float()
                    
                    target_pixel_size = 0.69
                    scale_factor = vis_pixel_size / target_pixel_size
                    
                    _, original_h, original_w = img_V.shape
                    new_h = int(original_h * scale_factor)
                    new_w = int(original_w * scale_factor)
                    
                    img_V = torch.nn.functional.interpolate(
                        img_V.unsqueeze(0),
                        size=(new_h, new_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    
                    img_V = torch.clamp(img_V, 0, 1)
                else:
                    # Create dummy VIS data
                    ir = batch['ir']
                    # IR shape is [batch_size, bands, height, width]
                    batch_size, bands, ir_h, ir_w = ir.shape
                    
                    # IR has pixel size of 2.0, VIS target is 0.69
                    # VIS should be higher resolution (larger dimensions)
                    ir_pixel_size = 2.0
                    vis_target_pixel_size = 0.69
                    scale_factor = ir_pixel_size / vis_target_pixel_size  # ~2.9
                    
                    # Calculate VIS dimensions (should be larger than IR)
                    vis_h = int(ir_h * scale_factor)
                    vis_w = int(ir_w * scale_factor)
                    
                    # Create dummy VIS data with 3 channels (RGB) and scaled dimensions
                    img_V = torch.zeros((3, vis_h, vis_w))
                
                # Normalize data
                if model_type == 1:
                    ir = batch['ir']
                    vis = img_V.unsqueeze(0)
                else:
                    ir = (batch['ir'] - 0.5) / 0.5
                    vis = (img_V.unsqueeze(0) - 0.5) / 0.5
                
                name = batch['name'][0]
                
                # Processing parameters
                fact = 1024 if model_type == 1 else 1472
                channels = 13 if model_type == 1 else 10
                scaling_factor = 0.69/2
                
                _, bands, _, _ = ir.shape
                b, _, _, _ = vis.shape
                _, _, r, c = vis.shape
                rn = np.ceil(r/fact)
                cn = np.ceil(c/fact)
                
                # Pre-allocate arrays for ONNX
                DATA = np.zeros((BATCH_SIZE, channels, fact, fact), dtype=np.float32)
                fhr = np.zeros((b, 3, int(rn)*fact+1, int(cn)*fact+1), dtype=np.float32)

                for kk in range(b):
                    patch_count = 0
                    patches = []
                    I = range(0, int(rn)*fact+1, fact)
                    J = range(0, int(cn)*fact+1, fact)
                    
                    for ii in range(len(I)-1):
                        for jj in range(len(J) - 1):
                            xx = ir[:, :, int(I[ii] * scaling_factor):int(I[ii+1] * scaling_factor), 
                                   int(J[jj] * scaling_factor):int(J[jj+1] * scaling_factor)]
                            
                            pad_rows = int(fact * scaling_factor) - xx.shape[2] if xx.shape[2] < int(fact * scaling_factor) else 0
                            pad_cols = int(fact * scaling_factor) - xx.shape[3] if xx.shape[3] < int(fact * scaling_factor) else 0
                            xx = F.pad(xx, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                            xx = F.interpolate(xx, size=(fact, fact), mode='bicubic', align_corners=True)
                            
                            vv = vis[:, :, int(I[ii]):int(I[ii+1]), int(J[jj]):int(J[jj+1])]
                            pad_rows = fact - vv.shape[2] if vv.shape[2] < fact else 0
                            pad_cols = fact - vv.shape[3] if vv.shape[3] < fact else 0
                            vv = F.pad(vv, (0, pad_cols, 0, pad_rows), mode='constant', value=0)

                            if model_type == 1:
                                DATA[patch_count, :bands, :, :] = xx.numpy()
                                DATA[patch_count, bands:, :, :] = vv.numpy()
                            else:
                                DATA[patch_count, :bands, :, :] = xx.numpy()
                                
                            patches.append((ii, jj))
                            patch_count += 1

                            if patch_count == BATCH_SIZE:
                                # Run ONNX inference
                                output = session.run([output_name], {input_name: DATA[:patch_count]})[0]
                                output = output * 0.5 + 0.5

                                for idx, (pi, pj) in enumerate(patches):
                                    fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]

                                patch_count = 0
                                patches = []

                    if patch_count > 0:
                        # Run ONNX inference for remaining patches
                        output = session.run([output_name], {input_name: DATA[:patch_count]})[0]
                        output = output * 0.5 + 0.5
                        
                        for idx, (pi, pj) in enumerate(patches):
                            fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]
                            
                fhr = fhr[:, :, :r, :c]
                
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Process and save
                write_start = time.time()
                img_array = (np.transpose(fhr[0], (1, 2, 0)) * 255).astype(np.uint8)
                create_szi(img_array, name, model_type)
                update_json(name, int(r), int(c), model_type)
                write_time = time.time() - write_start
                
                folder_total_time = time.time() - folder_start_time
                folder_times.append(folder_total_time)
                
                print(f"  BSQ Load Time: {np.round(batch['bsq_load_time'].item(), 2)} seconds")
                print(f"  Inference Time: {np.round(inference_time, 2)} seconds") 
                print(f"  Write Time: {np.round(write_time, 2)} seconds")
                print(f"  Total Folder Time: {np.round(folder_total_time, 2)} seconds")
                
                results.append({
                    'name': name,
                    'bsq_load_time': batch['bsq_load_time'].item(),
                    'inference_time': inference_time,
                    'write_time': write_time,
                    'total_time': folder_total_time
                })
                
        except FileNotFoundError as e:
            print(f"Skipping folder {folder_name}: {str(e)}")
            # For single folder processing, stop execution
            if target_folder_path:
                return [], 0, 0
            # For batch processing, continue to next folder
            continue
    
    if len(results) > 1:
        avg_bsq_load_time = np.mean(bsq_load_times)
        avg_inference_time = np.mean(inference_times)
        avg_folder_time = np.mean(folder_times)
        
        print(f"\n=== SUMMARY ===")
        print(f"Processed {len(results)} folders")
        print(f"Average BSQ Load Time: {np.round(avg_bsq_load_time, 2)} seconds")
        print(f"Average Inference Time: {np.round(avg_inference_time, 2)} seconds")
        print(f"Average Folder Processing Time: {np.round(avg_folder_time, 2)} seconds")
        
    return results

def create_szi(img_array, filename, model_type=0):
    """Create .szi file and thumbnail from numpy array using pyvips"""

    # Choose subdirectory based on model_type
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
        print(f"Deleted existing szi: {filename}.szi")

    vi.dzsave(szi_path,
              overlap=0,
              container='zip',
              suffix='.jpg',
              compression=0,
              tile_size=1024)

    # Save thumbnail
    thumbnail = vi.thumbnail_image(1024)
    thumbnail_path = os.path.join(viewer_thumbnail_dir, f"{filename}.jpg")
    thumbnail.jpegsave(thumbnail_path, Q=90)

def update_json(filename, height, width, model_type):
    """Update JSON list with image dimensions and model_type as unique key"""
    json_path = os.path.join(PARENT_DIR, '..', 'viewer', 'static', 'available.json')

    # Load existing list
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Format resolution string
    resolution_str = f"{width}x{height}"

    # Look for existing entry with same name and model_type
    for entry in data:
        if entry.get('name') == filename and entry.get('model_type') == model_type:
            entry['resolution'] = resolution_str
            break
    else:
        # Not found, add new entry
        data.append({'name': filename, 'resolution': resolution_str, 'model_type': model_type})

    # Save updated data
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    start0 = time.time()
    args = parse_arguments()
    opt = {
        'batch_size': 4,
        'n_cpu': 0
    }
    results = main(opt, target_folder_path=args.folder, parent_folder_path=args.parent_folder, model_type=args.model)