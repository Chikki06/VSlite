import os
import pyvips
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import numpy as np
from torch.utils.data import Dataset, TensorDataset
import spectral
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import argparse
from PIL import Image
import csv
import pandas as pd
from datetime import datetime
import gc
import sys

# Add viewer/misc path for database helper
VIEWER_MISC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'viewer', 'misc'))
if VIEWER_MISC_PATH not in sys.path:
    sys.path.insert(0, VIEWER_MISC_PATH)

try:
    from available_images_db_helper import add_available_image
    DB_HELPER_AVAILABLE = True
    print("Database helper loaded successfully - will add images directly to database")
except ImportError as e:
    print(f"Warning: Could not import database helper: {e}")
    print("Falling back to JSON file method")
    DB_HELPER_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_BASE_DIR = os.getenv('SOCKETINFERENCE_FILES_BASE_DIR', '/mnt/files')
DEFAULT_PROJECT_NAME = os.getenv('SOCKETINFERENCE_DEFAULT_PROJECT', 'main')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on hyperspectral images')
    parser.add_argument('--folder', type=str, default=None, 
                       help='Specific folder name to process (optional)')
    parser.add_argument('--parent_folder', type=str, default=None,
                       help='Parent directory containing multiple folders to process in batch (optional)')
    parser.add_argument('--model', type=int, default=0, choices=[0, 1],
                       help='Model type: 0 for 10-channel (default), 1 for 13-channel with VIS')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Custom model filename (located in saved_models/). If specified with --model 0 or 1, this will be used instead of the default 20_net_G.pth or 15_net_G.pth')
    parser.add_argument('--image_type', type=str, default='IR',
                       help="Image type string (e.g. 'IR' or 'IR+VIS') - kept for backward compatibility with older callers")
    # Add TensorRT arguments
    parser.add_argument('--use_tensorrt', action='store_true', default=False,
                       help='Use TensorRT optimization for inference')
    parser.add_argument('--tensorrt_precision', type=str, default=None, choices=['fp32', 'fp16'],
                   help='TensorRT precision mode (default: None - uses TorchScript without precision changes)')
    parser.add_argument('--tensorrt_workspace', type=int, default=1024,
                       help='TensorRT workspace size in MB (default: 1024)')
    # Add TIFF saving and filename arguments
    parser.add_argument('--save_png', action='store_true', default=False,
                       help='Save output as PNG instead of DZI')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Custom output filename (without extension)')
    parser.add_argument('--optim', action='store_true', default=False,
                       help='Enable optimized parallel mode: move per-patch preprocessing to GPU '
                            'to reduce CPU contention and PCIe traffic when two GPU jobs run concurrently')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT_NAME,
                       help='Project name for output under <files_base_dir>/<project>/...')
    parser.add_argument('--vs_modality', type=str, default=None,
                       help='DZI/thumbnail subfolder and DB image_type for VS output (e.g. VS_G10_10ch). '
                            'Default: IR(VS) for --model 0, IR+VIS(VS) for --model 1.')
    return parser.parse_args()


class SingleFolderDataset(Dataset):
    def __init__(self, root, folder_name, model_type=0):
        self.parent = root
        self.folder_name = folder_name
        self.model_type = model_type
        self.folder_category = self._get_folder_category()
        self.is_subfolder_structure = self.folder_category in ['ir_subfolder_tiff', 'ir_vis_subfolders', 'processed_siblings']
        
        # Validate that required folders exist during initialization
        self._validate_required_folders()
        
    def _check_subfolder_structure(self):
        """Check if this is a subfolder structure (IR/VIS subfolders) or direct folder"""
        folder_path = os.path.join(self.parent, self.folder_name)
        
        # Check for IR subfolder
        ir_path = os.path.join(folder_path, 'IR')
        if os.path.exists(ir_path) and os.path.isdir(ir_path):
            return True
        return False
    
    def _get_folder_category(self):
        """
        Determine which category this folder belongs to:
        1. Direct folder with sibling _VIS folder (e.g., folder/ and folder_VIS/)
        2. Parent contains IR/ subfolder with ../VIS/ having TIFF
        3. Parent folder with IR/ and VIS/ subfolders (NEW)
        4. Processed structure: /Processed/IR/sample and /Processed/VIS/sample as siblings
        
        Returns: 'direct', 'ir_subfolder_tiff', 'ir_vis_subfolders', or 'processed_siblings'
        """
        folder_path = os.path.join(self.parent, self.folder_name)
        
        # Check for Category 3: IR/ and VIS/ subfolders within the same parent
        ir_subfolder = os.path.join(folder_path, 'IR')
        vis_subfolder = os.path.join(folder_path, 'VIS')
        if (os.path.exists(ir_subfolder) and os.path.isdir(ir_subfolder) and
            os.path.exists(vis_subfolder) and os.path.isdir(vis_subfolder)):
            return 'ir_vis_subfolders'
        
        # Check for Category 4: Processed siblings
        # e.g., parent is /mnt/archive/Processed/IR, check if /mnt/archive/Processed/VIS/sample exists
        parent_basename = os.path.basename(self.parent).upper()
        if parent_basename == 'IR':
            vis_sibling_dir = os.path.join(os.path.dirname(self.parent), 'VIS')
            vis_sibling_folder = os.path.join(vis_sibling_dir, self.folder_name)
            if os.path.exists(vis_sibling_folder) and os.path.isdir(vis_sibling_folder):
                return 'processed_siblings'
            # VIS data may be a TIFF in the sibling VIS/ directory
            vis_tiff_path = os.path.join(vis_sibling_dir, f"{self.folder_name}_VIS_REG.tif")
            if os.path.exists(vis_tiff_path):
                return 'processed_siblings_tiff'
        
        # Check for Category 2: IR/ subfolder exists (VIS as TIFF)
        if os.path.exists(ir_subfolder) and os.path.isdir(ir_subfolder):
            return 'ir_subfolder_tiff'
        
        # Category 1: Direct folder (default)
        return 'direct'

    def _validate_required_folders(self):
        """Validate that all required folders exist and contain valid data files"""
        folder_path = os.path.join(self.parent, self.folder_name)
        
        # Validate IR folder based on category
        if self.folder_category in ('direct', 'processed_siblings', 'processed_siblings_tiff'):
            # IR data is directly in the folder
            ir_folder_path = os.path.join(self.parent, self.folder_name)
        else:
            # Category 2 & 3: IR subfolder
            ir_folder_path = os.path.join(folder_path, 'IR')
        
        if not self._validate_folder_contents(ir_folder_path):
            raise FileNotFoundError(f"Invalid IR folder or missing HDR/data files: {ir_folder_path}")
        
        # Validate VIS data if model_type requires it
        if self.model_type == 1:
            if self.folder_category == 'direct':
                # Category 1: Check for sibling _VIS folder
                vis_folder_path = os.path.join(self.parent, self.folder_name + '_VIS')
                vis_folder_exists = self._validate_folder_contents(vis_folder_path)
                
                if not vis_folder_exists:
                    raise FileNotFoundError(f"VIS folder not found: {vis_folder_path}")
                    
            elif self.folder_category == 'ir_vis_subfolders':
                # Category 3: Check for VIS subfolder
                vis_folder_path = os.path.join(folder_path, 'VIS')
                if not self._validate_folder_contents(vis_folder_path):
                    raise FileNotFoundError(f"Invalid VIS subfolder or missing HDR/data files: {vis_folder_path}")
            
            elif self.folder_category == 'processed_siblings':
                # Category 4: Check for VIS in sibling directory
                vis_sibling_dir = os.path.join(os.path.dirname(self.parent), 'VIS')
                vis_folder_path = os.path.join(vis_sibling_dir, self.folder_name)
                if not self._validate_folder_contents(vis_folder_path):
                    raise FileNotFoundError(f"VIS sibling folder not found: {vis_folder_path}")
                    
            elif self.folder_category == 'processed_siblings_tiff':
                # Category 4b: VIS as TIFF in sibling Processed/VIS/ directory
                tiff_path = self._get_vis_tiff_path()
                if not os.path.exists(tiff_path):
                    raise FileNotFoundError(f"VIS TIFF file not found: {tiff_path}")

            elif self.folder_category == 'ir_subfolder_tiff':
                # Category 2: Check for TIFF file
                tiff_path = self._get_vis_tiff_path()
                if not os.path.exists(tiff_path):
                    raise FileNotFoundError(f"VIS TIFF file not found: {tiff_path}")

    def _validate_folder_contents(self, folder_path):
        """Check if folder exists and contains valid HDR + data file pairs"""
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return False
        
        files = os.listdir(folder_path)
        
        # Find HDR files
        hdr_files = [f for f in files if f.endswith('.hdr')]
        if not hdr_files:
            return False
        
        # Check if at least one HDR file has a corresponding data file
        for hdr_file in hdr_files:
            base_name = hdr_file.replace('.hdr', '')
            data_file = self._find_data_file(base_name, files)
            if data_file is not None:
                return True
        
        return False

    def _get_vis_tiff_path(self):
        """Get the path to the VIS TIFF file: ../VIS/{folder_name}_VIS_REG.tif"""
        # Replace 'IR' with 'VIS' in the parent path
        vis_dir = self.parent.replace('/IR', '/VIS')
        tiff_filename = f"{self.folder_name}_VIS_REG.tif"
        return os.path.join(vis_dir, tiff_filename)

    def __getitem__(self, index):
        start = time.time()
        
        folder_path = os.path.join(self.parent, self.folder_name)
        
        # Determine IR folder path based on category
        if self.folder_category in ('direct', 'processed_siblings', 'processed_siblings_tiff'):
            ir_folder_path = os.path.join(self.parent, self.folder_name)
        else:
            # Both category 2 and 3 have IR subfolder
            ir_folder_path = os.path.join(folder_path, 'IR')
        
        # Load IR data
        img = self._load_ir_data(ir_folder_path)
        
        # Handle VIS data if needed
        vis_data = None
        vis_pixel_size = None
        
        if self.model_type == 1:
            if self.folder_category == 'direct':
                # Category 1: Sibling _VIS folder
                vis_folder_path = os.path.join(self.parent, self.folder_name + '_VIS')
                vis_data, vis_pixel_size = self._load_vis_data(vis_folder_path)
                
            elif self.folder_category == 'ir_vis_subfolders':
                # Category 3: VIS subfolder
                vis_folder_path = os.path.join(folder_path, 'VIS')
                vis_data, vis_pixel_size = self._load_vis_data(vis_folder_path)
            
            elif self.folder_category == 'processed_siblings':
                # Category 4: VIS in sibling directory (e.g., /Processed/VIS/sample_name/)
                vis_sibling_dir = os.path.join(os.path.dirname(self.parent), 'VIS')
                vis_folder_path = os.path.join(vis_sibling_dir, self.folder_name)
                vis_data, vis_pixel_size = self._load_vis_data(vis_folder_path)

            elif self.folder_category == 'processed_siblings_tiff':
                # Category 4b: VIS as TIFF in sibling /Processed/VIS/ directory
                tiff_path = self._get_vis_tiff_path()
                vis_data, vis_pixel_size = self._load_vis_tiff(tiff_path)

            elif self.folder_category == 'ir_subfolder_tiff':
                # Category 2: TIFF file
                tiff_path = self._get_vis_tiff_path()
                vis_data, vis_pixel_size = self._load_vis_tiff(tiff_path)
        
        bsq_load_time = time.time() - start
        
        return {
            'ir': img[:,:,:],
            'vis_raw': vis_data,
            'vis_pixel_size': vis_pixel_size,
            'name': self.folder_name,
            'bsq_load_time': torch.tensor(bsq_load_time),
            'is_subfolder_structure': self.is_subfolder_structure,
            'folder_category': self.folder_category
        }

    def _load_ir_data(self, ir_folder_path):
        """Load IR data from the specified folder path."""
        files = os.listdir(ir_folder_path)
        
        # Find HDR file
        hdr_files = [f for f in files if f.endswith('.hdr')]
        hdr_file = hdr_files[0]  # We know it exists from validation
        base_name = hdr_file.replace('.hdr', '')
        ir_data_file = self._find_data_file(base_name, files)
        
        # Load and process IR data
        image = spectral.envi.open(os.path.join(ir_folder_path, hdr_file),
                                os.path.join(ir_folder_path, ir_data_file))
        img = np.array(image.load())
        img = np.transpose(img, (2, 0, 1))
        img = img[:,:,:].copy()
        img[img > 5] = 5
        img[img<0] = 0
        img = img / 1
        img = torch.from_numpy(img).float()
        return img

    def _get_vis_folder_path(self):
        """Get VIS folder path - always folder_name + '_VIS'"""
        return os.path.join(self.parent, self.folder_name + '_VIS')

    def _load_vis_data(self, vis_folder_path):
        """Load VIS data from the specified folder path."""
        vis_files = os.listdir(vis_folder_path)
        
        # Find HDR file
        vis_hdr_files = [f for f in vis_files if f.endswith('.hdr')]
        vis_hdr_file = vis_hdr_files[0]  # We know it exists from validation
        vis_base_name = vis_hdr_file.replace('.hdr', '')
        vis_hdr_path = os.path.join(vis_folder_path, vis_hdr_file)
        
        vis_data_file = self._find_data_file(vis_base_name, vis_files)
        
        # Load and process VIS data
        vis_data_path = os.path.join(vis_folder_path, vis_data_file)
        vis_image = spectral.envi.open(vis_hdr_path, vis_data_path)
        vis_data = np.array(vis_image.load())
        vis_data = vis_data / 256.0
        vis_data = vis_data.astype(np.float32)
        vis_data = np.transpose(vis_data, (2, 0, 1))
        vis_data = torch.from_numpy(vis_data).float()  

        # Extract pixel size information
        pixel_size_info = vis_image.metadata.get('pixel size')
        if pixel_size_info:
            if isinstance(pixel_size_info, (list, tuple)):
                vis_pixel_size = float(pixel_size_info[0]) * 1e6
            else:
                vis_pixel_size = float(pixel_size_info) * 1e6
        else:
            vis_pixel_size = 2.0
        
        return vis_data, vis_pixel_size
    
    def _load_vis_tiff(self, tiff_path):
        """Load VIS data from TIFF file."""
        try:
            
            # Load TIFF image
            with Image.open(tiff_path) as img:
                vis_data = np.array(img)
            
            # Convert to float32 and normalize
            vis_data = vis_data.astype(np.float32)
            vis_data = vis_data / 256.0
            
            # Handle different image formats
            if len(vis_data.shape) == 2:  # Grayscale
                vis_data = vis_data[np.newaxis, :, :]  # Add channel dimension
            elif len(vis_data.shape) == 3:  # Color image
                vis_data = np.transpose(vis_data, (2, 0, 1))  # HWC to CHW
            
            vis_data = torch.from_numpy(vis_data).float()
            
            # Default pixel size for TIFF (you may want to read this from metadata if available)
            vis_pixel_size = 0.69
            
            return vis_data, vis_pixel_size
            
        except Exception as e:
            raise FileNotFoundError(f"Failed to load TIFF file {tiff_path}: {str(e)}")
    
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
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    Flatten(),
                    nn.Linear(input_channels, input_channels//reduction_ratio),
                    nn.ReLU(),
                    nn.Linear(input_channels//reduction_ratio, input_channels))

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
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
        self.encoder = IntermediateLayerGetter(backbone, return_layers={"conv1": 'feat', "layer1": 'feat0', "layer2": 'feat1', "layer3": 'feat2', "layer4": 'feat3'})

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

# Custom collate function to preserve numpy arrays
def custom_collate_fn(batch):
    """Custom collate function that preserves numpy arrays for vis_raw"""
    collated = {}
    for key in batch[0].keys():
        if key in ['vis_raw', 'vis_pixel_size', 'name', 'is_subfolder_structure', 'folder_category']:
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

def get_run_identifier(use_tensorrt, tensorrt_precision, model_type, batch_size):
    """Generate unique run identifier"""
    if use_tensorrt:
        if tensorrt_precision:
            return f"TensorRT_{tensorrt_precision}_model{model_type}_bs{batch_size}"
        else:
            return f"TorchScript_model{model_type}_bs{batch_size}"
    else:
        return f"PyTorch_model{model_type}_bs{batch_size}"

def get_next_run_index():
    """Get the next run index from the summary CSV"""
    summary_csv_path = os.path.join(PARENT_DIR, 'run_summary.csv')
    
    if os.path.exists(summary_csv_path):
        try:
            df = pd.read_csv(summary_csv_path)
            if not df.empty and 'run_index' in df.columns:
                return df['run_index'].max() + 1
        except Exception as e:
            print(f"Error reading run index: {e}")
    
    return 1

def format_run_column_name(run_index, model_type, precision):
    """Format run column name as: <index>_model<type>_<precision>"""
    if precision:
        return f"{run_index}_model{model_type}_{precision}"
    else:
        return f"{run_index}_model{model_type}"

def log_run_to_csv(results, run_config, summary_stats):
    """Log run results to CSV file"""
    csv_path = os.path.join(PARENT_DIR, 'runtime_logs.csv')
    
    # Get next run index
    run_index = get_next_run_index()
    
    # Create run column name
    run_column = format_run_column_name(
        run_index,
        run_config['model_type'],
        run_config['tensorrt_precision']
    )
    
    # Prepare data for CSV
    folder_data = []
    for i, result in enumerate(results):
        folder_data.append({
            'sample_id': f"sample_{i+1}_{result['name']}",
            'folder_name': result['name'],
            run_column + '_bsq_load_time': result['bsq_load_time'],
            run_column + '_inference_time': result['inference_time'], 
            run_column + '_write_time': result['write_time'],
            run_column + '_total_time': result['total_time']
        })
    
    # Add summary row
    if summary_stats:
        folder_data.append({
            'sample_id': 'SUMMARY_AVERAGE',
            'folder_name': 'AVERAGE',
            run_column + '_bsq_load_time': summary_stats['avg_bsq_load_time'],
            run_column + '_inference_time': summary_stats['avg_inference_time'],
            run_column + '_write_time': summary_stats.get('avg_write_time', 0),
            run_column + '_total_time': summary_stats['avg_folder_time']
        })
        
        folder_data.append({
            'sample_id': 'SUMMARY_TOTAL_FOLDERS',
            'folder_name': 'COUNT',
            run_column + '_bsq_load_time': len(results),
            run_column + '_inference_time': len(results),
            run_column + '_write_time': len(results),
            run_column + '_total_time': len(results)
        })
    
    # Load existing CSV or create new DataFrame
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            df_existing = pd.DataFrame()
    else:
        df_existing = pd.DataFrame()
    
    # Create new DataFrame for this run
    df_new = pd.DataFrame(folder_data)
    
    # Merge with existing data
    if not df_existing.empty:
        # Merge on sample_id and folder_name
        df_merged = pd.merge(df_existing, df_new, on=['sample_id', 'folder_name'], how='outer')
    else:
        df_merged = df_new
    
    # Save to CSV
    df_merged.to_csv(csv_path, index=False)
    
    # Also create a run summary CSV
    summary_csv_path = os.path.join(PARENT_DIR, 'run_summary.csv')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_summary = {
        'run_index': run_index,
        'run_id': run_column,
        'timestamp': timestamp,
        'use_tensorrt': run_config['use_tensorrt'],
        'tensorrt_precision': run_config.get('tensorrt_precision', 'N/A'),
        'model_type': run_config['model_type'],
        'batch_size': run_config['batch_size'],
        'num_folders': len(results),
        'avg_bsq_load_time': summary_stats['avg_bsq_load_time'] if summary_stats else 0,
        'avg_inference_time': summary_stats['avg_inference_time'] if summary_stats else 0,
        'avg_write_time': summary_stats.get('avg_write_time', 0) if summary_stats else 0,
        'avg_total_time': summary_stats['avg_folder_time'] if summary_stats else 0,
        'parent_folder': run_config.get('parent_folder', ''),
        'target_folder': run_config.get('target_folder', '')
    }
    
    # Append to summary CSV
    summary_df = pd.DataFrame([run_summary])
    if os.path.exists(summary_csv_path):
        summary_df.to_csv(summary_csv_path, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"Runtime data logged to: {csv_path}")
    print(f"Run summary logged to: {summary_csv_path}")
    print(f"Run indexed as: {run_column} (Run #{run_index})")

def main(opt, target_folder_path=None, parent_folder_path=None, image_type='IR', model_type=None, model_name=None,
         use_tensorrt=False, tensorrt_precision=None, tensorrt_workspace=1024, save_png=False,
         output_name=None, optim=False, project_name=DEFAULT_PROJECT_NAME, vs_modality=None):
    """
    Main inference function
    
    Args:
        image_type (str): Type of image - 'IR', 'IR+VIS', 'VIS', 'HE', etc.
    """
    # Support both numeric `model_type` (from CLI --model) and string `image_type` ('IR'/'IR+VIS').
    # If `model_type` is provided use it; otherwise derive from `image_type`.
    if model_type is None:
        model_type = 0
        if image_type == 'IR+VIS':
            model_type = 1
    else:
        try:
            model_type = int(model_type)
        except Exception:
            model_type = 0
    
    # Conditionally import TensorRT utilities only when needed
    if use_tensorrt or tensorrt_precision:
        try:
            from tensorrt_utils import (
                convert_to_torchscript, 
                convert_to_tensorrt,
                load_optimized_model,
                save_optimized_model,
                check_tensorrt_available
            )
            if not check_tensorrt_available():
                print("ERROR: TensorRT requested but not available. Exiting.")
                return [], 0, 0
        except ImportError as e:
            print(f"ERROR: Failed to import tensorrt_utils: {e}")
            print("TensorRT functionality is not available. Exiting.")
            return [], 0, 0
    
    # Store run configuration for logging
    run_config = {
        'use_tensorrt': use_tensorrt,
        'tensorrt_precision': tensorrt_precision,
        'model_type': model_type,
        'batch_size': opt['batch_size'],
        'parent_folder': parent_folder_path,
        'target_folder': target_folder_path,
        'save_png': save_png
    }
    
    # Use PARENT_DIR for default paths
    files_dir = os.path.join(PARENT_DIR, 'files')
    
    # Construct model path based on model_name or default
    if model_name:
        # Use custom model name from saved_models directory
        model_path = os.path.join(PARENT_DIR, 'saved_models', model_name)
    else:
        # Use default model based on model_type
        model_path = os.path.join(PARENT_DIR, 'saved_models', 
                                 '15_net_G.pth' if model_type == 1 else '20_net_G.pth')
    print(f"Using model path: {model_path}")
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

    # Initialize model
    netG = AttGenerator_ResNet50_SkipCon(input_channels=13 if model_type == 1 else 10)
    device = torch.device("cuda:0")

    netG.load_state_dict(torch.load(model_path))
    netG.to(device)
    torch.set_grad_enabled(False)
    netG.eval()

    fact = 1024
    channels = 13 if model_type == 1 else 10
    batch_size = opt['batch_size']
    input_shape = (batch_size, channels, fact, fact)
    scaling_factor = 0.69/2

    if use_tensorrt:
        # Get the actual batch size that will be used
        actual_batch_size = opt['batch_size']
        
        optimized_model = load_optimized_model(model_path, model_type, use_tensorrt, 
                                             tensorrt_precision, actual_batch_size)
        
        if optimized_model is None:
            print(f"Compiling TensorRT model for batch size {actual_batch_size}...")
            if tensorrt_precision is None:
                print("No TensorRT precision specified. Converting to TorchScript...")
                optimized_model = convert_to_torchscript(netG, input_shape)
                save_optimized_model(optimized_model, model_path, model_type, "TorchScript", 
                                   batch_size=actual_batch_size)
            else:
                optimized_model = convert_to_tensorrt(netG, input_shape, tensorrt_precision, 
                                                    tensorrt_workspace)
                save_optimized_model(optimized_model, model_path, model_type, "TensorRT", 
                                   tensorrt_precision, batch_size=actual_batch_size)
        
        netG = optimized_model
    torch.set_grad_enabled(False)

    Tensor2 = torch.FloatTensor
    BATCH_SIZE = opt['batch_size']
    
    print("BATCH_SIZE:", opt['batch_size'])
    print("NUM_WORKERS:", opt['n_cpu'])
    print("Using", torch.cuda.device_count(), "GPU")
    if optim:
        print("OPTIM MODE: enabled — patch preprocessing runs on GPU (reduced CPU contention, ~8x less PCIe traffic per patch)")
    
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
            
            with torch.no_grad():
                for batch in dataloader:
                    torch.cuda.empty_cache()
                    
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
                        
                        target_pixel_size = 0.69
                        scale_factor = vis_pixel_size / target_pixel_size
                        
                        _, original_h, original_w = vis_raw.shape
                        new_h = int(original_h * scale_factor)
                        new_w = int(original_w * scale_factor)
                        
                        img_V = torch.nn.functional.interpolate(
                            vis_raw.unsqueeze(0),
                            size=(new_h, new_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        
                        img_V = torch.clamp(img_V, 0, 1)

                        del vis_raw
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
                    
                    ir = batch['ir'].type(Tensor2)
                    vis = img_V.unsqueeze(0).type(Tensor2)
                
                    name = batch['name'][0]
                    
                    _, bands, _, _ = ir.shape
                    b, _, r, c = vis.shape
                    rn = np.ceil(r/fact)
                    cn = np.ceil(c/fact)
                    
                    DATA = torch.zeros(BATCH_SIZE, channels, fact, fact, device=device)
                    fhr = torch.zeros(b, 3, int(rn)*fact+1, int(cn)*fact+1).cpu().numpy()

                    for kk in range(b):
                        patch_count = 0
                        patches = []
                        I = range(0, int(rn)*fact+1, fact)
                        J = range(0, int(cn)*fact+1, fact)
                        
                        for ii in range(len(I)-1):
                            for jj in range(len(J) - 1):
                                if optim:
                                    # --optim: move the small pre-interpolation slice to GPU first.
                                    # Each slice is ~1/8 the size of the final patch (scaling_factor ~0.345)
                                    # so PCIe traffic is reduced ~8x vs moving the upscaled patch.
                                    # F.pad and F.interpolate then run on GPU, freeing the CPU core
                                    # and reducing CPU contention when two GPU workers run in parallel.
                                    xx = ir[:, :, int(I[ii] * scaling_factor):int(I[ii+1] * scaling_factor),
                                        int(J[jj] * scaling_factor):int(J[jj+1] * scaling_factor)].to(device)
                                    pad_rows = int(fact * scaling_factor) - xx.shape[2] if xx.shape[2] < int(fact * scaling_factor) else 0
                                    pad_cols = int(fact * scaling_factor) - xx.shape[3] if xx.shape[3] < int(fact * scaling_factor) else 0
                                    xx = F.pad(xx, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                                    xx = F.interpolate(xx, size=(fact, fact), mode='bicubic', align_corners=True)

                                    vv = vis[:, :, int(I[ii]):int(I[ii+1]), int(J[jj]):int(J[jj+1])].to(device)
                                    pad_rows = fact - vv.shape[2] if vv.shape[2] < fact else 0
                                    pad_cols = fact - vv.shape[3] if vv.shape[3] < fact else 0
                                    vv = F.pad(vv, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                                else:
                                    # Default: CPU preprocessing then copy assembled patch to GPU
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
                                    DATA[patch_count, :bands, :, :] = xx
                                    DATA[patch_count, bands:, :, :] = vv
                                else:
                                    DATA[patch_count, :bands, :, :] = xx
                                    
                                patches.append((ii, jj))
                                patch_count += 1

                                # Process when we reach batch size OR at the end of all patches
                                if patch_count == BATCH_SIZE or (ii == len(I)-2 and jj == len(J)-2):
                                    if use_tensorrt:
                                        # Always use full batch size for TensorRT
                                        if patch_count < BATCH_SIZE:
                                            # Pad remaining slots with zeros (or duplicate last patch)
                                            for pad_idx in range(patch_count, BATCH_SIZE):
                                                DATA[pad_idx] = DATA[patch_count - 1]
                                        
                                        output = netG(DATA) * 0.5 + 0.5  # Use full DATA tensor
                                        
                                        # Only use the first patch_count outputs
                                        output = output[:patch_count]
                                    else:
                                        output = netG(DATA[:patch_count]) * 0.5 + 0.5

                                    # CRITICAL FIX: Handle output shape properly
                                    # Ensure output has shape [N, C, H, W] even for single batch
                                    if output.dim() == 3:  # [C, H, W] - single image without batch dim
                                        output = output.unsqueeze(0)  # Make it [1, C, H, W]
                                    
                                    output = output.cpu().numpy()

                                    for idx, (pi, pj) in enumerate(patches):
                                        fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]

                                    patch_count = 0
                                    patches = []

                        # Handle any remaining patches
                        if patch_count > 0:
                            if use_tensorrt:
                                # Always use full batch size for TensorRT
                                if patch_count < BATCH_SIZE:
                                    # Pad remaining slots with zeros or duplicate last patch
                                    for pad_idx in range(patch_count, BATCH_SIZE):
                                        DATA[pad_idx] = DATA[patch_count - 1]
                                
                                output = netG(DATA) * 0.5 + 0.5  # Use full DATA tensor
                                
                                # Only use the first patch_count outputs
                                output = output[:patch_count]
                            else:
                                # For non-TensorRT models, can use variable batch size
                                output = netG(DATA[:patch_count]) * 0.5 + 0.5

                            # Ensure output has shape [N, C, H, W] even for single batch
                            if output.dim() == 3:  # [C, H, W] - single image without batch dim
                                output = output.unsqueeze(0)  # Make it [1, C, H, W]
                            
                            output = output.cpu().numpy()
                            
                            for idx, (pi, pj) in enumerate(patches):
                                fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]
                        fhr = fhr[:, :, :r, :c]
                        
                        inference_time = time.time() - inference_start
                        inference_times.append(inference_time)
                        write_start = time.time()
                        img_array = (np.transpose(fhr[0], (1, 2, 0)) * 255).astype(np.uint8)
                        
                        # Use custom filename if provided, otherwise use original name
                        final_name = output_name if output_name else name
                        
                        if save_png:
                            save_png_output(img_array, final_name, model_type)
                        else:
                            create_dzi(
                                img_array, final_name, model_type,
                                project_name=project_name, vs_modality=vs_modality,
                            )
                            # Add to database instead of updating JSON
                            if DB_HELPER_AVAILABLE:
                                db_type = vs_modality if vs_modality else image_type
                                add_available_image(
                                    final_name, f"{int(c)}x{int(r)}", db_type, project=project_name,
                                )
                            else:
                                # Fallback to JSON method
                                update_json(final_name, int(r), int(c), model_type)
                            
                        write_time = time.time() - write_start
                        
                        folder_total_time = time.time() - folder_start_time
                        folder_times.append(folder_total_time)

                        del fhr
                        del img_array
                        if 'DATA' in locals():
                            del DATA
                        if 'output' in locals():
                            del output
                        del dataloader
                        del single_folder_dataset
                        gc.collect()
                        torch.cuda.empty_cache()
                        
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
            if target_folder_path:
                return [], 0, 0
            continue
    
    if len(results) > 1:
        avg_bsq_load_time = np.mean(bsq_load_times)
        avg_inference_time = np.mean(inference_times)
        avg_folder_time = np.mean(folder_times)
        avg_write_time = np.mean([r['write_time'] for r in results])
        
        summary_stats = {
            'avg_bsq_load_time': avg_bsq_load_time,
            'avg_inference_time': avg_inference_time,
            'avg_write_time': avg_write_time,
            'avg_folder_time': avg_folder_time
        }
        
        print(f"\n=== SUMMARY ===")
        print(f"Processed {len(results)} folders")
        print(f"Average BSQ Load Time: {np.round(avg_bsq_load_time, 2)} seconds")
        print(f"Average Inference Time: {np.round(avg_inference_time, 2)} seconds")
        print(f"Average Write Time: {np.round(avg_write_time, 2)} seconds")
        print(f"Average Folder Processing Time: {np.round(avg_folder_time, 2)} seconds")
        
    else:
        summary_stats = None if len(results) == 0 else {
            'avg_bsq_load_time': results[0]['bsq_load_time'],
            'avg_inference_time': results[0]['inference_time'],
            'avg_write_time': results[0]['write_time'],
            'avg_folder_time': results[0]['total_time']
        }
    
    # Log results to CSV
    if results:  # Only log if we have results
        try:
            log_run_to_csv(results, run_config, summary_stats)
        except Exception as e:
            print(f"Error logging to CSV: {e}")
            import traceback
            traceback.print_exc()
        
    return results

def save_png_output(img_array, filename, model_type=0):
    """Save output as PNG file"""
    # Choose subdirectory based on model_type
    subdir = 'IR' if model_type == 0 else 'IR+VIS'
    output_dir = os.path.join(PARENT_DIR, 'imagesT')
    os.makedirs(output_dir, exist_ok=True)

    # Save as PNG
    output_path = os.path.join(output_dir, f'{filename}.png')
    
    # Convert numpy array to PIL Image and save
    img = Image.fromarray(img_array)
    
    # Remove file if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted existing PNG: {filename}.png")
        
    img.save(output_path, 'PNG')
    print(f"Saved PNG output to: {output_path}")

def create_dzi(img_array, filename, model_type=0, project_name=DEFAULT_PROJECT_NAME, vs_modality=None):
    """Create DZI folder and thumbnail from numpy array using pyvips"""

    # Subdirectory under dzi/ and thumbnail/ (viewer uses image_type == folder name)
    if vs_modality:
        subdir = vs_modality.strip()
    else:
        subdir = 'IR(VS)' if model_type == 0 else 'IR+VIS(VS)'
    project_slug = (project_name or DEFAULT_PROJECT_NAME or 'main').strip() or 'main'
    dzi_files_dir = os.path.join(FILES_BASE_DIR, project_slug, 'dzi', subdir)
    thumbnail_dir = os.path.join(FILES_BASE_DIR, project_slug, 'thumbnail', subdir)
    os.makedirs(dzi_files_dir, exist_ok=True)
    os.makedirs(thumbnail_dir, exist_ok=True)

    # Convert numpy array to vips image
    height, width, bands = img_array.shape
    linear = img_array.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands, 'uchar')

    # Save as deep zoom (DZI folder)
    dzi_base = os.path.join(dzi_files_dir, filename)
    dzi_descriptor = dzi_base + '.dzi'

    if os.path.exists(dzi_descriptor):
        os.remove(dzi_descriptor)
        import shutil
        tiles_dir = dzi_base + '_files'
        if os.path.isdir(tiles_dir):
            shutil.rmtree(tiles_dir)
        print(f"Deleted existing DZI: {filename}")

    vi.dzsave(dzi_base,
              overlap=0,
              container='fs',
              suffix='.jpg',
              compression=0,
              tile_size=1024)

    # Save thumbnail
    thumbnail = vi.thumbnail_image(400)
    thumbnail_path = os.path.join(thumbnail_dir, f"{filename}.jpg")
    thumbnail.jpegsave(thumbnail_path, Q=75)

def update_json(filename, height, width, model_type):
    """Update JSON list with image dimensions and model_type as unique key (DEPRECATED - use database instead)"""
    print("Warning: Using deprecated JSON method. Database helper not available.")
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
        'n_cpu': 0,
        'optim': args.optim
    }
    results = main(opt,
                   target_folder_path=args.folder,
                   parent_folder_path=args.parent_folder,
                   image_type=args.image_type,
                   model_type=args.model,
                   model_name=args.model_name,
                   use_tensorrt=args.use_tensorrt,
                   tensorrt_precision=args.tensorrt_precision,
                   tensorrt_workspace=args.tensorrt_workspace,
                   save_png=args.save_png,
                   output_name=args.output_name,
                   optim=args.optim,
                   project_name=args.project,
                   vs_modality=args.vs_modality)