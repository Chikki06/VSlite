# this is the main from the inference server which does not go OOM.
import os
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
import argparse
from PIL import Image
import gc
import sys

# Add viewer/misc path for database helper
VIEWER_MISC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'viewer', 'misc'))
if VIEWER_MISC_PATH not in sys.path:
    sys.path.insert(0, VIEWER_MISC_PATH)

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
    parser.add_argument('--model_path', type=str, default=None,
                       help='Absolute or relative path to a checkpoint file. If set, overrides --model_name/default model.')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Inference batch size for patch processing.')
    parser.add_argument('--n_cpu', type=int, default=0,
                       help='Number of dataloader workers.')
    parser.add_argument('--image_type', type=str, default='IR',
                       help="Image type string (e.g. 'IR' or 'IR+VIS') - kept for backward compatibility with older callers")
    parser.add_argument('--save_png', action='store_true', default=True,
                       help='Save output as PNG')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Custom output filename (without extension)')
    parser.add_argument('--optim', action='store_true', default=False,
                       help='Enable optimized parallel mode: move per-patch preprocessing to GPU '
                            'to reduce CPU contention and PCIe traffic when two GPU jobs run concurrently')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT_NAME,
                       help='Project name for output under <files_base_dir>/<project>/...')
    parser.add_argument('--vs_modality', type=str, default=None,
                       help='PNG subfolder for VS output (e.g. VS_G10_10ch). '
                            'Default: IR(VS) for --model 0, IR+VIS(VS) for --model 1.')
    parser.add_argument('--png_downscale', type=float, default=0.5,
                       help='Downscale factor applied only at PNG save time. Inference stays full resolution.')
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

def _candidate_project_slugs(project_name):
    candidates = []
    for candidate in [project_name, DEFAULT_PROJECT_NAME, 'main']:
        if candidate is None:
            continue
        slug = str(candidate).strip()
        if slug and slug not in candidates:
            candidates.append(slug)
    return candidates


def _output_subdir(model_type, vs_modality):
    if vs_modality:
        return vs_modality.strip()
    return 'IR(VS)' if model_type == 0 else 'IR+VIS(VS)'


def output_artifact_exists(filename, model_type=0, project_name=DEFAULT_PROJECT_NAME, vs_modality=None):
    subdir = _output_subdir(model_type, vs_modality)
    for project_slug in _candidate_project_slugs(project_name):
        output_path = os.path.join(FILES_BASE_DIR, project_slug, 'png', subdir, f'{filename}.png')
        if os.path.exists(output_path):
            return True, output_path
    return False, None


def main(opt, target_folder_path=None, parent_folder_path=None, image_type='IR', model_type=None, model_name=None,
         model_path=None, save_png=True, output_name=None, optim=False,
         project_name=DEFAULT_PROJECT_NAME, vs_modality=None, png_downscale=0.5):
    if not save_png:
        print('Warning: only PNG output is supported in this simplified script; forcing PNG output.')

    if model_type is None:
        model_type = 1 if image_type == 'IR+VIS' else 0
    else:
        model_type = int(model_type)

    if png_downscale <= 0:
        raise ValueError('png_downscale must be > 0')

    files_dir = os.path.join(PARENT_DIR, 'files')
    if target_folder_path:
        target_folder_path = target_folder_path.strip('"').strip("'")
        if not (os.path.exists(target_folder_path) and os.path.isdir(target_folder_path)):
            print(f"Error: Folder path does not exist: {target_folder_path}")
            return []
        files_dir = os.path.dirname(target_folder_path)
        folder_list = [os.path.basename(target_folder_path)]
    elif parent_folder_path:
        parent_folder_path = parent_folder_path.strip('"').strip("'")
        if not (os.path.exists(parent_folder_path) and os.path.isdir(parent_folder_path)):
            print(f"Error: Parent folder path does not exist: {parent_folder_path}")
            return []
        files_dir = parent_folder_path
        folder_list = [
            d for d in os.listdir(files_dir)
            if os.path.isdir(os.path.join(files_dir, d)) and not d.endswith('_VIS')
        ]
    else:
        folder_list = [
            d for d in os.listdir(files_dir)
            if os.path.isdir(os.path.join(files_dir, d)) and not d.endswith('_VIS')
        ]

    if not folder_list:
        print('Error: No folders found')
        return []

    if model_path:
        resolved_model_path = os.path.abspath(model_path)
    elif model_name:
        resolved_model_path = os.path.join(PARENT_DIR, 'saved_models', model_name)
    else:
        resolved_model_path = os.path.join(PARENT_DIR, 'saved_models', '15_net_G.pth' if model_type == 1 else '20_net_G.pth')

    if not os.path.isfile(resolved_model_path):
        print(f"Error: model checkpoint not found: {resolved_model_path}")
        return []

    device = torch.device('cuda:0')
    netG = AttGenerator_ResNet50_SkipCon(input_channels=13 if model_type == 1 else 10)
    netG.load_state_dict(torch.load(resolved_model_path, map_location=device))
    netG.to(device)
    netG.eval()

    fact = 1024
    channels = 13 if model_type == 1 else 10
    scaling_factor = 0.69 / 2
    batch_size = int(opt['batch_size'])

    print(f'Using model path: {resolved_model_path}')
    print(f'BATCH_SIZE: {batch_size}')
    print(f'NUM_WORKERS: {opt["n_cpu"]}')
    print(f'PNG downscale at save: {png_downscale}')

    results = []
    for folder_idx, folder_name in enumerate(folder_list):
        print(f"\nProcessing folder {folder_idx + 1}/{len(folder_list)}: {folder_name}")
        final_name = output_name if output_name else folder_name
        exists, existing_path = output_artifact_exists(final_name, model_type=model_type, project_name=project_name, vs_modality=vs_modality)
        if exists:
            print(f"Skipping folder {folder_name}: output already exists at {existing_path}")
            continue

        folder_start = time.time()
        try:
            dataset = SingleFolderDataset(files_dir, folder_name, model_type)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(opt['n_cpu']),
                collate_fn=custom_collate_fn,
            )

            with torch.inference_mode():
                for batch in dataloader:
                    bsq_load_time = float(batch['bsq_load_time'].item())
                    inference_start = time.time()

                    ir = batch['ir'].float()
                    vis_raw = batch['vis_raw'][0]
                    vis_pixel_size = batch['vis_pixel_size'][0]

                    vis_source = None
                    vis_scale_factor = None
                    if model_type == 1 and vis_raw is not None and vis_pixel_size is not None:
                        vis_source = vis_raw.unsqueeze(0).float()
                        vis_scale_factor = max(float(vis_pixel_size) / 0.69, 1e-6)

                    _, bands, ir_h, ir_w = ir.shape
                    if model_type == 1 and vis_source is not None and vis_scale_factor is not None:
                        b = ir.shape[0]
                        _, _, vis_h, vis_w = vis_source.shape
                        r = max(1, int(vis_h * vis_scale_factor))
                        c = max(1, int(vis_w * vis_scale_factor))
                    else:
                        b = ir.shape[0]
                        r = int(ir_h * (2.0 / 0.69))
                        c = int(ir_w * (2.0 / 0.69))

                    rn = np.ceil(r / fact)
                    cn = np.ceil(c / fact)
                    DATA = torch.zeros(batch_size, channels, fact, fact, device=device)
                    fhr = np.zeros((b, 3, int(rn) * fact + 1, int(cn) * fact + 1), dtype=np.float16)

                    for kk in range(b):
                        patch_count = 0
                        patches = []
                        I = range(0, int(rn) * fact + 1, fact)
                        J = range(0, int(cn) * fact + 1, fact)

                        for ii in range(len(I) - 1):
                            for jj in range(len(J) - 1):
                                if optim:
                                    xx = ir[:, :, int(I[ii] * scaling_factor):int(I[ii + 1] * scaling_factor),
                                            int(J[jj] * scaling_factor):int(J[jj + 1] * scaling_factor)].to(device)
                                else:
                                    xx = ir[:, :, int(I[ii] * scaling_factor):int(I[ii + 1] * scaling_factor),
                                            int(J[jj] * scaling_factor):int(J[jj + 1] * scaling_factor)]

                                pad_rows = int(fact * scaling_factor) - xx.shape[2] if xx.shape[2] < int(fact * scaling_factor) else 0
                                pad_cols = int(fact * scaling_factor) - xx.shape[3] if xx.shape[3] < int(fact * scaling_factor) else 0
                                xx = F.pad(xx, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                                xx = F.interpolate(xx, size=(fact, fact), mode='bicubic', align_corners=True)

                                if model_type == 1 and vis_source is not None and vis_scale_factor is not None:
                                    _, _, vis_h, vis_w = vis_source.shape
                                    y0 = max(0, min(vis_h - 1, int(I[ii] / vis_scale_factor)))
                                    y1 = max(y0 + 1, min(vis_h, int(I[ii + 1] / vis_scale_factor)))
                                    x0 = max(0, min(vis_w - 1, int(J[jj] / vis_scale_factor)))
                                    x1 = max(x0 + 1, min(vis_w, int(J[jj + 1] / vis_scale_factor)))
                                    vv = vis_source[:, :, y0:y1, x0:x1]
                                    if optim:
                                        vv = vv.to(device)
                                    vv = F.interpolate(vv, size=(fact, fact), mode='bilinear', align_corners=False)
                                    vv = torch.clamp(vv, 0, 1)
                                    DATA[patch_count, :bands, :, :] = xx
                                    DATA[patch_count, bands:, :, :] = vv
                                else:
                                    DATA[patch_count, :bands, :, :] = xx

                                patches.append((ii, jj))
                                patch_count += 1

                                if patch_count == batch_size or (ii == len(I) - 2 and jj == len(J) - 2):
                                    output = netG(DATA[:patch_count]) * 0.5 + 0.5
                                    if output.dim() == 3:
                                        output = output.unsqueeze(0)
                                    output = output.cpu().numpy()

                                    for idx, (pi, pj) in enumerate(patches):
                                        fhr[kk, :, int(I[pi]):int(I[pi + 1]), int(J[pj]):int(J[pj + 1])] = output[idx]
                                    patch_count = 0
                                    patches = []

                    fhr = fhr[:, :, :r, :c]
                    inference_time = time.time() - inference_start

                    write_start = time.time()
                    img_array = (np.transpose(fhr[0], (1, 2, 0)) * 255).astype(np.uint8)
                    save_png_output(
                        img_array,
                        final_name,
                        model_type=model_type,
                        project_name=project_name,
                        vs_modality=vs_modality,
                        png_downscale=png_downscale,
                    )
                    write_time = time.time() - write_start

                    folder_total = time.time() - folder_start
                    print(f"  BSQ Load Time: {np.round(bsq_load_time, 2)} seconds")
                    print(f"  Inference Time: {np.round(inference_time, 2)} seconds")
                    print(f"  Write Time: {np.round(write_time, 2)} seconds")
                    print(f"  Total Folder Time: {np.round(folder_total, 2)} seconds")

                    results.append({
                        'name': folder_name,
                        'bsq_load_time': bsq_load_time,
                        'inference_time': inference_time,
                        'write_time': write_time,
                        'total_time': folder_total,
                    })

        except FileNotFoundError as e:
            print(f"Skipping folder {folder_name}: {str(e)}")
            if target_folder_path:
                return []
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if results:
        print("\n=== SUMMARY ===")
        print(f"Processed {len(results)} folders")
        print(f"Average BSQ Load Time: {np.mean([r['bsq_load_time'] for r in results]):.2f} seconds")
        print(f"Average Inference Time: {np.mean([r['inference_time'] for r in results]):.2f} seconds")
        print(f"Average Write Time: {np.mean([r['write_time'] for r in results]):.2f} seconds")
        print(f"Average Folder Processing Time: {np.mean([r['total_time'] for r in results]):.2f} seconds")

    return results


def save_png_output(img_array, filename, model_type=0, project_name=DEFAULT_PROJECT_NAME, vs_modality=None, png_downscale=0.5):
    subdir = _output_subdir(model_type, vs_modality)
    project_slug = (project_name or DEFAULT_PROJECT_NAME or 'main').strip() or 'main'
    output_dir = os.path.join(FILES_BASE_DIR, project_slug, 'png', subdir)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{filename}.png')
    if os.path.exists(output_path):
        print(f"Skipping existing PNG: {output_path}")
        return

    img = Image.fromarray(img_array)
    if png_downscale != 1.0:
        new_w = max(1, int(round(img.width * png_downscale)))
        new_h = max(1, int(round(img.height * png_downscale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img.save(output_path, 'PNG')
    print(f"Saved PNG output to: {output_path}")


if __name__ == '__main__':
    args = parse_arguments()
    opt = {
        'batch_size': int(args.batch_size),
        'n_cpu': int(args.n_cpu),
        'optim': args.optim,
    }
    main(
        opt,
        target_folder_path=args.folder,
        parent_folder_path=args.parent_folder,
        image_type=args.image_type,
        model_type=args.model,
        model_name=args.model_name,
        model_path=args.model_path,
        save_png=args.save_png,
        output_name=args.output_name,
        optim=args.optim,
        project_name=args.project,
        vs_modality=args.vs_modality,
        png_downscale=args.png_downscale,
    )