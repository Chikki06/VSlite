import os
import pyvips
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import os
import numpy as np
from torch.utils.data import Dataset
import spectral
import time
from torch.amp import autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import argparse
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on hyperspectral images')
    parser.add_argument('--autocast', type=bool, default=True)
    parser.add_argument('--folder', type=str, default=None, 
                       help='Specific folder name to process (optional)')
    parser.add_argument('--parent_folder', type=str, default=None,
                       help='Parent directory containing multiple folders to process in batch (optional)')
    parser.add_argument('--model', type=int, default=0, choices=[0, 1],
                       help='Model type: 0 for 10-channel (default), 1 for 13-channel with VIS')
    parser.add_argument('--use_warmup', action='store_true', default=False,
                       help='Use warmup before inference')
    # Add TensorRT arguments
    parser.add_argument('--use_tensorrt', action='store_true', default=False,
                       help='Use TensorRT optimization for inference')
    parser.add_argument('--tensorrt_precision', type=str, default=None, choices=['fp32', 'fp16', 'int8'],
                       help='TensorRT precision mode (default: None - uses TorchScript without precision changes)')
    parser.add_argument('--tensorrt_workspace', type=int, default=1024,
                       help='TensorRT workspace size in MB (default: 1024)')
    return parser.parse_args()

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
    def __init__(self, input_channels=10, norm_layer=nn.BatchNorm2d, fpn_feature='decoder'):
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

def warmup_model(model, input_shape, num_warmup=10):
    """Warmup model for optimal performance"""
    start = time.time()
    print(f"Warming up model with {num_warmup} iterations...")
    dummy_input = torch.randn(input_shape).cuda()
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    print("Model warmup completed in {:.2f} seconds".format(time.time() - start))

class SingleFolderDataset(Dataset):
    def __init__(self, root, folder_name, model_type=0):
        self.parent = root
        self.folder_name = folder_name
        self.model_type = model_type
        
    def __getitem__(self, index):
        start = time.time()
        
        # Load IR data from parent/folder_name
        ir_folder_path = os.path.join(self.parent, self.folder_name)
        img = self._load_ir_data(ir_folder_path)
        
        # Handle VIS data if needed
        vis_data = None
        vis_pixel_size = None
        
        if self.model_type == 1:
            # Load VIS data from parent/../VIS/folder_name_VIS_REG.tif
            vis_folder_path = os.path.join(os.path.dirname(self.parent), 'VIS')
            vis_data, vis_pixel_size = self._load_vis_data_tiff(vis_folder_path)
        
        bsq_load_time = time.time() - start
        
        return {
            'ir': img[:,:,:],
            'vis_raw': vis_data,
            'vis_pixel_size': vis_pixel_size,
            'name': self.folder_name,
            'bsq_load_time': torch.tensor(bsq_load_time)
        }

    def _load_ir_data(self, ir_folder_path):
        """Load IR data from the specified folder path."""
        files = os.listdir(ir_folder_path)
        
        # Find HDR file
        hdr_files = [f for f in files if f.endswith('.hdr')]
        hdr_file = hdr_files[0]
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

    def _load_vis_data_tiff(self, vis_folder_path):
        """Load VIS TIFF data from parent/VIS/folder_name_VIS_REG.tif"""
        # Construct the expected TIFF filename
        vis_tiff_filename = f"{self.folder_name}_VIS_REG.tif"
        vis_tiff_path = os.path.join(vis_folder_path, vis_tiff_filename)
        
        # Also try .tiff extension if .tif doesn't exist
        if not os.path.exists(vis_tiff_path):
            vis_tiff_filename = f"{self.folder_name}_VIS_REG.tiff"
            vis_tiff_path = os.path.join(vis_folder_path, vis_tiff_filename)
        
        print(f"Loading VIS TIFF: {vis_tiff_path}")
        

        
        vis_image = Image.open(vis_tiff_path)
        vis_data = np.array(vis_image)
        
        # Handle different image formats
        if len(vis_data.shape) == 2:  # Grayscale
            vis_data = np.expand_dims(vis_data, axis=2)
        elif len(vis_data.shape) == 3 and vis_data.shape[2] > 3:  # Multi-channel
            # Keep all channels
            pass
        
        # Normalize and convert to float32
        vis_data = vis_data / 255.0  # Assuming 8-bit TIFF
        vis_data = vis_data.astype(np.float32)
        vis_data = np.transpose(vis_data, (2, 0, 1))  # HWC to CHW
        vis_data = torch.from_numpy(vis_data).float()
        
        # Fixed pixel size
        vis_pixel_size = 0.69  # microns per pixel
        
        return vis_data, vis_pixel_size
    
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

def main(opt, use_autocast, target_folder_path=None, parent_folder_path=None, model_type=0, use_tensorrt=False, 
         tensorrt_precision=None, use_warmup=False, tensorrt_workspace=1024): 
    # Use PARENT_DIR for default paths
    files_dir = os.path.join(PARENT_DIR, 'files')
    model_path = os.path.join(PARENT_DIR, 'saved_models', 
                             '15_net_G.pth' if model_type == 1 else '20_net_G.pth')

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    if use_tensorrt or use_warmup:
        warmup_model(netG, input_shape)
    torch.set_grad_enabled(False)

    Tensor2 = torch.FloatTensor
    BATCH_SIZE = opt['batch_size']
    
    print("BATCH_SIZE:", opt['batch_size'])
    print("NUM_WORKERS:", opt['n_cpu']) 
    print("Using", torch.cuda.device_count(), "GPU")
    
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
                if model_type == 1:
                    ir = batch['ir'].type(Tensor2)
                    vis = img_V.unsqueeze(0).type(Tensor2)
                else:
                    ir = (batch['ir'].type(Tensor2) - 0.5) / 0.5
                    vis = (img_V.unsqueeze(0).type(Tensor2) - 0.5) / 0.5
                
                name = batch['name'][0]
                
                _, bands, _, _ = ir.shape
                b, _, _, _ = vis.shape
                _, _, r, c = vis.shape
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
                                            DATA[pad_idx] = DATA[patch_count-1]  # Duplicate last patch
                                    
                                    if use_autocast:
                                        with autocast('cuda', enabled=True):
                                            output = netG(DATA).squeeze() * 0.5 + 0.5  # Use full DATA tensor
                                    else:
                                        output = netG(DATA).squeeze() * 0.5 + 0.5  # Use full DATA tensor
                                    
                                    # Only use the first patch_count outputs
                                    output = output[:patch_count]
                                else:
                                    # For non-TensorRT models, can use variable batch size
                                    if use_autocast:
                                        with autocast('cuda', enabled=True):
                                            output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5
                                    else:
                                        output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5

                                output = output.cpu().numpy()

                                for idx, (pi, pj) in enumerate(patches):
                                    fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]

                                patch_count = 0
                                patches = []

                    if patch_count > 0:
                        if use_tensorrt:
                            # Always use full batch size for TensorRT
                            if patch_count < BATCH_SIZE:
                                # Pad remaining slots with zeros or duplicate last patch
                                for pad_idx in range(patch_count, BATCH_SIZE):
                                    DATA[pad_idx] = DATA[patch_count-1]  # Duplicate last patch
                            
                            if use_autocast:
                                with autocast('cuda', enabled=True):
                                    output = netG(DATA).squeeze() * 0.5 + 0.5  # Use full DATA tensor
                            else:
                                output = netG(DATA).squeeze() * 0.5 + 0.5  # Use full DATA tensor
                            
                            # Only use the first patch_count outputs
                            output = output[:patch_count]
                        else:
                            # For non-TensorRT models, can use variable batch size
                            if use_autocast:
                                with autocast('cuda', enabled=True):
                                    output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5
                            else:
                                output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5

                        output = output.cpu().numpy() 
                        
                        for idx, (pi, pj) in enumerate(patches):
                            fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]                    
                fhr = fhr[:, :, :r, :c]
                
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
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
            if target_folder_path:
                return [], 0, 0
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
    thumbnail = vi.thumbnail_image(400)
    thumbnail_path = os.path.join(viewer_thumbnail_dir, f"{filename}.jpg")
    thumbnail.jpegsave(thumbnail_path, Q=75)

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
    results = main(opt, 
                   use_autocast=args.autocast, 
                   target_folder_path=args.folder, 
                   parent_folder_path=args.parent_folder, 
                   model_type=args.model, 
                   use_warmup=args.use_warmup,
                   use_tensorrt=args.use_tensorrt,
                   tensorrt_precision=args.tensorrt_precision,
                   tensorrt_workspace=args.tensorrt_workspace)