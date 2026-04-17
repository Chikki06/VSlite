"""
4-Channel Inference Script for VSlite
This script handles inference using only 4 IR channels (no VIS data required).
Based on main.py but simplified for 4-channel IR-only operation.
"""

import os
import pyvips
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import numpy as np
from torch.utils.data import Dataset, DataLoader
import spectral
import time
import torch.nn.functional as F
import json
import argparse
from PIL import Image
import pandas as pd
from datetime import datetime
import gc

Image.MAX_IMAGE_PIXELS = None

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run 4-channel inference on hyperspectral images')
    parser.add_argument('--folder', type=str, default=None, 
                       help='Specific folder name to process (optional)')
    parser.add_argument('--parent_folder', type=str, default=None,
                       help='Parent directory containing multiple folders to process in batch (optional)')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model filename (located in saved_models/) trained with 4 channels')
    # Add TensorRT arguments
    parser.add_argument('--use_tensorrt', action='store_true', default=False,
                       help='Use TensorRT optimization for inference')
    parser.add_argument('--tensorrt_precision', type=str, default=None, choices=['fp32', 'fp16'],
                   help='TensorRT precision mode (default: None - uses TorchScript without precision changes)')
    parser.add_argument('--tensorrt_workspace', type=int, default=1024,
                       help='TensorRT workspace size in MB (default: 1024)')
    # Add TIFF saving and filename arguments
    parser.add_argument('--save_png', action='store_true', default=False,
                       help='Save output as PNG instead of SZI')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Custom output filename (without extension)')
    return parser.parse_args()


class SingleFolderDataset4Channel(Dataset):
    """Dataset for loading 4-channel IR data only"""
    def __init__(self, root, folder_name):
        self.parent = root
        self.folder_name = folder_name
        self._validate_required_folders()
        
    def _validate_required_folders(self):
        """Validate that IR folder exists and contains valid data files"""
        ir_folder_path = os.path.join(self.parent, self.folder_name)
        
        if not self._validate_folder_contents(ir_folder_path):
            raise FileNotFoundError(f"Invalid IR folder or missing HDR/data files: {ir_folder_path}")

    def _validate_folder_contents(self, folder_path):
        """Check if folder exists and contains valid HDR + data file pairs"""
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return False
        
        files = os.listdir(folder_path)
        hdr_files = [f for f in files if f.endswith('.hdr')]
        
        if not hdr_files:
            return False
            
        for hdr_file in hdr_files:
            base_name = os.path.splitext(hdr_file)[0]
            data_file = self._find_data_file(base_name, files)
            if data_file:
                return True
        return False

    def _find_data_file(self, base_name, folder_files):
        """Find data file with supported extensions"""
        supported_extensions = ['.bsq', '.bip', '.bil', '.dat', '.img']
        for ext in supported_extensions:
            data_file = base_name + ext
            if data_file in folder_files:
                return data_file
        return None

    def __getitem__(self, index):
        """Load IR data for 4 channels only"""
        folder_path = os.path.join(self.parent, self.folder_name)
        
        start_time = time.time()
        
        # Find HDR files
        files_in_folder = os.listdir(folder_path)
        hdr_files = [f for f in files_in_folder if f.endswith('.hdr')]
        
        if not hdr_files:
            raise FileNotFoundError(f"No HDR files found in {folder_path}")
        
        # Use first HDR file
        hdr_file = hdr_files[0]
        base_name = os.path.splitext(hdr_file)[0]
        hdr_path = os.path.join(folder_path, hdr_file)
        
        # Load IR data using spectral library
        img = spectral.open_image(hdr_path)
        ir_data = img.load()
        
        # Extract only first 4 channels
        ir_data = ir_data[:, :, :4]
        
        # Normalize
        ir_data = ir_data.astype(np.float32)
        ir_data = np.clip(ir_data, 0, None)
        
        # Convert to tensor [C, H, W]
        ir_tensor = torch.from_numpy(ir_data).permute(2, 0, 1).float()
        
        load_time = time.time() - start_time
        
        return {
            'ir': ir_tensor,
            'bsq_load_time': torch.tensor(load_time),
            'name': self.folder_name
        }
    
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


class AttGenerator_ResNet50_SkipCon_4Channel(nn.Module):
    """4-Channel AttUNet Generator using ResNet50 backbone"""
    def __init__(self, input_channels=4, norm_layer=nn.BatchNorm2d, fpn_feature='decoder'):
        super(AttGenerator_ResNet50_SkipCon_4Channel, self).__init__()        

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer for 4 input channels
        new_conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1 = new_conv1
        self.encoder = IntermediateLayerGetter(backbone, return_layers={
            "conv1": 'feat', 
            "layer1": 'feat0', 
            "layer2": 'feat1', 
            "layer3": 'feat2', 
            "layer4": 'feat3'
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


def custom_collate_fn(batch):
    """Custom collate function"""
    collated = {}
    for key in batch[0].keys():
        if key in ['name']:
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated


def save_png_output(img_array, filename, channels=4):
    """Save output as PNG file"""
    output_dir = os.path.join(PARENT_DIR, 'imagesT')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{filename}.png')
    
    img = Image.fromarray(img_array)
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted existing PNG: {filename}.png")
        
    img.save(output_path, 'PNG')
    print(f"Saved PNG output to: {output_path}")


def create_szi(img_array, filename, channels=4):
    """Create .szi file and thumbnail from numpy array using pyvips"""
    subdir = 'IR_4ch'
    viewer_files_dir = os.path.join('home','labuser', 'SocketInference', 'inference-project', 'viewer', 'files', subdir)
    viewer_thumbnail_dir = os.path.join('home','labuser', 'SocketInference', 'inference-project', 'viewer', 'thumbnail', subdir)
    os.makedirs(viewer_files_dir, exist_ok=True)
    os.makedirs(viewer_thumbnail_dir, exist_ok=True)

    height, width, bands = img_array.shape
    linear = img_array.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands, 'uchar')

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

    thumbnail = vi.thumbnail_image(400)
    thumbnail_path = os.path.join(viewer_thumbnail_dir, f"{filename}.jpg")
    thumbnail.jpegsave(thumbnail_path, Q=75)


def update_json(filename, height, width, channels=4):
    """Update JSON list with image dimensions"""
    json_path = os.path.join(PARENT_DIR, '..', 'viewer', 'static', 'available.json')

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    resolution_str = f"{width}x{height}"

    for entry in data:
        if entry.get('name') == filename and entry.get('channels') == channels:
            entry['resolution'] = resolution_str
            break
    else:
        data.append({'name': filename, 'resolution': resolution_str, 'channels': channels})

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def main(opt, target_folder_path=None, parent_folder_path=None, model_name=None,
         use_tensorrt=False, tensorrt_precision=None, tensorrt_workspace=1024,
         save_png=False, output_name=None):
    """Main inference function for 4-channel mode"""
    
    files_dir = os.path.join(PARENT_DIR, 'files')
    
    # Construct model path
    if not model_name:
        raise ValueError("model_name is required for 4-channel inference")
    
    model_path = os.path.join(PARENT_DIR, 'saved_models', model_name)
    print(f"Using 4-channel model: {model_path}")
    
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
    # Handle parent folder path
    elif parent_folder_path:
        parent_folder_path = parent_folder_path.strip('"').strip("'")
        if os.path.exists(parent_folder_path) and os.path.isdir(parent_folder_path):
            files_dir = parent_folder_path
            folder_list = [
                dir_name for dir_name in os.listdir(files_dir)
                if os.path.isdir(os.path.join(files_dir, dir_name))
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
            if os.path.isdir(os.path.join(files_dir, dir_name))
        ]
        
        if not folder_list:
            print("Error: No folders found")
            return [], 0, 0

    # Initialize 4-channel model
    netG = AttGenerator_ResNet50_SkipCon_4Channel(input_channels=4)
    device = torch.device("cuda:0")

    netG.load_state_dict(torch.load(model_path))
    netG.to(device)
    torch.set_grad_enabled(False)
    netG.eval()

    fact = 1024
    channels = 4
    batch_size = opt['batch_size']
    scaling_factor = 0.69/2

    Tensor2 = torch.FloatTensor
    BATCH_SIZE = opt['batch_size']
    
    print("BATCH_SIZE:", opt['batch_size'])
    print("NUM_WORKERS:", opt['n_cpu']) 
    print("Using", torch.cuda.device_count(), "GPU")
    print("Model: 4-channel IR only")
    
    results = []
    bsq_load_times = []
    inference_times = []
    folder_times = []
    
    # Process each folder individually
    for folder_idx, folder_name in enumerate(folder_list):
        print(f"\nProcessing folder {folder_idx + 1}/{len(folder_list)}: {folder_name}")
        
        try:
            # Create single-folder dataset
            single_folder_dataset = SingleFolderDataset4Channel(files_dir, folder_name)
            dataloader = DataLoader(single_folder_dataset, batch_size=opt['batch_size'], 
                                  shuffle=False, num_workers=opt['n_cpu'], collate_fn=custom_collate_fn)
            
            folder_start_time = time.time()
            
            for batch in dataloader:
                torch.cuda.empty_cache()
                
                bsq_load_times.append(batch['bsq_load_time'].item())
                inference_start = time.time()
                
                ir = batch['ir'].type(Tensor2)
                name = batch['name'][0]
                
                _, bands, r, c = ir.shape
                rn = np.ceil(r/fact)
                cn = np.ceil(c/fact)
                
                DATA = torch.zeros(BATCH_SIZE, channels, fact, fact, device=device)
                fhr = torch.zeros(1, 3, int(rn)*fact+1, int(cn)*fact+1).cpu().numpy()

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
                        
                        # Use only IR channels
                        DATA[patch_count, :channels, :, :] = xx
                            
                        patches.append((ii, jj))
                        patch_count += 1

                        # Process when we reach batch size OR at the end
                        if patch_count == BATCH_SIZE or (ii == len(I)-2 and jj == len(J)-2):
                            output = netG(DATA[:patch_count]) * 0.5 + 0.5

                            if output.dim() == 3:
                                output = output.unsqueeze(0)
                            
                            output = output.cpu().numpy()

                            for idx, (pi, pj) in enumerate(patches):
                                fhr[0, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]

                            patch_count = 0
                            patches = []

                # Handle any remaining patches
                if patch_count > 0:
                    output = netG(DATA[:patch_count]) * 0.5 + 0.5

                    if output.dim() == 3:
                        output = output.unsqueeze(0)
                    
                    output = output.cpu().numpy()
                    
                    for idx, (pi, pj) in enumerate(patches):
                        fhr[0, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]
                
                fhr = fhr[:, :, :r, :c]
                
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                write_start = time.time()
                img_array = (np.transpose(fhr[0], (1, 2, 0)) * 255).astype(np.uint8)
                
                final_name = output_name if output_name else name
                
                if save_png:
                    save_png_output(img_array, final_name, channels=4)
                else:
                    create_szi(img_array, final_name, channels=4)
                    update_json(final_name, int(r), int(c), channels=4)
                    
                write_time = time.time() - write_start
                
                folder_total_time = time.time() - folder_start_time
                folder_times.append(folder_total_time)

                del fhr
                del img_array
                del dataloader
                del single_folder_dataset
                gc.collect()
                
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
                
        except Exception as e:
            print(f"Error processing folder {folder_name}: {str(e)}")
            continue
    
    # Calculate summary statistics
    if results:
        avg_bsq_load_time = np.mean(bsq_load_times)
        avg_inference_time = np.mean(inference_times)
        avg_folder_time = np.mean(folder_times)
        
        print("\n" + "="*50)
        print("SUMMARY STATISTICS (4-Channel Mode)")
        print("="*50)
        print(f"Total folders processed: {len(results)}")
        print(f"Average BSQ Load Time: {np.round(avg_bsq_load_time, 2)} seconds")
        print(f"Average Inference Time: {np.round(avg_inference_time, 2)} seconds")
        print(f"Average Total Folder Time: {np.round(avg_folder_time, 2)} seconds")
        
        return results, avg_inference_time, avg_folder_time
    else:
        print("No results to summarize")
        return [], 0, 0


if __name__ == '__main__':
    start0 = time.time()
    args = parse_arguments()
    opt = {
        'batch_size': 4,
        'n_cpu': 0
    }
    results = main(opt, 
                   target_folder_path=args.folder, 
                   parent_folder_path=args.parent_folder, 
                   model_name=args.model_name,
                   use_tensorrt=args.use_tensorrt,
                   tensorrt_precision=args.tensorrt_precision,
                   tensorrt_workspace=args.tensorrt_workspace,
                   save_png=args.save_png,
                   output_name=args.output_name)
    
    end0 = time.time()
    print(f"\nTotal execution time: {np.round(end0 - start0, 2)} seconds")
