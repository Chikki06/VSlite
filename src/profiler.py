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
from torch.utils.data import Dataset, TensorDataset
import spectral
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import argparse
import torch_tensorrt
from torch_tensorrt.ts import ptq
import tensorrt as trt
from torch.jit import script
from torch.profiler import profile, ProfilerActivity, record_function

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on hyperspectral images')
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
    parser.add_argument('--enable_profiler', action='store_true', default=False,
                       help='Enable PyTorch profiler for performance analysis')
    parser.add_argument('--tensorrt_precision', type=str, default=None, choices=['fp32', 'fp16', 'int8'],
                   help='TensorRT precision mode (default: None - uses TorchScript without precision changes)')
    parser.add_argument('--tensorrt_workspace', type=int, default=1024,
                       help='TensorRT workspace size in MB (default: 1024)')
    return parser.parse_args()

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

def create_calibration_dataset(files_dir, folder_list, model_type, max_samples=100):
    """Create a calibration dataset for INT8 quantization"""
    calibration_data = []
    sample_count = 0
    
    for folder_name in folder_list[:5]:  # Use first 5 folders for calibration
        if sample_count >= max_samples:
            break
            
        try:
            dataset = SingleFolderDataset(files_dir, folder_name, model_type)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
            
            for batch in dataloader:
                if sample_count >= max_samples:
                    break
                    
                # Process data same as main inference
                vis_raw = batch['vis_raw'][0]
                vis_pixel_size = batch['vis_pixel_size'][0]
                
                if vis_raw is not None and vis_pixel_size is not None:
                    target_pixel_size = 0.69
                    scale_factor = vis_pixel_size / target_pixel_size
                    _, original_h, original_w = vis_raw.shape
                    new_h = int(original_h * scale_factor)
                    new_w = int(original_w * scale_factor)
                    img_V = torch.nn.functional.interpolate(
                        vis_raw.unsqueeze(0), size=(new_h, new_w),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                    img_V = torch.clamp(img_V, 0, 1)
                else:
                    ir = batch['ir']
                    batch_size, bands, ir_h, ir_w = ir.shape
                    ir_pixel_size = 2.0
                    vis_target_pixel_size = 0.69
                    scale_factor = ir_pixel_size / vis_target_pixel_size
                    vis_h = int(ir_h * scale_factor)
                    vis_w = int(ir_w * scale_factor)
                    img_V = torch.zeros((3, vis_h, vis_w))

                if model_type == 1:
                    ir = batch['ir'].type(torch.FloatTensor)
                    vis = img_V.unsqueeze(0).type(torch.FloatTensor)
                else:
                    ir = (batch['ir'].type(torch.FloatTensor) - 0.5) / 0.5
                    vis = (img_V.unsqueeze(0).type(torch.FloatTensor) - 0.5) / 0.5

                # Create sample patches for calibration
                fact = 1024
                channels = 13 if model_type == 1 else 10
                scaling_factor = 0.69/2
                
                sample_input = torch.zeros(1, channels, fact, fact)
                
                xx = ir[:, :, :int(fact * scaling_factor), :int(fact * scaling_factor)]
                xx = torch.nn.functional.interpolate(xx, size=(fact, fact), mode='bicubic', align_corners=True)
                
                if model_type == 1:
                    vv = vis[:, :, :fact, :fact]
                    if vv.shape[2] < fact or vv.shape[3] < fact:
                        pad_rows = fact - vv.shape[2] if vv.shape[2] < fact else 0
                        pad_cols = fact - vv.shape[3] if vv.shape[3] < fact else 0
                        vv = torch.nn.functional.pad(vv, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                    
                    sample_input[0, :ir.shape[1], :, :] = xx[0]
                    sample_input[0, ir.shape[1]:, :, :] = vv[0]
                else:
                    sample_input[0, :ir.shape[1], :, :] = xx[0]
                
                calibration_data.append(sample_input)
                sample_count += 1
                
        except Exception as e:
            print(f"Skipping calibration folder {folder_name}: {e}")
            continue
    
    return calibration_data

def create_calibration_dataset(files_dir, folder_list, model_type, max_samples=100):
    """Create a calibration dataset for INT8 quantization"""
    calibration_data = []
    sample_count = 0
    
    for folder_name in folder_list[:5]:  # Use first 5 folders for calibration
        if sample_count >= max_samples:
            break
            
        try:
            dataset = SingleFolderDataset(files_dir, folder_name, model_type)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
            
            for batch in dataloader:
                if sample_count >= max_samples:
                    break
                    
                # Process data same as main inference
                vis_raw = batch['vis_raw'][0]
                vis_pixel_size = batch['vis_pixel_size'][0]
                
                if vis_raw is not None and vis_pixel_size is not None:
                    target_pixel_size = 0.69
                    scale_factor = vis_pixel_size / target_pixel_size
                    _, original_h, original_w = vis_raw.shape
                    new_h = int(original_h * scale_factor)
                    new_w = int(original_w * scale_factor)
                    img_V = torch.nn.functional.interpolate(
                        vis_raw.unsqueeze(0), size=(new_h, new_w),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                    img_V = torch.clamp(img_V, 0, 1)
                else:
                    ir = batch['ir']
                    batch_size, bands, ir_h, ir_w = ir.shape
                    ir_pixel_size = 2.0
                    vis_target_pixel_size = 0.69
                    scale_factor = ir_pixel_size / vis_target_pixel_size
                    vis_h = int(ir_h * scale_factor)
                    vis_w = int(ir_w * scale_factor)
                    img_V = torch.zeros((3, vis_h, vis_w))

                if model_type == 1:
                    ir = batch['ir'].type(torch.FloatTensor)
                    vis = img_V.unsqueeze(0).type(torch.FloatTensor)
                else:
                    ir = (batch['ir'].type(torch.FloatTensor) - 0.5) / 0.5
                    vis = (img_V.unsqueeze(0).type(torch.FloatTensor) - 0.5) / 0.5

                # Create sample patches for calibration
                fact = 1024
                channels = 13 if model_type == 1 else 10
                scaling_factor = 0.69/2
                
                sample_input = torch.zeros(1, channels, fact, fact)
                
                xx = ir[:, :, :int(fact * scaling_factor), :int(fact * scaling_factor)]
                xx = torch.nn.functional.interpolate(xx, size=(fact, fact), mode='bicubic', align_corners=True)
                
                if model_type == 1:
                    vv = vis[:, :, :fact, :fact]
                    if vv.shape[2] < fact or vv.shape[3] < fact:
                        pad_rows = fact - vv.shape[2] if vv.shape[2] < fact else 0
                        pad_cols = fact - vv.shape[3] if vv.shape[3] < fact else 0
                        vv = torch.nn.functional.pad(vv, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                    
                    sample_input[0, :ir.shape[1], :, :] = xx[0]
                    sample_input[0, ir.shape[1]:, :, :] = vv[0]
                else:
                    sample_input[0, :ir.shape[1], :, :] = xx[0]
                
                # Remove batch dimension and store as tensor
                calibration_data.append(sample_input.squeeze(0))
                sample_count += 1
                
        except Exception as e:
            print(f"Skipping calibration folder {folder_name}: {e}")
            continue
    
    return calibration_data

def create_calibration_dataloader(calibration_data, batch_size=1):
    """Convert calibration data list to DataLoader"""
    if not calibration_data:
        raise ValueError("Calibration data is empty")
    
    # Stack all calibration samples into a single tensor
    calibration_tensor = torch.stack(calibration_data)
    
    # Create a TensorDataset and DataLoader
    calibration_dataset = TensorDataset(calibration_tensor)
    calibration_dataloader = DataLoader(
        calibration_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True
    )
    
    return calibration_dataloader

def convert_to_tensorrt(model, input_shape, precision='fp16', workspace_size=1024, calibration_data=None):
    """Convert PyTorch model to TensorRT optimized model"""
    start = time.time()
    print(f"Converting model to TensorRT with {precision} precision...")
    
    # Create dummy input for tracing
    dummy_input = torch.randn(input_shape).cuda()
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Set precision and calibration
    if precision == 'int8':
        if calibration_data is None:
            raise ValueError("Calibration data is required for INT8 precision")
        
        # Convert calibration data to DataLoader if it's a list
        if isinstance(calibration_data, list):
            calibration_dataloader = create_calibration_dataloader(calibration_data, batch_size=1)
        else:
            calibration_dataloader = calibration_data
        
        # Create calibrator
        calibrator = ptq.DataLoaderCalibrator(
            calibration_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0")
        )
        
        enabled_precisions = {torch.int8}
        int8_calibrator = calibrator
    elif precision == 'fp16':
        enabled_precisions = {torch.half}
        int8_calibrator = None
    else:
        enabled_precisions = {torch.float}
        int8_calibrator = None
    
    # Create TensorRT Device object
    device = torch_tensorrt.Device(gpu_id=0)
    
    # Convert to TensorRT
    compile_args = {
        "ir": "torchscript",
        "inputs": [torch_tensorrt.Input(
            min_shape=input_shape,
            opt_shape=input_shape,
            max_shape=input_shape,
            dtype=torch.float32
        )],
        "enabled_precisions": enabled_precisions,
        "workspace_size": workspace_size * 1024 * 1024,
        "truncate_long_and_double": True,
        "device": device
    }
    
    # Add calibrator for INT8
    if int8_calibrator is not None:
        compile_args["calibrator"] = int8_calibrator
    
    trt_model = torch_tensorrt.compile(traced_model, **compile_args)
    
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

class SingleFolderDataset(Dataset):
    def __init__(self, root, folder_name, model_type=0):
        self.parent = root
        self.folder_name = folder_name
        self.model_type = model_type
        self.is_subfolder_structure = self._check_subfolder_structure()
        
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

    def _validate_required_folders(self):
        """Validate that all required folders exist and contain valid data files"""
        # Validate IR folder (the main folder)
        ir_folder_path = os.path.join(self.parent, self.folder_name)
        
        if not self._validate_folder_contents(ir_folder_path):
            raise FileNotFoundError(f"Invalid IR folder or missing HDR/data files: {ir_folder_path}")
        
        # Validate VIS folder if model_type requires it
        if self.model_type == 1:
            vis_folder_path = os.path.join(self.parent, self.folder_name + '_VIS')
            if not self._validate_folder_contents(vis_folder_path):
                raise FileNotFoundError(f"Invalid VIS folder or missing HDR/data files: {vis_folder_path}")

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

    def __getitem__(self, index):
        start = time.time()
        
        # IR folder is the main folder
        ir_folder_path = os.path.join(self.parent, self.folder_name)
        
        # Load IR data
        img = self._load_ir_data(ir_folder_path)
        
        # Handle VIS data if needed
        vis_data = None
        vis_pixel_size = None
        
        if self.model_type == 1:
            vis_folder_path = os.path.join(self.parent, self.folder_name + '_VIS')
            vis_data, vis_pixel_size = self._load_vis_data(vis_folder_path)
        
        bsq_load_time = time.time() - start
        
        return {
            'ir': img[:,:,:],
            'vis_raw': vis_data,
            'vis_pixel_size': vis_pixel_size,
            'name': self.folder_name,
            'bsq_load_time': torch.tensor(bsq_load_time),
            'is_subfolder_structure': self.is_subfolder_structure
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
        vis_data = torch.from_numpy(vis_data).float()  # Fixed: was img_V

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



def main(opt, target_folder_path=None, parent_folder_path=None, model_type=0, use_tensorrt=False, 
         tensorrt_precision=None, use_warmup=False, tensorrt_workspace=1024, enable_profiler=False,
         profiler_wait=1, profiler_warmup=1, profiler_active=3, profiler_repeat=2): 
    """
    Main inference function with fixed PyTorch profiler support.
    """
    
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
    with record_function("model_initialization"):
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
        with record_function("tensorrt_optimization"):
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
                    calibration_data = None
                    if tensorrt_precision == 'int8':
                        print("Creating calibration dataset for INT8...")
                        calibration_data = create_calibration_dataset(files_dir, folder_list, model_type)
                        print(f"Created calibration dataset with {len(calibration_data)} samples")
                    
                    optimized_model = convert_to_tensorrt(netG, input_shape, tensorrt_precision, 
                                                        tensorrt_workspace, calibration_data)
                    save_optimized_model(optimized_model, model_path, model_type, "TensorRT", 
                                       tensorrt_precision, batch_size=actual_batch_size)
            
            netG = optimized_model

    if use_tensorrt or use_warmup:
        with record_function("model_warmup"):
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
    
    # Create profiler logs directory if it doesn't exist
    if enable_profiler:
        profiler_logs_dir = './profiler_logs'
        os.makedirs(profiler_logs_dir, exist_ok=True)
        print(f"Profiler enabled - logs will be saved to: {os.path.abspath(profiler_logs_dir)}")
        
        # Initialize profiler with simpler schedule for single folder
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0,      # Wait 1 step before starting
                warmup=0,    # Warmup for 2 steps
                active=1000,    # Profile for 5 steps
                repeat=1     # Only 1 cycle for single folder
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_logs_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    else:
        profiler_context = None
    
    step_count = 0
    
    if enable_profiler:
        profiler_context.__enter__()
    
    try:
        for folder_idx, folder_name in enumerate(folder_list):
            print(f"\nProcessing folder {folder_idx + 1}/{len(folder_list)}: {folder_name}")
            
            try:
                with record_function(f"folder_processing_{folder_name}"):
                    # Create single-folder dataset with model_type - let it handle validation
                    with record_function("dataset_creation"):
                        single_folder_dataset = SingleFolderDataset(files_dir, folder_name, model_type)
                        dataloader = DataLoader(single_folder_dataset, batch_size=opt['batch_size'], 
                                              shuffle=False, num_workers=opt['n_cpu'], collate_fn=custom_collate_fn)
                    
                    folder_start_time = time.time()
                    
                    for batch_idx, batch in enumerate(dataloader):
                        with record_function(f"batch_processing_{batch_idx}"):
                            torch.cuda.empty_cache()
                            
                            bsq_load_times.append(batch['bsq_load_time'].item())
                            inference_start = time.time()
                            
                            # Check if this is a single folder processing and we have an error
                            if target_folder_path and batch['vis_raw'][0] is None and model_type == 1:
                                print(f"Error: Required VIS data not found for model type 1")
                                return [], 0, 0
                            
                            # Process VIS data
                            with record_function("vis_data_processing"):
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
                                    batch_size_actual, bands, ir_h, ir_w = ir.shape
                                    
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

                            with record_function("data_preprocessing"):
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

                            with record_function("patch_processing_and_inference"):
                                for kk in range(b):
                                    patch_count = 0
                                    patches = []
                                    I = range(0, int(rn)*fact+1, fact)
                                    J = range(0, int(cn)*fact+1, fact)
                                    
                                    for ii in range(len(I)-1):
                                        for jj in range(len(J) - 1):
                                            with record_function("patch_extraction"):
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
                                                with record_function("model_inference"):
                                                    if use_tensorrt:
                                                        # Always use full batch size for TensorRT
                                                        if patch_count < BATCH_SIZE:
                                                            # Pad remaining slots with zeros (or duplicate last patch)
                                                            for pad_idx in range(patch_count, BATCH_SIZE):
                                                                DATA[pad_idx] = DATA[patch_count-1]  # Duplicate last patch
                                                        
                                                        output = netG(DATA).squeeze() * 0.5 + 0.5  # Use full DATA tensor
                                                        
                                                        # Only use the first patch_count outputs
                                                        output = output[:patch_count]
                                                    else:
                                                        output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5

                                                with record_function("output_processing"):
                                                    output = output.cpu().numpy()

                                                    for idx, (pi, pj) in enumerate(patches):
                                                        fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]

                                                patch_count = 0
                                                patches = []
                                                
                                                # Update profiler step after each inference
                                                if enable_profiler:
                                                    step_count += 1
                                                    profiler_context.step()

                                    if patch_count > 0:
                                        with record_function("final_batch_inference"):
                                            if use_tensorrt:
                                                # Always use full batch size for TensorRT
                                                if patch_count < BATCH_SIZE:
                                                    # Pad remaining slots with zeros or duplicate last patch
                                                    for pad_idx in range(patch_count, BATCH_SIZE):
                                                        DATA[pad_idx] = DATA[patch_count-1]  # Duplicate last patch
                                                
                                                output = netG(DATA).squeeze() * 0.5 + 0.5  # Use full DATA tensor
                                                
                                                # Only use the first patch_count outputs
                                                output = output[:patch_count]
                                            else:
                                                # For non-TensorRT models, can use variable batch size
                                                output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5

                                        with record_function("final_output_processing"):
                                            output = output.cpu().numpy() 
                                            
                                            for idx, (pi, pj) in enumerate(patches):
                                                fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]
                                        
                                        # Update profiler step after final inference
                                        if enable_profiler:
                                            step_count += 1
                                            profiler_context.step()
                                            print(f"Final profiler step: {step_count}")

                            fhr = fhr[:, :, :r, :c]
                            
                            inference_time = time.time() - inference_start
                            inference_times.append(inference_time)
                            
                            with record_function("file_writing"):
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
    
    finally:
        # Close profiler if it was enabled
        if profiler_context is not None:
            profiler_context.__exit__(None, None, None)
            print(f"Profiler results saved to: {os.path.abspath('./profiler_logs')}")
            print("View results with: tensorboard --logdir=./profiler_logs")
            
            # Also check if files were actually created
            profiler_files = os.listdir('./profiler_logs') if os.path.exists('./profiler_logs') else []
            print(f"Files in profiler_logs directory: {profiler_files}")
    
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

def profile_inference_only(netG, data_batch, device, fact, channels, scaling_factor, model_type, BATCH_SIZE, use_tensorrt=False):
    """
    Standalone function to profile just the inference portion with fine-grained control.
    Useful for detailed performance analysis of the core inference loop.
    """
    Tensor2 = torch.FloatTensor
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        with record_function("inference_only_profiling"):
            ir = data_batch['ir']
            vis_raw = data_batch['vis_raw'][0]
            vis_pixel_size = data_batch['vis_pixel_size'][0]
            
            # Process VIS data
            with record_function("vis_preprocessing"):
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
                    batch_size_actual, bands, ir_h, ir_w = ir.shape
                    ir_pixel_size = 2.0
                    vis_target_pixel_size = 0.69
                    scale_factor = ir_pixel_size / vis_target_pixel_size
                    
                    vis_h = int(ir_h * scale_factor)
                    vis_w = int(ir_w * scale_factor)
                    img_V = torch.zeros((3, vis_h, vis_w))

            with record_function("tensor_preparation"):
                if model_type == 1:
                    ir = ir.type(Tensor2)
                    vis = img_V.unsqueeze(0).type(Tensor2)
                else:
                    ir = (ir.type(Tensor2) - 0.5) / 0.5
                    vis = (img_V.unsqueeze(0).type(Tensor2) - 0.5) / 0.5
                
                _, bands, _, _ = ir.shape
                b, _, _, _ = vis.shape
                _, _, r, c = vis.shape
                rn = np.ceil(r/fact)
                cn = np.ceil(c/fact)
                
                DATA = torch.zeros(BATCH_SIZE, channels, fact, fact, device=device)

            with record_function("patch_extraction_and_inference"):
                for kk in range(b):
                    patch_count = 0
                    I = range(0, int(rn)*fact+1, fact)
                    J = range(0, int(cn)*fact+1, fact)
                    
                    for ii in range(len(I)-1):
                        for jj in range(len(J) - 1):
                            with record_function(f"patch_extract_{ii}_{jj}"):
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
                                    
                                patch_count += 1

                            if patch_count == BATCH_SIZE or (ii == len(I)-2 and jj == len(J)-2):
                                with record_function(f"model_forward_pass_{patch_count}"):
                                    if use_tensorrt:
                                        if patch_count < BATCH_SIZE:
                                            for pad_idx in range(patch_count, BATCH_SIZE):
                                                DATA[pad_idx] = DATA[patch_count-1]
                                        output = netG(DATA).squeeze() * 0.5 + 0.5
                                        output = output[:patch_count]
                                    else:
                                        output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5
                                
                                with record_function(f"output_transfer_{patch_count}"):
                                    output = output.cpu().numpy()
                                
                                patch_count = 0

    # Print profiler results
    print("=== DETAILED PROFILER RESULTS ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Save detailed trace
    prof.export_chrome_trace("inference_trace.json")
    print("Detailed trace saved to 'inference_trace.json'")
    print("Open in Chrome at chrome://tracing/")
    
    return prof

def warmup_tensorrt_model(model, input_shape, num_warmup=10):
    """Warmup TensorRT model for optimal performance"""
    start = time.time()
    print(f"Warming up TensorRT model with {num_warmup} iterations...")
    dummy_input = torch.randn(input_shape).cuda()
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    print("TensorRT model warmup completed in {:.2f} seconds".format(time.time() - start))



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
                   target_folder_path=args.folder, 
                   parent_folder_path=args.parent_folder, 
                   model_type=args.model, 
                   use_warmup=args.use_warmup,
                   use_tensorrt=args.use_tensorrt,
                   tensorrt_precision=args.tensorrt_precision,
                   tensorrt_workspace=args.tensorrt_workspace, enable_profiler= args.enable_profiler)
    