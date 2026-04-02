import torch
import torch.onnx
import os
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

# Model classes (copied from your original code)
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: dict) -> None:
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

def detect_input_channels(model_path):
    """
    Detect the number of input channels from the saved model
    
    Args:
        model_path: Path to PyTorch model (.pth file)
        
    Returns:
        int: Number of input channels (10 or 13)
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        conv1_weight = checkpoint['encoder.conv1.weight']
        input_channels = conv1_weight.shape[1]
        print(f"Detected {input_channels} input channels from model")
        return input_channels
    except Exception as e:
        print(f"Warning: Could not detect input channels automatically: {e}")
        return None

def convert_pytorch_to_onnx(model_path, model_type=None, output_path=None):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to PyTorch model (.pth file)
        model_type: 0 for 10-channel, 1 for 13-channel, None for auto-detect
        output_path: Path to save ONNX model (optional)
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Auto-detect input channels if not specified
    if model_type is None:
        input_channels = detect_input_channels(model_path)
        if input_channels is None:
            raise ValueError("Could not auto-detect input channels. Please specify --model-type")
    else:
        input_channels = 13 if model_type == 1 else 10
    
    # Validate input channels
    if input_channels not in [10, 13]:
        raise ValueError(f"Unsupported number of input channels: {input_channels}. Expected 10 or 13.")
    
    print(f"Using {input_channels} input channels")
    
    # Set up model
    model = AttGenerator_ResNet50_SkipCon(input_channels=input_channels)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    except RuntimeError as e:
        if "size mismatch" in str(e):
            detected_channels = detect_input_channels(model_path)
            if detected_channels:
                raise ValueError(f"Model has {detected_channels} input channels, but you specified model_type for {input_channels} channels. Use --model-type {1 if detected_channels == 13 else 0}")
        raise e
    
    # Create dummy input
    batch_size = 1
    input_size = 1024 if model_type == 1 else 1472
    dummy_input = torch.randn(batch_size, input_channels, input_size, input_size)
    
    # Set output path
    if output_path is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{model_name}.onnx"
    
    # Export to ONNX
    print(f"Converting PyTorch model to ONNX...")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully converted to ONNX: {output_path}")
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX')
    parser.add_argument('model_path', help='Path to PyTorch model (.pth file)')
    parser.add_argument('--model-type', type=int, default=None, choices=[0, 1], 
                      help='Model type: 0 for 10-channel, 1 for 13-channel (auto-detect if not specified)')
    parser.add_argument('--output', help='Output path for ONNX model (optional)')
    
    args = parser.parse_args()
    
    try:
        convert_pytorch_to_onnx(args.model_path, args.model_type, args.output)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()