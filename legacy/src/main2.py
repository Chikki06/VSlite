import os
import time
import torch
import argparse
import numpy as np
import pyvips
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable

from DRB import AttGenerator_ResNet50_SkipCon
from datasets_testcrop_updated import ImageDataset_biopsies

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on hyperspectral images')
    parser.add_argument('--folder', type=str, default=None, 
                       help='Specific folder name to process (optional)')
    parser.add_argument('--parent_folder', type=str, default=None,
                       help='Parent directory containing multiple folders to process in batch (optional)')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--model_type', type=int, default=0, choices=[0, 1],
                       help='Model type: 0 for 10-channel (default), 1 for 13-channel with VIS')
    return parser.parse_args()

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
        return

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
    import json
    json_path = os.path.join(PARENT_DIR, '..', 'viewer', 'static', 'available.json')

    # Load existing list
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    # Format resolution string
    resolution_str = f"{width}x{height}"

    # Look for existing entry with same name and model_type
    for entry in data:
        if entry['name'] == filename and entry.get('model_type') == model_type:
            entry['resolution'] = resolution_str
            break
    else:
        # Add new entry if not found
        data.append({
            'name': filename,
            'resolution': resolution_str,
            'model_type': model_type
        })

    # Save updated data
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def process_image(netG, ir, vis, name, fact=1472, scaling_factor=0.69/2, batch_size=2):
    """Process a single image through the network"""
    _, bands, _, _ = ir.shape
    b, _, _ , _ = vis.shape
    _, _, r, c = vis.shape
    rn = np.ceil(r/fact)
    cn = np.ceil(c/fact)
    DATA = torch.zeros(batch_size, 10, fact, fact)

    fhr = torch.zeros(b, 3, int(rn)*fact+1, int(cn)*fact+1).data.cpu().numpy()
    total_time = 0
    num_iterations = 0

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

                DATA[patch_count, :bands, :, :] = xx
                patches.append((ii, jj))
                patch_count += 1

                if patch_count == batch_size:
                    start_time = time.time()
                    output = netG(DATA[:patch_count].cuda()).squeeze().data.cpu().numpy() * 0.5 + 0.5
                    for idx, (pi, pj) in enumerate(patches):
                        fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]
                    total_time += time.time() - start_time
                    num_iterations += 1
                    patch_count = 0
                    patches = []

        if patch_count > 0:  # Process remaining patches
            start_time = time.time()
            output = netG(DATA[:patch_count].cuda()).squeeze().data.cpu().numpy() * 0.5 + 0.5
            for idx, (pi, pj) in enumerate(patches):
                fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]
            total_time += time.time() - start_time
            num_iterations += 1

    average_time = total_time / num_iterations if num_iterations > 0 else 0
    print(f"Average time per operation: {average_time} seconds")
    
    fhr = np.uint8(np.transpose(fhr, (0, 2, 3, 1))*256)
    fhr = fhr[0, :r, :c, :]
    
    return fhr

def main():
    opt = parse_arguments()

    # Load model
    model_path = os.path.join(PARENT_DIR, 'saved_models', 
                             '15_net_G.pth' if opt.model_type == 1 else '20_net_G.pth')
    netG = AttGenerator_ResNet50_SkipCon()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        netG.load_state_dict(torch.load(model_path))
        netG.to(device)  # Use to(device) instead of cuda()
        torch.set_grad_enabled(False)
        netG.eval()

    # Set up the dataset based on arguments
    if opt.folder:
        dataset = ImageDataset_biopsies(
            root=opt.parent_folder or PARENT_DIR,
            folder_name=opt.folder,
            model_type=opt.model_type
        )
    else:
        dataset = ImageDataset_biopsies(
            root=opt.parent_folder or PARENT_DIR,
            model_type=opt.model_type
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu
    )

    for i, batch in enumerate(dataloader):
        ir = ((batch['ir'].float()) - 0.5) / 0.5
        vis = ((batch['vis'].float()) - 0.5) / 0.5
        name = batch['name'][0]

        # Process the image
        output = process_image(netG, ir, vis, name)

        # Create SZI and update JSON
        create_szi(output, name, opt.model_type)
        update_json(name, output.shape[0], output.shape[1], opt.model_type)

if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()
