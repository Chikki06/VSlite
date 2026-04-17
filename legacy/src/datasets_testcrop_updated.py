import os
import numpy as np
import torch
from torch.utils.data import Dataset
import spectral
import torch.nn.functional as F
import time

class ImageDataset_biopsies(Dataset):
    def __init__(self, root, folder_name=None, model_type=0):
        self.parent = root
        self.folder_name = folder_name
        self.model_type = model_type
        
        if folder_name:
            # Single folder mode
            self.is_subfolder = self._check_subfolder_structure()
            self.list_I = [folder_name]
        else:
            # Batch mode - check all folders
            self.list_I = []
            for folder in os.listdir(self.parent):
                if not folder.endswith('_VIS'):
                    folder_path = os.path.join(self.parent, folder)
                    if os.path.isdir(folder_path):
                        self.list_I.append(folder)

    def _check_subfolder_structure(self):
        """Check if this is a subfolder structure (IR/VIS subfolders) or direct folder"""
        folder_path = os.path.join(self.parent, self.folder_name)
        ir_path = os.path.join(folder_path, 'IR')
        return os.path.exists(ir_path) and os.path.isdir(ir_path)

    def __getitem__(self, index):
        start = time.time()
        folder_name = self.list_I[index]
        
        # Determine the correct path based on folder structure
        if hasattr(self, 'is_subfolder') and self.is_subfolder:
            base_path = os.path.join(self.parent, folder_name, 'IR')
        else:
            base_path = os.path.join(self.parent, folder_name)

        # Find HDR file
        folder_files = os.listdir(base_path)
        hdr_file = next((f for f in folder_files if f.endswith('.hdr')), None)
        if not hdr_file:
            raise FileNotFoundError(f"No .hdr file found in {base_path}")

        # Get corresponding data file
        data_file = self._find_data_file(hdr_file.rsplit('.', 1)[0], folder_files)
        if not data_file:
            raise FileNotFoundError(f"No matching data file found for {hdr_file}")

        # Load image data
        image = spectral.envi.open(os.path.join(base_path, hdr_file), 
                                 os.path.join(base_path, data_file))
        img = np.array(image.load())
        img = np.transpose(img, (2, 0, 1))

        # Normalize image
        img = img[:,:,:].copy()
        img[img>4] = 4
        img[img<0] = 0
        img = img/1.5

        # Create empty tensor for VIS data
        _, r, d = img.shape
        scale_factor = 2 / 0.69
        r, d = int(r * scale_factor), int(d * scale_factor)
        img_V = torch.zeros((1, 1, r, d))

        img = torch.from_numpy(img).float()

        return {
            'ir': img[:,:,:],
            'vis': img_V[0,:],
            'name': folder_name
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
        return len(self.list_I)
