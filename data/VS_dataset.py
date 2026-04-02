"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import numpy as np
import torchvision.transforms as transforms
import torch
import random
from data.AffineT import *
import spectral
import tifffile as tif
import cv2
import albumentations as A
# from IPython.core.debugger import set_trace
# import sys

# sys.stdout = sys.stderr


class GeoTransform:
    def __init__(self, transform=None):
        """
        Initializes the transformation class.
        
        Args:
            transform (albumentations.Compose): A transformation function to apply the same transformations to all images.
        """
        self.transform = transform

    def forward(self, ir, vis, he):
        """
        Applies the same transformation to the IR, VIS, and HE images.
        
        Args:
            ir (numpy.ndarray): IR image of shape (W, H, 10).
            vis (numpy.ndarray): VIS image of shape (W, H, 3).
            he (numpy.ndarray): HE image of shape (W, H, 3).
            
        Returns:
            tuple: Transformed IR, VIS, and HE images as torch tensors.
        """
        # Apply the same transformation to all images
        if self.transform:
            augmented = self.transform(image=ir, image0=vis, image1=he)
            ir = augmented['image']
            vis = augmented['image0']
            he = augmented['image1']
        
        # Convert to torch tensors
        ir = torch.tensor(ir, dtype=torch.float32).permute(2, 0, 1)  # (W, H, 10) -> (10, W, H)
        vis = torch.tensor(vis, dtype=torch.float32).permute(2, 0, 1)  # (W, H, 3) -> (3, W, H)
        he = torch.tensor(he, dtype=torch.float32).permute(2, 0, 1)  # (W, H, 3) -> (3, W, H)
        
        return ir, vis, he

class RGBaug:
    def __init__(self, 
                 jitters=False,
                 noise=True,
                 blur=True,
                 defocus=True,
                 motion=True,
                 zoom=False,
                 gamma=True,  # Add gamma augmentation
                 brightness_max=0.4,
                 contrast_max=-0.4,
                 hue_max=20,  # Add hue control
                 saturation_max=30,  # Add saturation control
                 value_max=0.4,  # Value control (for brightness)
                 gamma_limit=(80, 120),  # Default gamma range (0.8 to 1.2)
                 noise_var_max=0.1,
                 blur_kernel=(3, 5),
                 zoom_factor=(1, 1.05),
                 original_size=(256, 256)):  # Add original size for resizing
        """
        Initializes the augmentation class for RGB images.
        
        Args:
            jitters (bool): Whether to apply brightness/contrast jittering and color changes.
            noise (bool): Whether to apply Gaussian noise.
            blur (bool): Whether to apply general blurring.
            defocus (bool): Whether to apply defocus blur.
            motion (bool): Whether to apply motion blur.
            zoom (bool): Whether to apply random zoom.
            gamma (bool): Whether to apply random gamma augmentation.
            brightness_max (float): Maximum brightness factor for jittering.
            contrast_max (float): Maximum contrast factor for jittering.
            hue_max (int): Maximum hue shift for color jittering.
            saturation_max (int): Maximum saturation change for color jittering.
            value_max (float): Maximum value (brightness) change for color jittering.
            gamma_limit (tuple): Range for gamma correction, usually between 0.5 to 1.5 (expressed as percentages here).
            noise_var_max (float): Maximum variance for Gaussian noise.
            blur_kernel (tuple): Kernel size for blurring.
            zoom_factor (tuple): Range for zoom scaling.
            original_size (tuple): Size to resize the image back to after transformations.
        """
        self.original_size = original_size
        self.transform = self.build_transforms(jitters, noise, blur, defocus, motion, zoom, gamma,
                                               brightness_max, contrast_max, hue_max, saturation_max, value_max,
                                               gamma_limit, noise_var_max, blur_kernel, zoom_factor)
    
    def build_transforms(self, 
                         jitters, noise, blur, defocus, motion, zoom, gamma,
                         brightness_max, contrast_max, hue_max, saturation_max, value_max,
                         gamma_limit, noise_var_max, blur_kernel, zoom_factor):
        """
        Build the transformation pipeline.
        
        Args:
            All args are passed from the constructor to define augmentations.
        
        Returns:
            albumentations.Compose: A composed set of augmentations.
        """
        transforms = []
        
        # Apply brightness/contrast jittering and color jittering
        if jitters:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=brightness_max, contrast_limit=contrast_max, p=.5))
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=hue_max, 
                sat_shift_limit=saturation_max, 
                val_shift_limit=value_max, p=.5))
        
        # Apply gamma correction
        if gamma:
            transforms.append(A.RandomGamma(gamma_limit=gamma_limit, p=.5))
        
        # Apply Gaussian noise
        if noise:
            transforms.append(A.GaussNoise(var_limit=(0, noise_var_max), p=0.5))
        
        # Apply blurring
        if blur:
            transforms.append(A.Blur(blur_limit=blur_kernel, p=0.5))
        
        # Apply defocus blur (use 'radius' instead of 'blur_limit')
        if defocus:
            transforms.append(A.Defocus(radius=(1.5, 3), p=0.5))
        
        # Apply motion blur
        if motion:
            transforms.append(A.MotionBlur(blur_limit=blur_kernel, p=0.5))
        
        # Apply zoom
        if zoom:
            transforms.append(A.RandomScale(scale_limit=zoom_factor, p=0.5))

        # Ensure that the image is resized back to the original size
        transforms.append(A.Resize(height=self.original_size[0], width=self.original_size[1]))
        
        # Compose all transforms
        return A.Compose(transforms)

    def forward(self, image):
        """
        Apply augmentations to the input RGB image.
        
        Args:
            image (numpy.ndarray): An input RGB image of shape (W, H, 3).
        
        Returns:
            torch.Tensor: Transformed image as a PyTorch tensor of shape (3, W, H).
        """
        # Apply the transformations
      #  if self.transform:
       #     augmented = self.transform(image=image)
        #    image = augmented['image']
        
        # Convert to torch tensor and permute to (C, W, H)
      #  image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (W, H, 3) -> (3, W, H)
        
        return image



class VSDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(label_nc=5-1)
        parser.set_defaults(batch_size=8)
        parser.add_argument(
            '--num_ir_channels',
            type=int,
            default=10,
            help='Number of IR channels to use (default: 10, can be set to 4 for subset)'
        )
        return parser

    def initialize(self, opt):
        self.opt = opt
        
        # Try different possible IR directory paths
        possible_ir_dirs = [
            os.path.join(opt.dataroot, "ir"),  # Direct ir folder in dataroot
            os.path.join(opt.dataroot, "data_same_batch_final_1024", "ir"),  # Nested structure
            os.path.join("dataset", "data_same_batch_final_1024", "ir"),  # Local dataset folder
            os.path.join("/mnt/VSlite/dataset", "data_same_batch_final_1024", "ir")  # Absolute local path
        ]
        
        ir_dir = None
        for possible_dir in possible_ir_dirs:
            if os.path.exists(possible_dir):
                ir_dir = possible_dir
                break
        
        if ir_dir is None:
            raise ValueError(f"IR directory not found. Tried paths: {possible_ir_dirs}")
        
        print(f"Using IR directory: {ir_dir}")
        
        # Get all .npy files from the IR directory
        ir_files = [f for f in os.listdir(ir_dir) if f.endswith('.npy')]
        ir_files.sort()  # Sort for consistency
        
        # Create full paths for IR files
        IR = [os.path.join(ir_dir, f) for f in ir_files]

        # Generate corresponding VIS and HE paths
        VIS = [x.replace("/ir/", "/vis/") for x in IR]
        HE = [x.replace("/ir/", "/he/") for x in IR]

        IR = IR[: opt.max_dataset_size]
        VIS = VIS[: opt.max_dataset_size]
        HE = HE[: opt.max_dataset_size]

        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5)
        ], additional_targets={'image0': 'image', 'image1': 'image'})

        self.Allaug = GeoTransform(transform=transform)

        self.rgb_augmenter = RGBaug(
            jitters=True,
            noise=True,
            blur=True,
            defocus=True,
            motion=True,
            zoom=False,
            gamma=True,
            brightness_max=.2,
            contrast_max=0.2,
            hue_max=1000,
            saturation_max=20,
            value_max=0.9,
            gamma_limit=(30, 160),
            noise_var_max=0.1,
            blur_kernel=(3, 5),
            zoom_factor=(1, 1.05))

        self.list_IR = IR
        self.list_VIS = VIS
        self.list_HE = HE

        size = len(self.list_IR)
        self.dataset_size = size


    def __getitem__(self, index):
        
        try:
            
            I_name = os.path.join(self.list_IR[index%len(self.list_IR)])
            img = np.load(I_name)
          #  img = np.transpose(img, (2, 0, 1))
            ir_data = img[:, :, :]

            # Use only specified number of IR channels
            num_ir_channels = getattr(self.opt, 'num_ir_channels', 10)
            ir_data = ir_data[:, :, :num_ir_channels]

            ir_data[ir_data>5] = 5
            ir_data[ir_data<0] = 0
            img = ir_data/1

            target_size = (1024, 1024)

            img_upsampled = np.zeros((target_size[1], target_size[0], img.shape[2]), dtype=img.dtype)

            for channel in range(img.shape[2]):
                img_upsampled[:, :, channel] = cv2.resize(img[:, :, channel], target_size, interpolation=cv2.INTER_CUBIC)



            V_name = os.path.join(self.list_VIS[index%len(self.list_VIS)])
            img_V = np.load(V_name)#/256
            img_V_upsampled = np.zeros((target_size[1], target_size[0], img_V.shape[2]), dtype=img_V.dtype)

            for channel in range(img_V.shape[2]):
                img_V_upsampled[:, :, channel] = cv2.resize(img_V[:, :, channel], target_size, interpolation=cv2.INTER_CUBIC)
           # img_V = np.transpose(img_V, (2, 0, 1))

            img_V = self.rgb_augmenter.forward(img_V_upsampled)/256  
           # print(img_upsampled.shape)
            H_name = os.path.join(self.list_HE[index%len(self.list_HE)])
            img_H = np.load(H_name)#/256
            img_H_upsampled = np.zeros((target_size[1], target_size[0], img_H.shape[2]), dtype=img_H.dtype)
            for channel in range(img_H.shape[2]):
                img_H_upsampled[:, :, channel] = cv2.resize(img_H[:, :, channel], target_size, interpolation=cv2.INTER_CUBIC)
            img_upsampled, img_V, img_H= self.Allaug.forward(img_upsampled,img_V,img_H_upsampled)
            img_H = img_H/256
            img_H[img_H>1] = 1
            img_H[img_H<0] = 0
            img_H = (img_H - .5)/.5



            trans1 = transforms.ToTensor()


        except Exception as e:
            return self.__getitem__((index + 1) % len(self.list_IR))

        # Determine whether to include VIS channels based on input_nc
        input_nc = getattr(self.opt, 'input_nc', 13)
        num_ir_channels = getattr(self.opt, 'num_ir_channels', 10)
        
        # If input_nc equals num_ir_channels, use IR only (no VIS)
        # Otherwise, concatenate IR and VIS
        if input_nc == num_ir_channels:
            # IR only mode
            combined_data = img_upsampled
        else:
            # IR + VIS mode (default)
            combined_data = np.concatenate([img_upsampled, img_V], axis=0)

        combined_data = ((torch.from_numpy(combined_data).float()) - 0)/1

        name = self.list_IR[index%len(self.list_IR)];
        
        
        img_T = img_H

        return {'input': combined_data[:,:,:], 'target': img_T, 'name': name}



    def __len__(self):
        return self.dataset_size





