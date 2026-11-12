"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import math
import os
import numpy as np
import torchvision.transforms as transforms
import torch
import random
import hashlib
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
        
        # Apply Gaussian noise (albumentations >=1.4: std_range is std as fraction of dtype max)
        if noise:
            v = max(0.0, float(noise_var_max))
            std_hi = min(1.0, math.sqrt(v) / 255.0) if v > 0 else 0.0
            transforms.append(A.GaussNoise(std_range=(0.0, std_hi), p=0.5))
        
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
        parser.set_defaults(label_nc=5 - 1)
        parser.set_defaults(batch_size=8)
        parser.set_defaults(no_input_semantics=True)
        parser.add_argument(
            '--num_ir_channels',
            type=int,
            default=10,
            help='Number of IR channels to use (default: 10, can be set to 4 for subset)',
        )
        parser.add_argument(
            '--ir_channel_indices',
            type=str,
            default='',
            help='Optional comma-separated list of IR channel indices to select (e.g. "0,8,9"). '
            'If set, overrides --num_ir_channels slicing.',
        )
        parser.add_argument(
            '--test_list_path',
            type=str,
            default='',
            help='Optional path to newline-separated sample ids (test.txt). '
            'If not set, will use dataroot/test.txt if present. '
            'Patches are assigned by sample prefix in <sample>_<y>_<x>.',
        )
        parser.add_argument(
            '--val_list_path',
            type=str,
            default='',
            help='Optional path to newline-separated sample ids (val.txt). '
            'If not set, will use dataroot/val.txt if present. '
            'Used only when phase==val. Samples in test.txt always take precedence. '
            'A present-but-empty val.txt means no val split (do not hash-split).',
        )
        parser.add_argument(
            '--train_list_path',
            type=str,
            default='',
            help='Optional path to newline-separated sample ids for phase==train. '
            'When set, training keeps only matching samples (exact id if line contains "-", '
            'else prefix). Applied after test/val exclusions.',
        )
        parser.add_argument(
            '--test_includes_val',
            action='store_true',
            default=False,
            help='If set, the test split is the union of test.txt and val.txt sample ids. '
            'Only affects phase=="test"; train and val behavior are unchanged.',
        )
        parser.add_argument(
            '--split_seed',
            type=int,
            default=1337,
            help='Seed for deterministic train/val split on non-test samples.',
        )
        parser.add_argument(
            '--split_fracs',
            type=str,
            default='0.9,0.05,0.05',
            help='Comma-separated split fractions train,val,test. '
            'When test.txt exists, test fraction is controlled by that file; the remaining (train+val) ratio is used.',
        )
        return parser

    def initialize(self, opt):
        self.opt = opt

        ir_dir = os.path.join(opt.dataroot, 'ir')
        if not os.path.isdir(ir_dir):
            raise ValueError(
                f'IR directory not found: {ir_dir}. Expected dataroot/ir with .npy patches.'
            )
        print(f'Using IR directory: {ir_dir}')
        
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

        # Optional: fixed test split based on sample ids from test.txt.
        # - opt.phase == "test": keep only patches from test samples
        # - otherwise (train/val): exclude all patches from test samples
        test_list_path = str(getattr(opt, "test_list_path", "") or "").strip()
        if test_list_path == "":
            candidate = os.path.join(opt.dataroot, "test.txt")
            test_list_path = candidate if os.path.isfile(candidate) else ""

        test_samples = None
        if test_list_path != "" and os.path.isfile(test_list_path):
            exact = set()
            prefixes = set()
            with open(test_list_path, "r", encoding="utf-8") as f:
                for raw in f.read().splitlines():
                    s = raw.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "-" in s:
                        exact.add(s)
                    else:
                        prefixes.add(s)
            if len(exact) == 0 and len(prefixes) == 0:
                test_samples = None
            else:
                test_samples = (exact, sorted(prefixes))

        # Preload val rules so the test filter can optionally include them
        # (when opt.test_includes_val is set, the "test" split is test.txt ∪ val.txt).
        # The same rules are reused for the train/val split block below.
        val_list_path_pre = str(getattr(opt, "val_list_path", "") or "").strip()
        if val_list_path_pre == "":
            v_candidate = os.path.join(opt.dataroot, "val.txt")
            val_list_path_pre = v_candidate if os.path.isfile(v_candidate) else ""

        val_samples_pre = None
        if val_list_path_pre != "" and os.path.isfile(val_list_path_pre):
            v_exact = set()
            v_prefixes = set()
            with open(val_list_path_pre, "r", encoding="utf-8") as f:
                for raw in f.read().splitlines():
                    s = raw.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "-" in s:
                        v_exact.add(s)
                    else:
                        v_prefixes.add(s)
            if len(v_exact) > 0 or len(v_prefixes) > 0:
                val_samples_pre = (v_exact, sorted(v_prefixes))

        if test_samples is not None:
            phase = str(getattr(opt, "phase", "train") or "train").strip().lower()
            test_includes_val = bool(getattr(opt, "test_includes_val", False))

            def _sample_id_from_path(p: str) -> str:
                base = os.path.splitext(os.path.basename(p))[0]
                return base.split("_", 1)[0]

            exact, prefixes = test_samples

            if phase == "test" and test_includes_val and val_samples_pre is not None:
                v_exact, v_prefixes = val_samples_pre
                exact_eff: set = set(exact) | set(v_exact)
                prefixes_eff: list = sorted(set(prefixes) | set(v_prefixes))
            else:
                exact_eff = exact
                prefixes_eff = prefixes

            def _in_test(sid: str) -> bool:
                if sid in exact_eff:
                    return True
                for pfx in prefixes_eff:
                    if sid.startswith(pfx):
                        return True
                return False

            keep_idx = []
            for i, p in enumerate(IR):
                sid = _sample_id_from_path(p)
                in_test = _in_test(sid)
                if (phase == "test" and in_test) or (phase != "test" and not in_test):
                    keep_idx.append(i)

            IR = [IR[i] for i in keep_idx]
            VIS = [VIS[i] for i in keep_idx]
            HE = [HE[i] for i in keep_idx]

        # Validation split:
        # - If val.txt exists (or --val_list_path), use it to define *val by sample id/prefix*.
        #   A present-but-empty file is an explicit empty val set (no hash-split).
        # - Otherwise fall back to deterministic hash subset of non-test samples.
        # This keeps:
        #   - train: all non-test and non-val samples
        #   - val: val-listed samples (or hash-subset if no list file)
        #   - test: test-listed samples
        phase = str(getattr(opt, "phase", "train") or "train").strip().lower()
        if phase in {"train", "val"}:
            # Load val list if present.
            val_list_path = str(getattr(opt, "val_list_path", "") or "").strip()
            if val_list_path == "":
                candidate = os.path.join(opt.dataroot, "val.txt")
                val_list_path = candidate if os.path.isfile(candidate) else ""

            val_samples = None
            val_list_present = bool(val_list_path != "" and os.path.isfile(val_list_path))
            if val_list_present:
                v_exact = set()
                v_prefixes = set()
                with open(val_list_path, "r", encoding="utf-8") as f:
                    for raw in f.read().splitlines():
                        s = raw.strip()
                        if not s or s.startswith("#"):
                            continue
                        if "-" in s:
                            v_exact.add(s)
                        else:
                            v_prefixes.add(s)
                val_samples = (v_exact, sorted(v_prefixes))

            if val_list_present:
                def _sample_id_from_path(p: str) -> str:
                    base = os.path.splitext(os.path.basename(p))[0]
                    return base.split("_", 1)[0]

                v_exact, v_prefixes = val_samples

                def _in_val(sid: str) -> bool:
                    if sid in v_exact:
                        return True
                    for pfx in v_prefixes:
                        if sid.startswith(pfx):
                            return True
                    return False

                keep_idx = []
                for i, p in enumerate(IR):
                    sid = _sample_id_from_path(p)
                    in_val = _in_val(sid)
                    if (phase == "val" and in_val) or (phase == "train" and not in_val):
                        keep_idx.append(i)
                IR = [IR[i] for i in keep_idx]
                VIS = [VIS[i] for i in keep_idx]
                HE = [HE[i] for i in keep_idx]
            elif phase == "val":
                # Fallback: deterministic hash split when no val.txt is provided.
                # Parse split_fracs.
                raw_fracs = str(getattr(opt, "split_fracs", "0.9,0.05,0.05") or "0.9,0.05,0.05")
                try:
                    parts = [float(x.strip()) for x in raw_fracs.split(",")]
                    if len(parts) != 3:
                        raise ValueError
                    f_train, f_val, _f_test = parts
                except Exception:
                    f_train, f_val, _f_test = 0.9, 0.05, 0.05
                tv = float(f_train + f_val)
                # avoid div-by-zero; fall back to 95/5
                train_cut = float(f_train) / tv if tv > 1e-9 else 0.95
                seed = int(getattr(opt, "split_seed", 1337))

                def _hash01(s: str) -> float:
                    h = hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()
                    return int(h[:8], 16) / float(0xFFFFFFFF)

                keep_idx = []
                for i, p in enumerate(IR):
                    # use basename (stable across machines) rather than full path
                    key = os.path.basename(p)
                    r = _hash01(key)
                    bucket = "train" if r < train_cut else "val"
                    if bucket == "val":
                        keep_idx.append(i)
                IR = [IR[i] for i in keep_idx]
                VIS = [VIS[i] for i in keep_idx]
                HE = [HE[i] for i in keep_idx]

        # Optional train allowlist (organ experiments, etc.). Applied last so it
        # intersects with whatever remains after test/val exclusions.
        phase = str(getattr(opt, "phase", "train") or "train").strip().lower()
        train_list_path = str(getattr(opt, "train_list_path", "") or "").strip()
        if phase == "train" and train_list_path != "" and os.path.isfile(train_list_path):
            t_exact = set()
            t_prefixes = set()
            with open(train_list_path, "r", encoding="utf-8") as f:
                for raw in f.read().splitlines():
                    s = raw.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "-" in s:
                        t_exact.add(s)
                    else:
                        t_prefixes.add(s)
            if len(t_exact) > 0 or len(t_prefixes) > 0:
                t_prefixes_sorted = sorted(t_prefixes)

                def _sample_id_from_path(p: str) -> str:
                    base = os.path.splitext(os.path.basename(p))[0]
                    return base.split("_", 1)[0]

                def _in_train(sid: str) -> bool:
                    if sid in t_exact:
                        return True
                    for pfx in t_prefixes_sorted:
                        if sid.startswith(pfx):
                            return True
                    return False

                keep_idx = []
                for i, p in enumerate(IR):
                    if _in_train(_sample_id_from_path(p)):
                        keep_idx.append(i)
                IR = [IR[i] for i in keep_idx]
                VIS = [VIS[i] for i in keep_idx]
                HE = [HE[i] for i in keep_idx]

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
            
            I_name = self.list_IR[index % len(self.list_IR)]
            # Use mmap to reduce page-cache pressure and speed random access on network FS.
            img = np.load(I_name, mmap_mode='r')
          #  img = np.transpose(img, (2, 0, 1))
            ir_data = img[:, :, :]

            # IR channel selection: either explicit indices or first N channels.
            num_ir_channels = getattr(self.opt, 'num_ir_channels', 10)
            raw_idx = getattr(self.opt, 'ir_channel_indices', '')
            raw_idx = str(raw_idx).strip() if raw_idx is not None else ''
            if raw_idx:
                try:
                    idx = [int(x.strip()) for x in raw_idx.split(',') if x.strip() != '']
                except Exception:
                    raise ValueError(f'Invalid --ir_channel_indices={raw_idx!r}. Expected comma-separated ints like "0,8,9".')
                if not idx:
                    raise ValueError(f'Invalid --ir_channel_indices={raw_idx!r}. No indices parsed.')
                c_tot = int(ir_data.shape[2])
                bad = [i for i in idx if i < 0 or i >= c_tot]
                if bad:
                    raise ValueError(f'--ir_channel_indices out of range for IR channels={c_tot}: {bad}')
                ir_data = ir_data[:, :, idx]
                num_ir_channels = len(idx)
            else:
                ir_data = ir_data[:, :, :num_ir_channels]

            # VIS-only run (input_nc=3, num_ir_channels=0): no IR bands. Keep a
            # 1ch dummy so resize/augmentation geometry stays synced with VIS/HE;
            # it is dropped when the combined tensor is built below.
            vis_only = (int(getattr(self.opt, 'input_nc', 13)) == 3
                        and int(num_ir_channels) == 0)
            if vis_only:
                ir_data = np.zeros((ir_data.shape[0], ir_data.shape[1], 1), dtype=np.float32)

            # fp16 on-disk IR (IR_fp16) — promote before clip/resize for OpenCV.
            if getattr(ir_data, 'dtype', None) != np.float32:
                ir_data = np.asarray(ir_data, dtype=np.float32)
            ir_data = np.clip(ir_data, 0, 5)
            img = ir_data

            # Resize patches to a configurable size (defaults to historical 1024).
            # Several training scripts (e.g. VQGAN) expect smaller patches like 256.
            sz = getattr(self.opt, 'image_size', 1024)
            try:
                sz_int = int(sz)
            except Exception:
                sz_int = 1024
            target_size = (sz_int, sz_int)

            # Resize all channels in one call (much faster than per-channel Python loops).
            # OpenCV drops the channel axis for single-channel arrays (H,W,1)->(H,W).
            img_upsampled = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
            if img_upsampled.ndim == 2:
                img_upsampled = img_upsampled[:, :, np.newaxis]



            V_name = self.list_VIS[index % len(self.list_VIS)]
            img_V = np.load(V_name, mmap_mode='r')
            img_V_upsampled = cv2.resize(img_V, target_size, interpolation=cv2.INTER_CUBIC)
           # img_V = np.transpose(img_V, (2, 0, 1))

            no_aug = bool(getattr(self.opt, 'no_aug', False))
            if not no_aug:
                img_V = self.rgb_augmenter.forward(img_V_upsampled) / 256
            else:
                img_V = img_V_upsampled / 256.0
            H_name = self.list_HE[index % len(self.list_HE)]
            img_H = np.load(H_name, mmap_mode='r')
            img_H_upsampled = cv2.resize(img_H, target_size, interpolation=cv2.INTER_CUBIC)
            if not no_aug:
                img_upsampled, img_V, img_H = self.Allaug.forward(img_upsampled, img_V, img_H_upsampled)
            else:
                img_H = img_H_upsampled
            img_H = img_H / 256
            img_H = np.clip(img_H, 0, 1)
            he_space = str(getattr(self.opt, 'he_color_space', 'srgb')).lower()
            if he_space == 'linear':
                from util.color_space import srgb_to_linear

                # Plan C1: sRGB→linear on VIS and H&E before [-1,1].
                # Keep CHW torch tensors (Allaug already returns CHW); never HWC —
                # HWC targets break D cat (fake BCHW vs real BHWC → size 3 vs 1024).
                def _as_chw01(arr):
                    if isinstance(arr, torch.Tensor):
                        t = arr.float()
                        if t.ndim == 3 and t.shape[0] in (1, 3):
                            return t
                        if t.ndim == 3 and t.shape[-1] in (1, 3):
                            return t.permute(2, 0, 1).contiguous()
                        return t
                    a = np.asarray(arr, dtype=np.float32)
                    if a.ndim == 3 and a.shape[-1] in (1, 3):
                        a = a.transpose(2, 0, 1)
                    return torch.from_numpy(np.ascontiguousarray(a))

                img_H = srgb_to_linear(_as_chw01(img_H), minus1_1=False)
                img_H = img_H.clamp(0.0, 1.0) * 2.0 - 1.0
                img_V = srgb_to_linear(_as_chw01(img_V), minus1_1=False).clamp(0.0, 1.0)
            else:
                img_H = (img_H - 0.5) / 0.5
                if isinstance(img_H, np.ndarray) and img_H.ndim == 3 and img_H.shape[-1] in (1, 3):
                    img_H = torch.from_numpy(
                        np.ascontiguousarray(img_H, dtype=np.float32)
                    ).permute(2, 0, 1)
                elif isinstance(img_H, np.ndarray):
                    img_H = torch.from_numpy(np.ascontiguousarray(img_H, dtype=np.float32))


        except Exception as e:
            # Skip corrupt/unreadable samples, but never recurse forever (Vis+1
            # previously OOM'd when every sample failed the same way).
            nxt = (int(index) + 1) % max(1, len(self.list_IR))
            depth = int(getattr(self, "_getitem_retry_depth", 0)) + 1
            if depth >= min(32, max(1, len(self.list_IR))):
                raise RuntimeError(
                    f"VS_dataset.__getitem__ failed after {depth} retries "
                    f"(start_index={index}, last_error={type(e).__name__}: {e})"
                ) from e
            self._getitem_retry_depth = depth
            try:
                return self.__getitem__(nxt)
            finally:
                self._getitem_retry_depth = 0

        # Channel policy must match generator input_nc (see AttUnet first conv).
        # IR-only: input_nc == selected_num_ir_channels (e.g. 10 or 3).
        # IR+VIS:  input_nc == selected_num_ir_channels + 3 (e.g. 13 or 6).
        # Any other pair used to fall through to IR+VIS and produced 13ch tensors while opt.input_nc stayed 3.
        def _ensure_chw_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            a = np.asarray(x, dtype=np.float32)
            if a.ndim == 3 and a.shape[0] not in (1, 3, 10, 13) and a.shape[-1] in (1, 3, 10, 13):
                a = a.transpose(2, 0, 1)
            return torch.from_numpy(np.ascontiguousarray(a))

        img_upsampled = _ensure_chw_tensor(img_upsampled)
        if not isinstance(img_V, torch.Tensor) or (
            isinstance(img_V, torch.Tensor) and img_V.ndim == 3 and img_V.shape[0] not in (1, 3)
            and img_V.shape[-1] in (1, 3)
        ):
            img_V = _ensure_chw_tensor(img_V)
        else:
            img_V = img_V.float()
        img_H = _ensure_chw_tensor(img_H)

        input_nc = getattr(self.opt, 'input_nc', 13)
        # num_ir_channels may be overridden by --ir_channel_indices above; infer from tensor if needed.
        num_ir_channels = int(img_upsampled.shape[0]) if hasattr(img_upsampled, "shape") else getattr(self.opt, 'num_ir_channels', 10)
        if int(input_nc) == 3 and int(getattr(self.opt, 'num_ir_channels', 10)) == 0:
            # VIS-only: drop the 1ch aug dummy, feed VIS RGB alone (c13-style net, 3ch in).
            combined_data = img_V.float()
        elif input_nc == num_ir_channels:
            combined_data = img_upsampled.float()
        elif input_nc == num_ir_channels + 3:
            combined_data = torch.cat([img_upsampled.float(), img_V.float()], dim=0)
        else:
            raise ValueError(
                f'VS_dataset: input_nc={input_nc} with num_ir_channels={num_ir_channels} is invalid. '
                f'Use input_nc={num_ir_channels} for IR-only or input_nc={num_ir_channels + 3} for IR+VIS.'
            )

        I_name = self.list_IR[index % len(self.list_IR)]
        img_T = img_H

        return {
            'input': combined_data,
            'target': img_T,
            'name': I_name,
            'image': img_T,
            'label': combined_data,
            'path': I_name,
        }



    def __len__(self):
        return self.dataset_size





