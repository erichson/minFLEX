

import numpy as np
import torch
import h5py
import os
import torch.nn.functional as F
import random
from torchvision import datasets, transforms
from abc import ABC, abstractmethod


    
class PatchDataset(torch.utils.data.Dataset, ABC):
    """
    SAFE for num_workers > 0 (lazy HDF-5 handles).
    """
    def __init__(self, factor=8, patch_size=128, stride=64, oversampling=40):
        self.factor          = factor
        self.oversampling    = oversampling
        self.patch_size      = patch_size
        self.stride          = stride

        # defer heavy work:
        self.paths = self.build_file()

        # discover data shape from a *temporary* handle
        with h5py.File(self.paths, "r") as f:
            self.data_shape = f['fields'].shape

        self.max_row = (self.data_shape[-2] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[-1] - self.patch_size) // self.stride + 1
        self.mean, self.std = self.get_norm()

        self._datasets = None          # will hold per-process handles

    # ------------------------------------------------------------------ #
    # ---------- abstract helpers to be implemented by subclass --------- #
    
    @abstractmethod
    def build_file(self): ...

    @abstractmethod
    def get_norm(self): ...

    # ---------- lazy opener ------------------------------------------- #
    def _ensure_open(self):
        if self._datasets is None:     # first touch *in this process*
            self._datasets = h5py.File(self.paths, "r", libver="latest", swmr=True)

    # ------------------------------------------------------------------ #
    def __len__(self):
        #print((self.data_shape[0]-20) * self.oversampling)
        return (self.data_shape[0]-20) * self.oversampling
    
    def normalize(self, x):   return (x - self.mean) / self.std
    def undo_norm(self, x):   return x * self.std + self.mean

    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        self._ensure_open()
        index = index // self.oversampling

        row = np.random.randint(0, self.max_row) * self.stride
        col = np.random.randint(0, self.max_col) * self.stride

        dataset  = self._datasets['fields']

        if len(self.data_shape) == 4:
            patch = dataset[index : index + 1, 0, row:row+self.patch_size, col:col+self.patch_size]
        else:
            patch = dataset[index : index + 1, row:row+self.patch_size, col:col+self.patch_size]

        patch  = torch.from_numpy(patch).float()

        #patch  = self.normalize(patch)

        lowres = patch[None, :, ::self.factor, ::self.factor]
        lowres = F.interpolate(lowres, size=patch.shape[1:], mode="bilinear")[0, -1:]

        return lowres, patch, torch.tensor(0).unsqueeze(0)


        
class ERA5(PatchDataset):
    def __init__(self, factor, patch_size=128, stride=32, scratch_dir="./", oversampling=20):
        self.scratch_dir = scratch_dir
        super().__init__(factor, patch_size, stride, oversampling)

    def build_file(self):
        return os.path.join(self.scratch_dir, "2013_subregion.h5")

    def get_norm(self):
        return 0.0, 1.0




class EvalLoader(torch.utils.data.Dataset, ABC):
    """
    Lazy-HDF5, safe with num_workers > 0.
    If snapshot_idx is None, we sweep through all timesteps;
    otherwise we always use that single snapshot.
    """
    def __init__(self,
                 factor: int,
                 patch_size: int = 128,
                 stride: int = 128,
                 scratch_dir: str = './',
                 snapshot_idx: int = None):
        self.factor         = factor
        self.patch_size     = patch_size
        self.stride         = stride
        self.snapshot_idx   = snapshot_idx

        # path to your saved subregion HDF5
        self.path = os.path.join(scratch_dir, '2013_subregion.h5')

        # peek at shape without holding the file open
        with h5py.File(self.path, 'r') as f:
            # assume f['fields'] is either (T, H, W) or (T, C, H, W)
            shp = f['fields'].shape

        # we only handle the 3- or 4-dim case
        if len(shp) == 3:
            # (T, H, W)
            self.has_channel = False
            T, H, W = shp
            C = 1
        else:
            # (T, C, H, W)
            self.has_channel = True
            T, C, H, W = shp

        self.T, self.C, self.H, self.W = T, C, H, W

        # how many patches fit per image?
        self.max_row = (H - patch_size) // stride + 1
        self.max_col = (W - patch_size) // stride + 1
        self.patches_per_image = self.max_row * self.max_col

        # normalization stats
        self.mean, self.std = self.get_norm()

        # will hold h5py.Dataset, opened in each worker
        self._fields = None

    @abstractmethod
    def get_norm(self):
        """Return (mean, std) for normalize()/undo_norm()."""
        pass

    def normalize(self, x):
        return (x - self.mean) / self.std

    def undo_norm(self, x):
        return x * self.std + self.mean

    def _ensure_open(self):
        if self._fields is None:
            f = h5py.File(self.path, 'r', libver="latest", swmr=True)
            self._fields = f['fields']

    def __len__(self):
        # if single snapshot, only one image; else sweep all T
        num_images = 1 if self.snapshot_idx is not None else self.T
        return num_images * self.patches_per_image

    def __getitem__(self, idx):
        self._ensure_open()

        # pick which time‐index to use
        if self.snapshot_idx is not None:
            t = self.snapshot_idx
        else:
            img_i = idx // self.patches_per_image
            t = img_i * self.skip_factor + self.shift_factor

        # pick which patch in that image
        patch_i = idx % self.patches_per_image
        row = (patch_i // self.max_col) * self.stride
        col = (patch_i %  self.max_col) * self.stride

        # slice out cond_snapshots frames ending at t
        # handle both 3D and 4D field layouts
        if self.has_channel:
            arr = self._fields[t : t + 1,  # index
                : ,                        # all channels
                row:row+self.patch_size,
                col:col+self.patch_size,
            ]  # shape = (cond_snapshots, C, Hp, Wp)
        else:
            arr = self._fields[
                t : t + 1,
                row:row+self.patch_size,
                col:col+self.patch_size,
            ]  # shape = (cond_snapshots, Hp, Wp)
            # add channel axis
            arr = arr[:, None, :, :]  # → (cond_snapshots, 1, Hp, Wp)

        # to torch
        patch = torch.from_numpy(arr).float()  # (Twin, C, Hp, Wp)
        #patch = self.normalize(patch)


        # downsample by factor
        down = patch[:, :, ::self.factor, ::self.factor]

        # then upsample back to full patch size
        up = F.interpolate(
            down,
            size=(self.patch_size, self.patch_size),
            mode='bilinear',
        )  # → (1, C, Hp, Wp)

        lowres = up.squeeze(0)     # → (C, Hp, Wp)

        # return (lowres, full-res stack, dummy label)
        return lowres, patch, torch.tensor(0).unsqueeze(0)


class ERA5_eval(EvalLoader):
    def __init__(self,
                 factor: int,
                 patch_size: int = 128,
                 stride: int = 128,
                 scratch_dir: str = 'data/',
                 snapshot_idx: int = None):
        super().__init__(
            factor=factor,
            patch_size=patch_size,
            stride=stride,
            scratch_dir=scratch_dir,
            snapshot_idx=snapshot_idx
        )

    def get_norm(self):
        return 0.0, 1.0

