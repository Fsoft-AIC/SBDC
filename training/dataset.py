# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from torchvision import transforms, datasets
import cv2
try:
    import pyspng
except ImportError:
    pyspng = None
import blobfile as bf
#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
        verify      = False,    # Return (image, label) mismatch? False = only get image and label.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None
        self._verify = verify

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            # assert self._raw_labels.shape[0] == self._raw_shape[0], f"{self._raw_labels.shape[0]} - {self._raw_shape[0]}"
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image, embed_image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = (image, embed_image)
        else:    
            image, embed_image = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        
        if not self._verify:
            return image.copy(), self.get_label(idx) 

        real_or_not = self._real_or_not[self._raw_idx[idx]].reshape(1).astype(np.float32)
        label = self.get_label(idx)
        if real_or_not == 1 and np.random.rand(1) < self.rand_shuf_prob:
            label = self.get_label(idx, shuffle=True)
            real_or_not = np.zeros_like(real_or_not)

        if embed_image is not None:
            return image.copy(), embed_image.clone(), label, real_or_not

        return image.copy(), label, real_or_not

    def get_label(self, idx, shuffle=False):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            if shuffle:
                label_list = np.arange(self.label_shape[0])
                label_list = label_list[label_list != label]
                label = np.random.choice(label_list)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        real_fake_label_path = None, # Path to real/fake label of data.
        resolution      = None, # Ensure specific resolution, None = highest available.
        emb_resolution  = None, # Additional resolution for embedding network input shape.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self._emb_transform = None
        if emb_resolution:
            self._emb_transform = transforms.Compose([
                transforms.Resize(emb_resolution, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(emb_resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        if os.path.isdir(self._path):
            if 'webvision' in self._path:
                self._type = 'webvision'
                self._all_fnames = datasets.ImageFolder(root=os.path.join(self._path, "train"))
            else:
                self._type = 'dir'
                self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}

        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        elif self._file_ext(self._path) == '.pk':
            self._type = '.pk'
        elif self._file_ext(self._path) == '.npz':
            self._type = '.npz'    
        else:
            raise IOError('Path must point to a directory, pk, npz or zip')

        self.transform = transforms.Resize((resolution, resolution), PIL.Image.BILINEAR)
        self._real_or_not = None if real_fake_label_path is None else np.load(real_fake_label_path)["arr_0"].reshape(-1)

        name = os.path.splitext(os.path.basename(self._path))[0]
        if self._type == '.pk':
            import pickle
            with open(self._path, "rb") as f:
                data = pickle.load(f)

            self._image_fnames = None
            self._label_cls = None
            for m in ["train", "val"]:
                assert m in "train,val,test", f"Mode: train/val/test, currently {m}."
                _image = np.concatenate([data[f"{m}_clean"]["image"], data[f"{m}_noisy"]["image"]])
                self._image_fnames = _image if self._image_fnames is None \
                    else np.concatenate([self._image_fnames, _image])

                lab = np.concatenate([data[f"{m}_clean"]["label"].reshape(-1),
                                      data[f"{m}_noisy"]["label"].reshape(-1)]
                                     )
                cls = np.concatenate([data[f"{m}_clean"]["class"].reshape(-1),
                                      data[f"{m}_noisy"]["class"].reshape(-1)]
                                     )
                _label_cls = np.concatenate([lab.reshape(-1, 1), cls.reshape(-1, 1)], axis=1)
                self._label_cls = _label_cls if self._label_cls is None \
                    else np.concatenate([self._label_cls, _label_cls])
            self.transform = None

        elif self._type == '.npz':
            data = np.load(self._path)
            self._image_fnames = data["arr_0"]
            self._label = data["arr_1"]

        elif self._type == 'food':
            from datasets import load_dataset
            data = load_dataset(self._path)
            self._image_fnames = data["train"]
        elif self._type == 'webvision':
            self._image_fnames = self._all_fnames
        else:
            PIL.Image.init()
            # self._image_fnames = sorted(fname for fname in self._all_fnames
            #                             if self._file_ext(fname) in PIL.Image.EXTENSION)
            self._image_fnames = self._all_fnames
            if real_fake_label_path is not None:
                data = np.load(real_fake_label_path)
                self._label = data["arr_0"]
                self._class = data["arr_1"]
                self._real_or_not = data["arr_2"].reshape(-1)

        if self._real_or_not is not None:
            self.rand_shuf_prob = max(1 - 0.5 / (self._real_or_not.mean() + 1e-5), 0)

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0)[0].shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        if type(raw_idx) != int: 
            raw_idx = raw_idx.item()
        fname = self._image_fnames[raw_idx]
        if self._type in ['.pk', '.npz']:
            image = fname.astype(np.uint8)
        elif self._type == 'food':
            image = np.array(self.transform(fname['image'].convert('RGB')))
        elif self._type == 'webvision':
            image = np.array(self.transform(fname[0].convert('RGB')))
        else:
            with self._open_file(fname) as f:
                image = np.array(self.transform(PIL.Image.open(f))) if self.transform is not None \
                    else np.array(PIL.Image.open(f))

        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        
        if image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] == 2:
            image = np.concatenate([image, image[:,:,:1]], axis=0)

        if self._emb_transform:
            embed_image = PIL.Image.fromarray(image)
            embed_image = self._emb_transform(embed_image)
            image = image.transpose(2, 0, 1) # HWC => CHW
            return image, embed_image
        
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image, None

    def _load_raw_labels(self):
        if self._type == 'food':
            labels = self._image_fnames['label']
            labels = np.array(labels)
        elif self._type == 'webvision':
            labels = self._image_fnames.targets
            labels = np.array(labels)
        elif self._type == '.pk':
            labels = self._label_cls[:, 0]
        elif self._type == '.npz':
            labels = self._label
        elif self._type == 'dir':
            labels = self._label
        else:
            fname = 'dataset.json'
            if fname not in self._all_fnames:
                return None
            with self._open_file(fname) as f:
                labels = json.load(f)['labels']
            if labels is None:
                return None
            labels = dict(labels)
            labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
            labels = np.array(labels)
        
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

        return labels

#----------------------------------------------------------------------------