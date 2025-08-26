from PIL import Image
import os
import os.path

import numpy as np
import sys
from datasets import load_dataset

import torch.utils.data as data
from torchvision import transforms
import PIL.Image

class CLOTHING1M(data.Dataset):
    """`CLOTHING1M`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dg_transform = transforms.Resize((64, 64), PIL.Image.BILINEAR)
        self.train = train  # training set or test set
        self.dataset='CLOTHING1M'
        self.noise_type=noise_type
        self.nb_classes=14
        idx_each_class_noisy = [[] for i in range(14)]
        
        if self.train:
            data = np.load(root)
            self.train_data = data["arr_0"].astype(np.uint8)
            
            self.train_labels = data["arr_1"]     # if use 25k images
            # self.train_labels = data["arr_1"]
            # if noise_type is not None:
            if noise_type != 'clean':
                # noisify train data
                self.train_noisy_labels = data["arr_1"]
                _train_labels = [i for i in self.train_labels]
                for i in range(len(_train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(14)]
                # self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy) # known noisy distribution
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(_train_labels)
            print(self.train_data.shape)
        else:
            data = np.load(root)
            self.test_data = data["arr_0"].astype(np.uint8)
            self.test_labels = data["arr_1"]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type !='clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


