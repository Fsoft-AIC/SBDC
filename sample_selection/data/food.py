from PIL import Image
import os
import os.path

import numpy as np
import sys
from datasets import load_dataset

import torch.utils.data as data
from torchvision import transforms
import PIL.Image

class FOOD101(data.Dataset):
    """`FOOD101`_ Dataset.

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
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
 

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dg_transform = transforms.Resize((64, 64), PIL.Image.BILINEAR)
        self.train = train  # training set or test set
        self.dataset='food101'
        self.noise_type=noise_type
        self.nb_classes=101
        idx_each_class_noisy = [[] for i in range(101)]
        
        data = load_dataset(self.root)
        # now load the picked numpy arrays
        if self.train:
            self._train_data = data["train"]
            self.train_labels = self._train_data["label"]

            # if noise_type is not None:
            if noise_type != 'clean':
                # noisify train data
                self.train_noisy_labels = self.train_labels
                _train_labels = [i for i in self.train_labels]
                for i in range(len(_train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(101)]
                # self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy) # known noisy distribution
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(_train_labels)
            print(self._train_data.shape)
            self.train_data = np.zeros([self._train_data.shape[0], 64, 64, 3])
        else:
            self.test_data = data['validation']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self._train_data[index]['image'], self.train_noisy_labels[index]
            if self.train_data[index].sum() == 0:
                image = np.array(self.dg_transform(img.convert('RGB')))
                self.train_data[index] = image
        else:
            img, target = self.test_data[index]['image'], self.test_data[index]['label']
        
        if self.transform is not None:
            img = self.transform(img.convert('RGB'))
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


