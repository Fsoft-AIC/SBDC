from __future__ import print_function
from PIL import Image
import os
import os.path

import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from .utils import download_url, check_integrity, noisify, noisify_instance

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

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
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'cifar10'
        self.noise_type = noise_type
        self.nb_classes = 10
        idx_each_class_noisy = [[] for i in range(10)]
        # if download:
        #    self.download()

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = None
            labels_classes = None
            with open(root, "rb") as f:
                data = pickle.load(f)

            for m in ["train", "val"]:
                image = np.concatenate([data[f"{m}_clean"]["image"], data[f"{m}_noisy"]["image"]])
                self.train_data = image if self.train_data is None \
                    else np.concatenate([self.train_data, image])

                lab = np.concatenate([data[f"{m}_clean"]["label"].reshape(-1),
                                      data[f"{m}_noisy"]["label"].reshape(-1)]
                                     )
                cls = np.concatenate([data[f"{m}_clean"]["class"].reshape(-1),
                                      data[f"{m}_noisy"]["class"].reshape(-1)]
                                     )
                labels = np.concatenate([lab.reshape(-1, 1), cls.reshape(-1, 1)], axis=1).astype(np.int16)
                labels_classes = labels if labels_classes is None \
                    else np.concatenate([labels_classes, labels])

            self.train_labels = labels_classes[:, 1]
            self.train_data = self.train_data.astype(np.uint8)  # convert to HWC
            # if noise_type is not None:
            if noise_type != 'clean':
                # noisify train data
                self.train_noisy_labels = labels_classes[:, 0]
                _train_labels = [i for i in self.train_labels]
                for i in range(len(_train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                # self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy) # known noisy distribution
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(_train_labels)
            print(self.train_data.shape)
        else:
            self.test_data = None
            labels_classes = None
            with open(root, "rb") as f:
                data = pickle.load(f)

            for m in ["test"]:
                image = np.concatenate([data[f"{m}_clean"]["image"], data[f"{m}_noisy"]["image"]])
                self.test_data = image if self.test_data is None else np.concatenate([self.test_data, image])

                lab = np.concatenate([data[f"{m}_clean"]["label"].reshape(-1),
                                      data[f"{m}_noisy"]["label"].reshape(-1)]
                                     )
                cls = np.concatenate([data[f"{m}_clean"]["class"].reshape(-1),
                                      data[f"{m}_noisy"]["class"].reshape(-1)]
                                     )
                labels = np.concatenate([lab.reshape(-1, 1), cls.reshape(-1, 1)], axis=1).astype(np.int16)
                labels_classes = labels if labels_classes is None \
                    else np.concatenate([labels_classes, labels])

            self.test_labels = labels_classes[:, 1]
            self.test_data = self.test_data.astype(np.uint8)  # convert to HWC

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

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

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

class CIFAR100(data.Dataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

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
        self.train = train  # training set or test set
        self.dataset='cifar100'
        self.noise_type=noise_type
        self.nb_classes=100
        idx_each_class_noisy = [[] for i in range(100)]

        # now load the picked numpy arrays
        if self.train:
            self.train_data = None
            data = np.load(root)
            self.train_data = data["arr_0"]
            self.train_labels = data["arr_2"]

            self.train_data = self.train_data.astype(np.uint8)  # convert to HWC
            # if noise_type is not None:
            if noise_type != 'clean':
                # noisify train data
                self.train_noisy_labels = data["arr_1"]
                _train_labels = [i for i in self.train_labels]
                for i in range(len(_train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
                # self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy) # known noisy distribution
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(_train_labels)
            print(self.train_data.shape)
            
        else:
            file = os.path.splitext(self.root)
            baseroot = os.path.abspath(os.path.join(self.root, os.pardir))
            data = np.load(os.path.join(baseroot, f"{os.path.basename(file[0]).split('_')[0]}_eval{file[1]}"))
            self.test_data = data['arr_0']
            self.test_labels = data['arr_1']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type is not None:
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

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

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


