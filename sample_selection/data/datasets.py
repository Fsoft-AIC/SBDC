import numpy as np 
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100
from .food import FOOD101
from .clothing import CLOTHING1M

train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
def input_dataset(dataset_path, dataset, noise_type, noise_ratio):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root=dataset_path,
                                download=False,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                           )
        test_dataset = CIFAR10(root=dataset_path,
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                          )
        num_classes = 10
        num_training_samples = 50000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root=dataset_path,
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                            )
        test_dataset = CIFAR100(root=dataset_path,
                                download=True,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                            )
        num_classes = 100
        num_training_samples = 50000
    return train_dataset, test_dataset, num_classes, num_training_samples


def input_dataset_clip(dataset_path, dataset, noise_type, noise_ratio, transform, test_dataset_path=None):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root=dataset_path,
                                download=False,
                                train=True,
                                transform = transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                           )
        test_dataset = CIFAR10(root=dataset_path,
                                download=False,
                                train=False,
                                transform = transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                          )
        num_classes = 10
        num_training_samples = 50000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root=dataset_path,
                                download=False,
                                train=True,
                                transform=transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                            )
        test_dataset = CIFAR100(root=dataset_path,
                                download=False,
                                train=False,
                                transform=transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                            )
        num_classes = 100
        num_training_samples = 50000
    elif dataset == 'food':
        train_dataset = FOOD101(root=dataset_path,
                                download=False,
                                train=True,
                                transform=transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                            )
        test_dataset = FOOD101(root=dataset_path,
                                download=False,
                                train=False,
                                transform=transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                            )
        num_classes = 101
        num_training_samples = 75750
    elif dataset == 'clothing':
        train_dataset = CLOTHING1M(root=dataset_path,
                                train=True,
                                transform=transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                            )
        test_dataset = CLOTHING1M(root=test_dataset_path if test_dataset_path else dataset_path,
                                train=False,
                                transform=transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                            )
        num_classes = 14
        num_training_samples = 1000000
    else:
        raise f"Dataset name {dataset} does not exist!"
    return train_dataset, test_dataset, num_classes, num_training_samples








