import torch
from torchvision import transforms
import torchvision.transforms.functional as FT

from PIL import Image
import numpy as np


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_model_input_size(model):
    if model in ["cnn", "resnet"]:
        return 32
    elif model in ["simclr", "dino"]:
        return 224
    else:
        assert f"Model {model} is not implemented!"


def get_normalize(dataset_name=None, model_name=None):
    if dataset_name:
        if dataset_name in ['cifar10', 'cifar100', 'food', 'clothing']:
            return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif dataset_name == "mnist":
            return transforms.Normalize((0.44671097, 0.4398105, 0.4066468), (0.2603405, 0.25657743, 0.27126738))
        elif dataset_name == "fmnist":
            return transforms.Normalize((0.44671097, 0.4398105, 0.4066468), (0.2603405, 0.25657743, 0.27126738))
        else:
            raise NameError('Undefined dataset')
    elif model_name:
        if model_name == "simclr":
            return transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        elif model_name == "clip":
            return transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        elif model_name == "dino":
            return transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        else:
            raise NameError('Undefined model')
    else:
        raise NameError('Undefined dataset and model')


def get_data_and_model_transform(dataset_name, model_name, model_input_size=-1):
    if model_input_size < 0:
        model_input_size = get_model_input_size(model_name)

    if model_name in ["cnn", "resnet"]:
        transform_list = [
            transforms.Resize(model_input_size, Image.BILINEAR),
            transforms.CenterCrop(model_input_size),
            transforms.ToTensor(),
            get_normalize(dataset_name=dataset_name),
        ]
    elif model_name in ["simclr", "clip", "dino"]:
        if dataset_name in ['food', 'clothing']:
            transform_list = [
                transforms.Resize(model_input_size, interpolation=Image.BILINEAR),
                transforms.CenterCrop(model_input_size),
                transforms.ToTensor(),
                get_normalize(model_name=model_name),
            ]
        else:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(model_input_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(model_input_size),
                # transforms.Resize(model_input_size, Image.BICUBIC),
                transforms.ToTensor(),
                get_normalize(model_name=model_name),
            ]
    else:
        assert "Not any model type in [CNN, ResNet, SimCLR, CLIP, DinoV2]"

    return transform_list

