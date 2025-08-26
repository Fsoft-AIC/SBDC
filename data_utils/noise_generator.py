import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from math import inf
from scipy import stats


# for noise generation
# Generate symmetric noise
def generate_dict_sym(dataset, label, noise_ratio, data_dict, mode="train", n_class=10):
    noise_ratio *= n_class / (n_class - 1)  # For MNIST/FMNIST/CIFAR10, n_class==10
    data_size = len(dataset)
    assert mode in ["train", "test"], "Choose mode between train/test."
    keep_labels = np.random.random(data_size) > noise_ratio
    if mode == "train":
        new_labels = np.random.randint(low=0, high=n_class, size=((~keep_labels[: int(data_size * 0.1)]).sum(),))
        data_dict["val_clean"]['image'] = dataset[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_clean"]['label'] = label[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_clean"]['class'] = label[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_noisy"]['image'] = dataset[:int(data_size * 0.1)][~keep_labels[: int(data_size * 0.1)]]
        data_dict["val_noisy"]['label'] = new_labels
        data_dict["val_noisy"]['class'] = label[:int(data_size * 0.1)][~keep_labels[: int(data_size * 0.1)]]

        new_labels = np.random.randint(low=0, high=n_class, size=((~keep_labels[int(data_size * 0.1):]).sum(),))
        data_dict["train_clean"]['image'] = dataset[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_clean"]['label'] = label[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_clean"]['class'] = label[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_noisy"]['image'] = dataset[int(data_size * 0.1):][~keep_labels[int(data_size * 0.1):]]
        data_dict["train_noisy"]['label'] = new_labels
        data_dict["train_noisy"]['class'] = label[int(data_size * 0.1):][~keep_labels[int(data_size * 0.1):]]
    else:
        new_labels = np.random.randint(low=0, high=n_class, size=((~keep_labels).sum(),))
        data_dict["test_clean"]['image'] = dataset[keep_labels]
        data_dict["test_clean"]['label'] = label[keep_labels]
        data_dict["test_clean"]['class'] = label[keep_labels]
        data_dict["test_noisy"]['image'] = dataset[~keep_labels]
        data_dict["test_noisy"]['label'] = new_labels
        data_dict["test_noisy"]['class'] = label[~keep_labels]

    print(f'{mode}_data finished')
    return data_dict


# Generate asymmetric noise
def generate_dict_asym(dataset, label, noise_ratio, data_dict, label_list, mode="train"):
    data_size = len(dataset)
    assert mode in ["train", "test"], "Choose mode between train/test."
    label_list = torch.tensor(label_list).view(-1, 1)
    keep_labels = np.random.random(data_size) > noise_ratio
    if mode == "train":
        new_labels = label_list[label[:int(data_size * 0.1)][~keep_labels[: int(data_size * 0.1)]]]
        data_dict["val_clean"]['image'] = dataset[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_clean"]['label'] = label[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_clean"]['class'] = label[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_noisy"]['image'] = dataset[:int(data_size * 0.1)][~keep_labels[: int(data_size * 0.1)]]
        data_dict["val_noisy"]['label'] = new_labels.numpy()
        data_dict["val_noisy"]['class'] = label[:int(data_size * 0.1)][~keep_labels[: int(data_size * 0.1)]]

        new_labels = label_list[label[int(data_size * 0.1):][~keep_labels[int(data_size * 0.1):]]]
        data_dict["train_clean"]['image'] = dataset[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_clean"]['label'] = label[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_clean"]['class'] = label[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_noisy"]['image'] = dataset[int(data_size * 0.1):][~keep_labels[int(data_size * 0.1):]]
        data_dict["train_noisy"]['label'] = new_labels.numpy()
        data_dict["train_noisy"]['class'] = label[int(data_size * 0.1):][~keep_labels[int(data_size * 0.1):]]
    else:
        new_labels = label_list[label[~keep_labels]]
        data_dict["test_clean"]['image'] = dataset[keep_labels]
        data_dict["test_clean"]['label'] = label[keep_labels]
        data_dict["test_clean"]['class'] = label[keep_labels]
        data_dict["test_noisy"]['image'] = dataset[~keep_labels]
        data_dict["test_noisy"]['label'] = new_labels.numpy()
        data_dict["test_noisy"]['class'] = label[~keep_labels]

    print(f'{mode}_data finished')
    return data_dict


# Generate sridn noise
def generate_dict_sridn(original_dict, noise_ratio, data_dict, mode="train"):
    if mode == "train":
        data_to_see = original_dict['Train']
        data_len = len(data_to_see['l_true'])

        index = np.argsort(np.array(data_to_see['p_true']))
        num_noisy = int(data_len * noise_ratio)
        noisy_index = index[:num_noisy]
        clean_index = index[num_noisy:]

        # clean
        np.random.shuffle(clean_index)
        valid_num = int(len(clean_index) * 0.1)
        train_index = clean_index[valid_num:]
        valid_index = clean_index[:valid_num]

        data_dict['Train_Clean']['image'] = np.float32(np.array(data_to_see['images'])[train_index])
        data_dict['Train_Clean']['label'] = np.int64(np.array(data_to_see['l_true'])[train_index].reshape(-1, 1))
        data_dict['Train_Clean']['class'] = np.int64(np.array(data_to_see['l_true'])[train_index].reshape(-1, 1))

        data_dict['Val_Clean']['image'] = np.float32(np.array(data_to_see['images'])[valid_index])
        data_dict['Val_Clean']['label'] = np.int64(np.array(data_to_see['l_true'])[valid_index].reshape(-1, 1))
        data_dict['Val_Clean']['class'] = np.int64(np.array(data_to_see['l_true'])[valid_index].reshape(-1, 1))

        # noisy
        np.random.shuffle(noisy_index)
        valid_num = int(len(noisy_index) * 0.1)
        train_index = noisy_index[valid_num:]
        valid_index = noisy_index[:valid_num]

        data_dict['Train_Noisy']['image'] = np.float32(np.array(data_to_see['images'])[train_index])
        data_dict['Train_Noisy']['label'] = np.int64(np.array(data_to_see['l_model'])[train_index].reshape(-1, 1))
        data_dict['Train_Noisy']['class'] = np.int64(np.array(data_to_see['l_true'])[train_index].reshape(-1, 1))

        data_dict['Val_Noisy']['image'] = np.float32(np.array(data_to_see['images'])[valid_index])
        data_dict['Val_Noisy']['label'] = np.int64(np.array(data_to_see['l_model'])[valid_index].reshape(-1, 1))
        data_dict['Val_Noisy']['class'] = np.int64(np.array(data_to_see['l_true'])[valid_index].reshape(-1, 1))

    else:
        data_to_see = original_dict['Test']

        data_dict['Test_Clean']['image'] = np.float32(np.array(data_to_see['images']))
        data_dict['Test_Clean']['label'] = np.int64(np.array(data_to_see['l_true']).reshape(-1, 1))
        data_dict['Test_Clean']['class'] = np.int64(np.array(data_to_see['l_true']).reshape(-1, 1))
    return data_dict


# Generate idn noise
def generate_dict_idn(dataset, label, noise_ratio, data_dict, feature_size, n_class, mode="train"):
    data_size = len(dataset)
    flip_distribution = stats.truncnorm((0 - noise_ratio) / 0.1, (1 - noise_ratio) / 0.1, loc=noise_ratio, scale=0.1)
    flip_rate = flip_distribution.rvs(len(dataset))
    W = torch.randn(n_class, feature_size, n_class)
    assert mode in ["train", "test"], "Choose mode between train/test."

    p = torch.tensor(dataset).view(data_size, -1, 1)  * (W[label])
    p = p.sum(dim=-2)
    for i in range(data_size):
        p[i, label[i]] = -inf
    p = torch.tensor(flip_rate).view(-1, 1) * torch.softmax(p, dim=-1)
    for i in range(data_size):
        p[i, label[i]] += 1 - flip_rate[i]
    new_labels = torch.multinomial(p, 1)
    keep_labels = new_labels.squeeze().numpy() == label

    if mode == "train":
        data_dict["val_clean"]['image'] = dataset[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_clean"]['label'] = label[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_clean"]['class'] = label[:int(data_size * 0.1)][keep_labels[: int(data_size * 0.1)]]
        data_dict["val_noisy"]['image'] = dataset[:int(data_size * 0.1)][~keep_labels[: int(data_size * 0.1)]]
        data_dict["val_noisy"]['label'] = new_labels[:int(data_size * 0.1)][~keep_labels[: int(data_size * 0.1)]].squeeze().numpy()
        data_dict["val_noisy"]['class'] = label[:int(data_size * 0.1)][~keep_labels[: int(data_size * 0.1)]]

        data_dict["train_clean"]['image'] = dataset[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_clean"]['label'] = label[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_clean"]['class'] = label[int(data_size * 0.1):][keep_labels[int(data_size * 0.1):]]
        data_dict["train_noisy"]['image'] = dataset[int(data_size * 0.1):][~keep_labels[int(data_size * 0.1):]]
        data_dict["train_noisy"]['label'] = new_labels[int(data_size * 0.1):][~keep_labels[int(data_size * 0.1):]].squeeze().numpy()
        data_dict["train_noisy"]['class'] = label[int(data_size * 0.1):][~keep_labels[int(data_size * 0.1):]]
    else:
        data_dict["test_clean"]['image'] = dataset[keep_labels]
        data_dict["test_clean"]['label'] = label[keep_labels]
        data_dict["test_clean"]['class'] = label[keep_labels]
        data_dict["test_noisy"]['image'] = dataset[~keep_labels]
        data_dict["test_noisy"]['label'] = new_labels[~keep_labels].squeeze().numpy()
        data_dict["test_noisy"]['class'] = label[~keep_labels]

    print(f'{mode}_data finished')
    return data_dict


def save_dict_to_pickle(data_name, Noise_ratio, total_data, noise_type):
    os.makedirs('data/', exist_ok=True)

    if Noise_ratio > 0.0:
        noise_percent = str(Noise_ratio * 100)
    else:
        noise_percent = '00.0'

    with open('data/' + data_name + "_" + noise_percent + '_' + noise_type + ".pk", "wb") as f:
        pickle.dump(total_data, f)
    f.close()
