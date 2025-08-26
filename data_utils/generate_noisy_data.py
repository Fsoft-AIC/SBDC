import torchvision
import torchvision.transforms as transforms

from noise_generator import *

####################################################################################################################
### MNIST_28*28
####################################################################################################################

# Define Transform
transform = transforms.Compose([transforms.ToTensor()])

# Load Data
train_dataset = torchvision.datasets.MNIST(root='./', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./', train=False, transform=transform)

train_labels = train_dataset.targets.numpy()
test_labels = test_dataset.targets.numpy()
train_dataset = np.array(list((map(lambda a: transform(a), train_dataset.data.numpy()))))
test_dataset = np.array(list((map(lambda a: transform(a), test_dataset.data.numpy()))))

# Sym
for n_ratio in [0.2, 0.5, 0.8]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean':dict(), 'val_noisy':dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    total_data = generate_dict_sym(train_dataset, train_labels, n_ratio, total_data)
    total_data = generate_dict_sym(test_dataset, test_labels, 0.5, total_data, mode="test")
    save_dict_to_pickle('MNIST', n_ratio, total_data, 'sym')
    print('MNIST symmetric noise', n_ratio)

# Asym
for Noise_ratio in [0.2, 0.4]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean': dict(), 'val_noisy': dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    label_list = [0, 1, 7, 8, 4, 6, 5, 7, 8, 9]  # from iclr 2021 REL
    total_data = generate_dict_asym(train_dataset, train_labels, Noise_ratio, total_data, label_list)
    total_data = generate_dict_asym(test_dataset, test_labels, 0.5, total_data, label_list, mode="test")
    save_dict_to_pickle('MNIST', Noise_ratio, total_data, 'asym')
    print('MNIST asymmetric noise1', Noise_ratio)

# IDN
for Noise_ratio in [0.2,0.4]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean': dict(), 'val_noisy': dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    total_data = generate_dict_idn(train_dataset, train_labels, Noise_ratio, total_data, 1*28*28, 10)
    total_data = generate_dict_idn(test_dataset, test_labels, 0.5, total_data, 1 * 28 * 28, 10, mode="test")
    save_dict_to_pickle('MNIST', Noise_ratio, total_data, 'idn')
    print('MNIST IDN noise', Noise_ratio)

# # SRIDN
# with open('idn_data/MNIST.pk', 'rb') as f:
#     data_dict = pickle.load(f)
#
# for n_ratio in [0.1,0.2,0.3,0.4]:
#     total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean': dict(), 'val_noisy': dict(),
#                   "test_clean": dict(), "test_noisy": dict()}
#     total_data = generate_dict_sridn(data_dict, train_labels, n_ratio, total_data)
#     total_data = generate_dict_sridn(data_dict, test_labels, 0.5, total_data, mode="test")
#     save_dict_to_pickle('MNIST', n_ratio, total_data, 'sridn')
#     print('MNIST SRIDN Noise', n_ratio)

####################################################################################################################
### FMNIST_28*28
####################################################################################################################

# Define Transform
transform = transforms.Compose([transforms.ToTensor()])

# Load Data
train_dataset = torchvision.datasets.FashionMNIST(root='./', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./', train=False, transform=transform)

train_labels = train_dataset.targets.numpy()
test_labels = test_dataset.targets.numpy()
train_dataset = np.array(list((map(lambda a: transform(a), train_dataset.data.numpy()))))
test_dataset = np.array(list((map(lambda a: transform(a), test_dataset.data.numpy()))))

# Sym
for n_ratio in [0.2, 0.5, 0.8]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean':dict(), 'val_noisy':dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    total_data = generate_dict_sym(train_dataset, train_labels, n_ratio, total_data)
    total_data = generate_dict_sym(test_dataset, test_labels, 0.5, total_data, mode="test")
    save_dict_to_pickle('FMNIST', n_ratio, total_data, 'sym')
    print('FMNIST symmetric noise', n_ratio)

# Asym
for Noise_ratio in [0.2, 0.4]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean': dict(), 'val_noisy': dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    label_list = [6, 1, 4, 3, 4, 7, 6, 7, 8, 9]
    total_data = generate_dict_asym(train_dataset, train_labels, Noise_ratio, total_data, label_list)
    total_data = generate_dict_asym(test_dataset, test_labels, 0.5, total_data, label_list, mode="test")
    save_dict_to_pickle('FMNIST', Noise_ratio, total_data, 'asym')
    print('FMNIST asymmetric noise', Noise_ratio)

# IDN
for Noise_ratio in [0.2,0.4]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean': dict(), 'val_noisy': dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    total_data = generate_dict_idn(train_dataset, train_labels, Noise_ratio, total_data, 1*28*28, 10)
    total_data = generate_dict_idn(test_dataset, test_labels, 0.5, total_data, 1 * 28 * 28, 10, mode="test")
    save_dict_to_pickle('FMNIST', Noise_ratio, total_data, 'idn')
    print('FMNIST IDN noise', Noise_ratio)

# # SRIDN
# with open('idn_data/FMNIST.pk', 'rb') as f:
#     data_dict = pickle.load(f)
#
# for n_ratio in [0.1,0.2,0.3,0.4]:
#     total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean': dict(), 'val_noisy': dict(),
#                   "test_clean": dict(), "test_noisy": dict()}
#     total_data = generate_dict_sridn(data_dict, train_labels, n_ratio, total_data)
#     total_data = generate_dict_sridn(data_dict, test_labels, 0.5, total_data, mode="test")
#     save_dict_to_pickle('FMNIST', n_ratio, total_data, 'sridn')
#     print('FMNIST SRIDN Noise', n_ratio)

####################################################################################################################
### CIFAR10 dataset
####################################################################################################################

# Define Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load Data
train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)

train_labels = np.array(train_dataset.targets)
test_labels = np.array(test_dataset.targets)
train_dataset = np.array(list((map(lambda a: transform(a), train_dataset.data))))
test_dataset = np.array(list((map(lambda a: transform(a), test_dataset.data))))

# Sym
for Noise_ratio in [0.2,0.5,0.8]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean':dict(), 'val_noisy':dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    total_data = generate_dict_sym(train_dataset, train_labels, Noise_ratio, total_data)
    total_data = generate_dict_sym(test_dataset, test_labels, 0.5, total_data, mode="test")
    save_dict_to_pickle('CIFAR10',Noise_ratio, total_data, 'sym')
    print('CIFAR10 symmetric noise',Noise_ratio)

#[plane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
# Asym
for Noise_ratio in [0.2, 0.4]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean':dict(), 'val_noisy':dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    label_list = [0,1,0,5,7,3,6,7,8,1]
    total_data = generate_dict_asym(train_dataset, train_labels, Noise_ratio,total_data, label_list)
    total_data = generate_dict_asym(test_dataset, test_labels, 0.5,total_data, label_list, mode="test")
    save_dict_to_pickle('CIFAR10',Noise_ratio, total_data, 'asym')
    print('CIFAR10 asymmetric noise',Noise_ratio)

# IDN
for Noise_ratio in [0.2,0.4]:
    total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean': dict(), 'val_noisy': dict(),
                  "test_clean": dict(), "test_noisy": dict()}
    total_data = generate_dict_idn(train_dataset, train_labels, Noise_ratio, total_data, 3*32*32, 10)
    total_data = generate_dict_idn(test_dataset, test_labels, 0.5, total_data, 3*32*32, 10, mode="test")
    save_dict_to_pickle('CIFAR10', Noise_ratio, total_data, 'idn')
    print('CIFAR10 IDN noise', Noise_ratio)

# # SRIDN
# with open('idn_data/CIFAR10.pk', 'rb') as f:
#     data_dict = pickle.load(f)
#
# for n_ratio in [0.1,0.2,0.3,0.4]:
#     total_data = {"train_clean": dict(), "train_noisy": dict(), 'val_clean': dict(), 'val_noisy': dict(),
#                   "test_clean": dict(), "test_noisy": dict()}
#     total_data = generate_dict_sridn(data_dict, train_labels, n_ratio, total_data)
#     total_data = generate_dict_sridn(data_dict, test_labels, 0.5, total_data, mode="test")
#     save_dict_to_pickle('CIFAR10', n_ratio, total_data, 'sridn')
#     print('CIFAR10 SRIDN Noise', n_ratio)

####################################################################################################################
