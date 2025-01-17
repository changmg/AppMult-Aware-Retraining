import torch
from torchvision import datasets, transforms
import os


def get_cifar10(batch_size, data_root='~/dataset/cifar10', **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []

    # training data with data augmentation
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    ds.append(train_loader)

    # testing data
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        batch_size=batch_size, shuffle=False, **kwargs)
    ds.append(test_loader)

    # training data without data augmentation
    train_loader_no_aug = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    ds.append(train_loader_no_aug)

    return ds


def get_cifar100(batch_size, data_root='~/dataset/cifar100', **kwargs):
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []

    # training data with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    cifar100_training = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        cifar100_training, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    ds.append(train_loader)

    # testing data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    cifar100_test = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        cifar100_test, shuffle=False, num_workers=num_workers, batch_size=batch_size)    
    ds.append(test_loader)

    # training data without data augmentation
    cifar100_training = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_test)
    train_loader_no_aug = torch.utils.data.DataLoader(
        cifar100_training, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    
    ds.append(train_loader_no_aug)

    return ds


def get_mnist(batch_size, data_root='~/dataset/mnist', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'mnist-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, 
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds