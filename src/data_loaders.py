import torch
from torchvision import datasets, transforms

from .config import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD
from .dip import transform_pil


def build_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5),
    ])


def build_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dip_test_transform():
    return transforms.Compose([
        transforms.Lambda(transform_pil),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_cifar10_datasets(data_dir=DATA_DIR, download=True):
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=download, transform=build_train_transform())
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=download, transform=build_test_transform())
    dip_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=build_dip_test_transform())
    return train_ds, test_ds, dip_ds


def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir=DATA_DIR, download=True):
    train_ds, test_ds, dip_ds = get_cifar10_datasets(data_dir=data_dir, download=download)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dip_loader = torch.utils.data.DataLoader(dip_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, train_loader, test_loader, dip_loader
