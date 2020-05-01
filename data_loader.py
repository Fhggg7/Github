import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def data_loader(batch_size=256, workers=10, pin_memory=True):
    # traindir = os.path.join('G:\\Imagenet', 'test')
    traindir = 'G:\\Imagenet'
    print(traindir)
    # traindir = 'G:/Imagenet/test'
    # valdir = 'G:/Imagenet/test'
    valdir = traindir
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    print("done")
    return train_loader, val_loader
