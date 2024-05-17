from torchvision import datasets, transforms
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import os
import numpy as np


def dataloader(sample_type, batch_size):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data = datasets.ImageFolder(root=os.path.join("OfficeHome_dataset/train"), transform=transform_train)
    test_data = datasets.ImageFolder(root=os.path.join("OfficeHome_dataset/test"), transform=transform_test)
    if sample_type == 'base':
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
    else:
        class_sample_count = np.array(
            [len(np.where(train_data.targets == t)[0]) for t in np.unique(train_data.targets)]
        )
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_data.targets])
        samples_weight = torch.from_numpy(samples_weight)

        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler, shuffle=True,
                                                   num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
