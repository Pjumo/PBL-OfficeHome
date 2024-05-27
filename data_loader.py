from torchvision import datasets, transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class BasicDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(BasicDataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


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
    elif sample_type == 'sample_weight':
        class_sample_count = np.array(
            [len(np.where(train_data.targets == t)[0]) for t in np.unique(train_data.targets)]
        )
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_data.targets])
        samples_weight = torch.from_numpy(samples_weight)

        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=4)
    elif sample_type == 'smote':
        train_data_label = train_data.targets
        train_data_flat = []
        for data in train_data:
            train_data_flat.append(data[0].numpy())

        train_data_flat = np.array(train_data_flat).reshape(-1, 224 * 224 * 3)

        smote = SMOTE()
        train_data_x, train_data_y = smote.fit_resample(train_data_flat, train_data_label)

        train_data_x = train_data_x.reshape(-1, 3, 224, 224)

        train_data_resample = BasicDataset(torch.tensor(train_data_x), torch.tensor(train_data_y))
        train_loader = torch.utils.data.DataLoader(dataset=train_data_resample, batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
    elif sample_type == 'under_sampling':
        train_data_label = train_data.targets
        train_data_flat = []
        for data in train_data:
            train_data_flat.append(data[0].numpy())

        train_data_flat = np.array(train_data_flat).reshape(-1, 224 * 224 * 3)

        rus = RandomUnderSampler()
        train_data_x, train_data_y = rus.fit_resample(train_data_flat, train_data_label)

        train_data_x = train_data_x.reshape(-1, 3, 224, 224)

        train_data_resample = BasicDataset(torch.tensor(train_data_x), torch.tensor(train_data_y))
        train_loader = torch.utils.data.DataLoader(dataset=train_data_resample, batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
