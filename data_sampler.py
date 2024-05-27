from torchvision import datasets, transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
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


torch.multiprocessing.freeze_support()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=os.path.join("OfficeHome_dataset/train"), transform=transform_train)

train_data_label = train_data.targets
train_data_flat = []
for data in train_data:
    train_data_flat.append(data[0].numpy())

train_data_flat = np.array(train_data_flat)

train_data_flat = np.array(train_data_flat).reshape(-1, 224 * 224 * 3)

rus = RandomUnderSampler()
train_data_x, train_data_y = rus.fit_resample(train_data_flat, train_data_label)

train_data_x = train_data_x.reshape(-1, 3, 224, 224)

counter = Counter(train_data_y)
plt.figure()
plt.bar(counter.keys(), counter.values())
plt.show()

train_data_resample = BasicDataset(torch.tensor(train_data_x), torch.tensor(train_data_y))
train_loader = torch.utils.data.DataLoader(dataset=train_data_resample, batch_size=32, shuffle=True)

torch.cuda.empty_cache()
