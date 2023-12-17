import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms


from transform import *


def set_all_seeds(seed):
    # for reproducibility
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_all_seeds(seed=123)


class BaseDataset:
    def __init__(self, batch_size, input_dim=32, test_batch_size=2000, normalize=True):
        """
        :param input_dim: required dim for neural net
        :param normalize: set False only when dataloaders are defined to visualize images
        """

        self.batch_size = batch_size

        self.test_batch_size = test_batch_size

        self.trans_train = [transforms.Resize((input_dim, input_dim)),
                            # transforms.RandomCrop(input_dim, padding=2),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()]
        self.trans_test = [transforms.Resize((input_dim, input_dim)),
                           transforms.ToTensor()]

        if normalize:
            self.trans_train += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            self.trans_test += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    def build_loaders(self, train_dataset, valid_dataset):
        # define the data loaders
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=self.test_batch_size,
                                  shuffle=False)
        return train_loader, valid_loader


class MNIST(BaseDataset):
    # define transforms
    def __init__(self, batch_size, input_dim, test_batch_size=2000, normalize=False):
        super().__init__(batch_size, input_dim, test_batch_size, normalize)

        train_dataset = datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose(self.trans_train)
                                      )
        valid_dataset = datasets.MNIST(root='./data',
                                      train=False,
                                      download=True,
                                      transform=transforms.Compose(self.trans_test)
                                      )
        self.train_loader, self.valid_loader = self.build_loaders(train_dataset, valid_dataset)


    def __num_classes__(self):
        return len(self.train_loader.dataset.classes)



class SVHN(BaseDataset):
    def __init__(self, batch_size, input_dim, test_batch_size=2000, normalize=True):
        super().__init__(batch_size, input_dim, test_batch_size, normalize)

        # download and create datasets
        train_dataset = datasets.SVHN(root='./data',
                                      split='train',
                                      download=True,
                                      transform=transforms.Compose(self.trans_train)
                                      )
        valid_dataset = datasets.SVHN(root='./data',
                                      split='test',
                                      download=True,
                                      transform=transforms.Compose(self.trans_test)
                                      )
        self.train_loader, self.valid_loader = self.build_loaders(train_dataset, valid_dataset)

    def __num_classes__(self):
        return 10


class CIFAR10(BaseDataset):
    def __init__(self, batch_size, input_dim, test_batch_size=2000, normalize=True):
        super().__init__(batch_size, input_dim, test_batch_size, normalize)

        # download and create datasets
        train_dataset = datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transforms.Compose(self.trans_train)
                                         )
        valid_dataset = datasets.CIFAR10(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transforms.Compose(self.trans_test)
                                         )

        self.train_loader, self.valid_loader = self.build_loaders(train_dataset, valid_dataset)

    def __num_classes__(self):
        return len(self.train_loader.dataset.classes)


class CIFAR100(BaseDataset):
    def __init__(self, batch_size, input_dim, test_batch_size=2000, normalize=True):
        super().__init__(batch_size, input_dim, test_batch_size, normalize)

        # download and create datasets
        train_dataset = datasets.CIFAR100(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose(self.trans_train)
                                          )
        valid_dataset = datasets.CIFAR100(root='./data',
                                          train=False,
                                          download=True,
                                          transform=transforms.Compose(self.trans_test)
                                          )

        self.train_loader, self.valid_loader = self.build_loaders(train_dataset, valid_dataset)

    def __num_classes__(self):
        return len(self.train_loader.dataset.classes)