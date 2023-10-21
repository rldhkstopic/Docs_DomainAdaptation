import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from datasets.mnistm import MNISTM

class DataSet:
    def __init__(self, source, target, batch_size, num_workers, type='train'):
        self.source_dataset_name = source
        self.target_dataset_name = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.type = type
        
        self.source_transform = self._get_transform(self.source_dataset_name)
        self.target_transform = self._get_transform(self.target_dataset_name)
        
        self.source_dataset = self._get_dataset(self.source_dataset_name, self.source_transform, type)
        self.target_dataset = self._get_dataset(self.target_dataset_name, self.target_transform, type)

        self.source_loader = self._get_loader(self.source_dataset, True) 
        self.target_loader = self._get_loader(self.target_dataset, False) # False indicates target

    def _get_transform(self, dataset_name):
        if dataset_name == 'mnist':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.29730626,), (0.32780124,))
            ])
        elif dataset_name == 'mnistm':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
                                    (0.32780124, 0.32292358, 0.32056796))
            ])
        elif dataset_name == 'svhn':
            return transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

    def _get_dataset(self, dataset_name, transform, type):
        if dataset_name == 'mnist':
            return datasets.MNIST(root='datasets/', train=(type=='train'), download=True, transform=transform)
        elif dataset_name == 'mnistm':
            return MNISTM(root='datasets/MNIST-M', train=(type=='train'), download=True, transform=transform)
        elif dataset_name == 'svhn':
            split = 'train' if type == 'train' else 'test'
            return datasets.SVHN(root='datasets/SVHN', split=split, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

    def _get_loader(self, dataset, is_source):
        if is_source:
            validation_size = 0.2 
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_size * dataset_size))
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers)
            valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler, num_workers=self.num_workers)
            
            return {"train": train_loader, "valid": valid_loader}
        else:
            test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            return {"test": test_loader}
