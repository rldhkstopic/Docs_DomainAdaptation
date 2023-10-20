import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from datasets.mnistm import MNISTM

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.29730626,), (0.32780124,))
])

mtrasnform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
                                                    (0.32780124, 0.32292358, 0.32056796))
                                ])

validation_size = 5000

mnist_train = datasets.MNIST(root='datasets/', train=True, download=True, transform=transform)
mnist_valid = datasets.MNIST(root='datasets/', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='datasets/', train=False, transform=transform)
mnistm_train = MNISTM(root='datasets/MNIST-M', train=True, download=True, transform=mtrasnform)
mnistm_valid = MNISTM(root='datasets/MNIST-M', train=True, download=True, transform=mtrasnform)
mnistm_test = MNISTM(root='datasets/MNIST-M', train=False, transform=mtrasnform)

indices = list(range(len(mnist_train)))
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler, valid_sample = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)

mindices = list(range(len(mnistm_train)))
mtrain_idx, mvalid_idx = mindices[validation_size:], mindices[:validation_size]
mtrain_sampler, mvalid_sample = SubsetRandomSampler(mtrain_idx), SubsetRandomSampler(mvalid_idx)

mnist_train_loader = DataLoader(mnist_train, batch_size=32, sampler=train_sampler, num_workers=4)
mnist_valid_loader = DataLoader(mnist_valid, batch_size=32, sampler=train_sampler, num_workers=4)
mnist_test_loader = DataLoader(mnist_test, batch_size=32, num_workers=4)
mnistm_train_loader = DataLoader(mnistm_train, batch_size=32, sampler=mtrain_sampler, num_workers=4)
mnistm_valid_loader = DataLoader(mnistm_valid, batch_size=32, sampler=mtrain_sampler, num_workers=4)
mnistm_test_loader = DataLoader(mnistm_test, batch_size=32, num_workers=4)
