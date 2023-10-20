import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from datasets.mnistm import MNISTM

def MNIST_loaders(batch_size, num_workers, type='train'):
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

    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    mnist_valid_loader = DataLoader(mnist_valid, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, num_workers=num_workers)

    mnistm_train_loader = DataLoader(mnistm_train, batch_size=batch_size, sampler=mtrain_sampler, num_workers=num_workers)
    mnistm_valid_loader = DataLoader(mnistm_valid, batch_size=batch_size, sampler=mtrain_sampler, num_workers=num_workers)
    mnistm_test_loader = DataLoader(mnistm_test, batch_size=batch_size, num_workers=num_workers)

    if type=='train':
        return mnist_train_loader, mnistm_train_loader
    elif type=='valid':
        return mnist_valid_loader, mnistm_valid_loader
    elif type=='test':
        return mnist_test_loader, mnistm_test_loader
    else:
        return mnist_train_loader, mnistm_train_loader, mnist_valid_loader, mnistm_valid_loader, mnist_test_loader, mnistm_test_loader
        
