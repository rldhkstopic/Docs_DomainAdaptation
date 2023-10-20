import torch
import torch.nn as nn
import torch.optim as optims
import torch.functional as F
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import warnings
warnings.filterwarnings('ignore')


from tqdm import tqdm

params = {'epochs': 100, 'batch_size': 32, 'num_workers': 4}

class Network(nn.Module):
    def __init__(self, source, target):
        super(Network, self).__init__()
        
        self.dataset_source = source
        self.dataset_target = target  
        
        self.model = models.resnet18(pretrained=True)
        self.model = nn.DataParallel(self.model) if torch.cuda.device_count() > 1 else self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.E = nn.Sequential(*list(self.model.children())[:-1]).to(self.device) # Feature-Extractor(Encoder) with ResNet18
        self.C = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 10)).to(self.device) # Classifier
        self.D = Discriminator().to(self.device) # Domain Classifier
        
        self.CELoss = nn.CrossEntropyLoss().to(self.device)
        self.BCELoss = nn.BCELoss().to(self.device)
        self.optim = optims.Adam(self.model.parameters(), lr=0.01)
        # DataParallel은 Feature Extractor 후순위에서 선언해야됨. (안그러면 Sequential이 비어있음)

    def _train_source(self):
        self.model.train()
        optim = optims.Adam(
            list(self.C.parameters()) + 
            list(self.E.parameters()), lr=0.01)
        print(">> Only-Source Domain Training Start")    
        for epoch in range(params['epochs']):
            start_steps = epoch * len(self.dataset_source)
            total_steps = params['epochs'] * len(self.dataset_source)
            bar = tqdm(enumerate(self.dataset_source), total=len(self.dataset_source))
            for i, (data, labels) in bar:
                source_data = torch.cat((data, data, data), 1) # MNIST convert to 3 channel
                source_data, source_labels = source_data.to(self.device), labels.to(self.device)
                
                optim = optim_scheduler(optim=optim, p = float(i + start_steps) / total_steps)
                optim.zero_grad()
                source_feature = self.E(source_data)
                cls_predicted = self.C(source_feature)
                classification_loss = self.CELoss(cls_predicted, source_labels)
                classification_loss.backward(distance='source')
                optim.step()
                
                bar.set_description("Epoch: {}/{} [{}/{} ({:.0f}%)]\tClass Loss: {:.6f}".format(epoch, params['epochs'], i * len(source_data), len(self.dataset_source), 100. * i / len(self.dataset_source), classification_loss.item()))    

        if (epoch + 1) % 10 == 0:
            self._test_source()
        
        modelSave(self.E, self.C, self.D, 'source_only')
    
    def _test_source(self):
        self.model.eval()
        print(">> Only-Source Domain Test Start")
        correct_source, total_source = 0, 0
        with torch.no_grad():
            for images, labels in self.dataset_source:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total_source += labels.size(0)
                correct_source += predicted.eq(labels).sum().item()
                
        print(f'Source Domain Accuracy: {100 * correct_source / total_source:.2f}%')        
        
    def _train_dann(self):
        optim = optims.Adam(
            list(self.C.parameters()) + 
            list(self.E.parameters()) + 
            list(self.D.parameters()), lr=0.01, momentum=0.9)
        print(">> Domain-Adversarial Training Start")
        for epoch in range(params['epochs']):
            print('Epoch : {}'.format(epoch))
            start_steps = epoch * len(self.dataset_source)
            total_steps = params['epochs'] * len(self.dataset_target)
            for i, (source_data, target_data) in enumerate(zip(self.dataset_source, self.dataset_target)):
                source, source_labels = source_data
                target, _ = target_data
                
                p = float(i + start_steps) / total_steps
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                source_data = torch.cat((source, source, source), 1) # MNIST convert to 3 channel
                source_data, source_labels = source_data.to(self.device), source_labels.to(self.device)
                target_data, target_labels = target.to(self.device), target.to(self.device)
                total_data = torch.cat((source_data, target_data), 0)

                optim = optim_scheduler(optim=optim, p=p)
                optim.zero_grad()
                
                source_feature = self.E(source_data)
                total_feature = self.E(total_data)
                
                cls_predicted = self.C(source_feature)
                cls_loss = self.CELoss(cls_predicted, source_labels)
                
                domain_predicted = self.D(total_feature, alpha)
                domain_source_labels = torch.zeros(source_labels.shape[0]).type(torch.LongTensor)
                domain_target_labels = torch.ones(target_labels.shape[0]).type(torch.LongTensor)
                domain_total_labels = torch.cat((domain_source_labels, domain_target_labels), 0).to(self.device)
                domain_loss = self.BCELoss(domain_predicted, domain_total_labels)
                
                total_loss = cls_loss + domain_loss
                total_loss.backward()
                optim.step()
                
                if (i + 1) % 50 == 0:
                    print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                        i * len(target), len(self.dataset_target), 100. * i / len(self.dataset_target), total_loss.item(), cls_loss.item(), domain_loss.item()))
            
            if (epoch + 1) % 10 == 0:
                self._test_dann()
        
        modelSave(self.E, self.C, self.D, '_dann', isDann=True)
        
    def _test_dann(self):
        self.model.eval()
        print(">> Domain-Adversarial Test Start")
        correct_source, total_source = 0, 0
        correct_target, total_target = 0, 0
        with torch.no_grad():
            for images, labels in self.dataset_source:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total_source += labels.size(0)
                correct_source += predicted.eq(labels).sum().item()
                
                
            for images, labels in self.dataset_target:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total_target += labels.size(0)
                correct_target += predicted.eq(labels).sum().item()

        print(f'> | Source Domain Accuracy: {100 * correct_source / total_source:.2f}%')
        print(f'> | Target Domain Accuracy: {100 * correct_target / total_target:.2f}%')



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )
    
    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return F.softmax(x)


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    
    


def optim_scheduler(optim, p):
    for params in optim.param_groups:
        params['lr'] = 0.01 / (1. + 10 * p) ** 0.75
    return optim

import os
def modelSave(E, C, D, name, isDann=False):
    path = './models/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(E.state_dict(), f'{path}/Encoder/{name}_E.pt')
    torch.save(C.state_dict(), f'{path}/Classifier/{name}_C.pt')
    if isDann:
        torch.save(D.state_dict(), f'{path}/Discriminator/{name}_D.pt')
    
    print(f'Save {name} Model into /models/')