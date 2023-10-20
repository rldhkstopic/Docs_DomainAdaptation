import warnings
import numpy as np
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optims
import torch.nn.functional as F

from tqdm import tqdm
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

class Network():
    def __init__(self, source, target):
        super(Network, self).__init__()
        
        self.dataset_source = source
        self.dataset_target = target
        self.writer = SummaryWriter()
        
        self.model = models.resnet18(pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            layout = list(self.model.module.children())
        else:
            layout = list(self.model.children())
            
        self.D = Discriminator().to(self.device)                              # Discriminator
        self.E = nn.Sequential(*layout[:-1]).to(self.device)                  # Encoder
        self.C = nn.Sequential(nn.Linear(512, 100, bias=True),
                                nn.ReLU(), 
                                nn.Linear(100, 10)).to(self.device)           # Classifier
        
        self.CELoss = nn.CrossEntropyLoss().to(self.device)
        self.BCELoss = nn.BCELoss().to(self.device)
        self.optim = optims.Adam(self.model.parameters(), lr=0.01)

    def _load_model(self):
        E_path = os.path.join(f'models/Encoder/source_only_E.pt')
        C_path = os.path.join(f'models/Classifier/source_only_C.pt')
        D_path = os.path.join(f'models/Discriminator/source_only_D.pt')
        
        if os.path.exists(E_path) and os.path.exists(C_path):
            self.E.load_state_dict(torch.load(E_path))
            self.C.load_state_dict(torch.load(C_path))
            print('>> Load Encoder & Classifier Model')
            
            if os.path.exists(D_path):
                self.D.load_state_dict(torch.load(D_path))
                print('>> + Load Discriminator Model')

            return True
        else:
            print('<< No Encoder & Classifier Model checkpoint...')
            return False        
        
        
    def _train_source(self, epochs):
        self.model.train()
        optim = optims.Adam(
            list(self.C.parameters()) + 
            list(self.E.parameters()), lr=0.01)
        
        update_interval = 10
        print(">> Only-Source Domain Training Start")    
        for epoch in range(epochs):
            start_steps = epoch * len(self.dataset_source)
            total_steps = epochs * len(self.dataset_source)
            bar = tqdm(enumerate(self.dataset_source), total=len(self.dataset_source))
            for i, (data, labels) in bar:
                source_data = torch.cat((data, data, data), 1) # MNIST convert to 3 channel
                source_data, source_labels = source_data.to(self.device), labels.to(self.device)
                
                optim = optim_scheduler(optim=optim, p = float(i + start_steps) / total_steps)
                optim.zero_grad()
                
                source_feature = self.E(source_data) # source_feature : [32, 512, 1, 1] 
                cls_predicted = self.C(source_feature.view(source_feature.size(0), -1)) # cls_predicted : [32, 10]
                
                classification_loss = self.CELoss(cls_predicted, source_labels)
                classification_loss.backward()
                optim.step()
                
                self.writer.add_scalar('Loss/Classification', classification_loss, epoch)
                if i % update_interval == 0 :
                    bar.set_description("Epoch: {}/{} - Class Loss: {:.6f}".format(epoch+1, epochs, classification_loss.item()), refresh=True)
        
            if (epoch + 1) % 2 == 0:
                self._test_source()
        
        self.writer.close()
        modelSave(self.E, self.C, self.D, 'source_only')
    
    def _test_source(self):
        self.E.eval()  
        self.C.eval() 
        print(">> Only-Source Domain Test Start")
        correct_source, total_source = 0, 0
        with torch.no_grad():
            for data, labels in self.dataset_source:
                source_data = torch.cat((data, data, data), 1)
                source_data, labels = source_data.to(self.device), labels.to(self.device)
                
                features = self.E(source_data) 
                features = features.view(features.size(0), -1) 
                outputs = self.C(features) 
                _, predicted = outputs.max(1)
                
                total_source += labels.size(0)
                correct_source += predicted.eq(labels).sum().item()
                
        print(f'Source Domain Accuracy: {100 * correct_source / total_source:.2f}%')
        
    def _train_dann(self, epochs):
        optim = optims.Adam(
            list(self.C.parameters()) + 
            list(self.E.parameters()) + 
            list(self.D.parameters()), lr=0.01)
        
        update_interval = 50
        print(">> Domain-Adversarial Training Start")
        for epoch in range(epochs):
            print('Epoch : {}'.format(epoch))
            start_steps = epoch * len(self.dataset_source)
            total_steps = epochs * len(self.dataset_target)
            bar = tqdm(enumerate(zip(self.dataset_source, self.dataset_target)), total=min(len(self.dataset_source), len(self.dataset_target)))
            for i, (source_data, target_data) in bar:
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
                
                source_feature = self.E(source_data) # source_feature : [32, 512, 1, 1]
                total_feature = self.E(total_data)   # total_feature : [64, 512, 1, 1]
                total_feature = total_feature.view(total_feature.size(0), -1) # total_feature : [64, 512]
                
                cls_predicted = self.C(source_feature.view(source_feature.size(0), -1)) 
                cls_loss = self.CELoss(cls_predicted, source_labels)
                
                domain_predicted = self.D(total_feature, alpha)
                domain_source_labels = torch.zeros(source_labels.shape[0]).type(torch.LongTensor)
                domain_target_labels = torch.ones(target_labels.shape[0]).type(torch.LongTensor)
                domain_total_labels = torch.cat((domain_source_labels, domain_target_labels), 0)
                domain_total_labels_onehot = F.one_hot(domain_total_labels, num_classes=2).type(torch.FloatTensor).to(self.device)
                domain_loss = self.BCELoss(domain_predicted, domain_total_labels_onehot)
                
                total_loss = cls_loss + domain_loss
                total_loss.backward()
                optim.step()
                
                if i % update_interval == 0 :
                    bar.set_description("Epoch: {}/{} - Class Loss: {:.6f} - Domain Loss: {:.6f} - Total Loss: {:.6f}".format(epoch+1, epochs, cls_loss.item(), domain_loss.item(), total_loss.item()), refresh=True)

                if (epoch + 1) % 10 == 0:
                    self._test_dann()
        
        modelSave(self.E, self.C, self.D, '_dann', isDann=True)
        
    def _test_dann(self):
        self.E.eval() 
        self.C.eval() 
        print(">> Domain-Adversarial Test Start")
        correct_source, total_source = 0, 0
        correct_target, total_target = 0, 0
        with torch.no_grad():          
            for images, labels in self.dataset_source:
                images, labels = images.to(self.device), labels.to(self.device)
                
                features = self.E(images) 
                features = features.view(features.size(0), -1) 
                outputs = self.C(features) 
                _, predicted = outputs.max(1)
                
                total_source += labels.size(0)
                correct_source += predicted.eq(labels).sum().item()
                
                print(f'> | Source Domain Accuracy: {100 * correct_source / total_source:.2f}%')
                
            for images, labels in self.dataset_target:
                images, labels = images.to(self.device), labels.to(self.device)
                
                features = self.E(images) 
                features = features.view(features.size(0), -1) 
                outputs = self.C(features) 
                _, predicted = outputs.max(1)
                
                total_target += labels.size(0)
                correct_target += predicted.eq(labels).sum().item()

                print(f'> | Target Domain Accuracy: {100 * correct_target / total_target:.2f}%')

            print(f'> | Total Domain Accuracy: {100 * (correct_source + correct_target) / (total_source + total_target):.2f}%')


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )
    
    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return F.softmax(x, dim=1)


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