from domain import Network
from dataset import (mnist_train_loader, 
                    mnist_test_loader,
                    mnistm_train_loader,
                    mnistm_test_loader)

def main():
    train_source, _ = mnist_train_loader, mnist_test_loader    
    train_target, _ = mnistm_train_loader, mnistm_test_loader
    
    network = Network(train_source, train_target)
    
    network._train_source()
    network._train_dann()    

if __name__ == '__main__':
    main()