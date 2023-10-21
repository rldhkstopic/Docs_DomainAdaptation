from domain import Network
from dataset import DataSet

batch_size = 128
num_workers = 4

def main():
    set = DataSet(source='mnist', target='svhn', batch_size=batch_size, num_workers=num_workers, type='train')
    source, target = set.source_loader, set.target_loader
    
    net = Network(source, target)
    net._load_model()
    net._test_source()
    
    if not net._load_model():
        net._train_source(epochs=20)
    
    net._train_dann(epochs=100)   

if __name__ == '__main__':
    main()