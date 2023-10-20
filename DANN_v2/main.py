from domain import Network
from dataset import MNIST_loaders

batch_size = 256
num_workers = 4

def main():
    source, target = MNIST_loaders(batch_size, num_workers, type='train')
    
    net = Network(source, target)
    if not net._load_model():
        net._train_source(epochs=10)
    net._train_dann(epochs=100)   

if __name__ == '__main__':
    main()