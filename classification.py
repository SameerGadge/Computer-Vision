import torchvision
import PIL
from IPython.display import display
from torchvision import transforms
from torchvision.datasets import mnist

train_dataset = mnist.MNIST(root='./data', train=True, download=True)
test_dataset = mnist.MNIST(root='./data', train=False, download=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset.transform = transform
test_dataset.transform = transform

x_transformed = train_dataset[0]
