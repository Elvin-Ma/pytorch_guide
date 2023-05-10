import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim

def add_image():
  # Writer will output to ./runs/ directory by default
  # --logdir=./runs
  writer = SummaryWriter("mtn_log")

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
  model = torchvision.models.resnet50(False)
  # Have ResNet model take in grayscale rather than RGB
  model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
  images, labels = next(iter(trainloader))

  grid = torchvision.utils.make_grid(images)
  writer.add_image('images', grid, 0)
  writer.add_graph(model, images)
  writer.close()

def add_scalar():
  writer = SummaryWriter("scalar_log")
  for n_iter in range(100):
      writer.add_scalars('Loss/train', {"a":np.random.random(), "b":np.random.random()}, n_iter)
      # writer.add_scalar('Loss/test', np.random.random(), n_iter)
      # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
      # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def add_embedding():
  transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

  training_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)

  training_loader = torch.utils.data.DataLoader(training_set,
                                                batch_size=4,
                                                shuffle=True,
                                                num_workers=2)

  # select random images and their target indices
  images, labels = select_n_random(training_set.data, training_set.targets)

  # get the class labels for each image
  classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
  class_labels = [classes[lab] for lab in labels]

  # log embeddings
  features = images.view(-1, 28 * 28)
  writer = SummaryWriter("embedding_log")

  writer.add_embedding(features,
                      metadata=class_labels,
                      label_img=images.unsqueeze(1))
  writer.flush()
  writer.close()

if __name__ == "__main__":
  # net_loss()
  # add_image()
  # add_scalar()
  add_embedding()
  print("run hello_tensorboard.py successfully !!!")