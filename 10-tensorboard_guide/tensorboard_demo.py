import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim

def add_scalar():
  writer = SummaryWriter("scalar_log")
  for n_iter in range(100,200):
      writer.add_scalars('Loss/train', {"a":n_iter * 2, "b": n_iter*n_iter}, n_iter)
      # writer.add_scalar('Loss/test1', n_iter + 3, n_iter)
      # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
      # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
      
def add_image():
  # Writer will output to ./runs/ directory by default
  # --logdir=./runs
  writer = SummaryWriter("mtn_log")

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
  model = torchvision.models.resnet50(False)
  torch.onnx.export(model, torch.randn(64, 3, 224, 224), "resnet50_ttt.onnx")
  # Have ResNet model take in grayscale rather than RGB
  model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
  images, labels = next(iter(trainloader)) # 拿到 输入 和label
  
  print("============images shape: ", images.shape)
  output = model.conv1(images)
  output = output[:, 0, :, :].reshape(64, 1, 14, 14).expand(64, 3, 14, 14)
  print("============output shape: ", output.shape)
  
  
  grid = torchvision.utils.make_grid(images)
  grid = torchvision.utils.make_grid(output)
  writer.add_image('output', grid, 0) # 保存图片
  # writer.add_graph(model, images) # 保存模型
  writer.close()

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
  # add_scalar()
  add_image()
  # net_loss()
  # add_embedding()
  print("run hello_tensorboard.py successfully !!!")