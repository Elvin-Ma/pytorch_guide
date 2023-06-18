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
  for n_iter in range(200, 300):
      # writer.add_scalars('Loss/train', {"a":n_iter * 2, "b": n_iter*n_iter}, n_iter)
      writer.add_scalar('Loss/test1', 200, n_iter)
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
  
def add_graph():
  import torchvision.models as models
  net = models.resnet50(pretrained=False)
  writer = SummaryWriter("graph_log")
  writer.add_graph(net, torch.rand(16, 3, 224, 224))
  writer.flush()
  writer.close()
  
def add_images():
  img_batch = np.zeros((16, 3, 100, 100))
  for i in range(16):
      img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
      img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

  writer = SummaryWriter("image_log")
  writer.add_images('my_image_batch', img_batch, 0)
  writer.flush()
  writer.close()
  
def add_image():
  import numpy as np
  img = np.zeros((1, 100, 100))
  img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
  # img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

  # img_HWC = np.zeros((100, 100, 3))
  # img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
  # img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

  writer = SummaryWriter("image_log")
  writer.add_image('my_image', img, 0)

  # If you have non-default dimension setting, set the dataformats argument.
  # writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
  writer.close()
  
  
def add_embedding_v2():
  import keyword
  import torch
  meta = []
  while len(meta)<100:
      meta = meta+keyword.kwlist # get some strings
  meta = meta[:100]

  for i, v in enumerate(meta):
      meta[i] = v+str(i)

  label_img = torch.rand(100, 3, 10, 32)
  for i in range(100):
      label_img[i]*=i/100.0

  writer = SummaryWriter("emb_log")
  # writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
  writer.add_embedding(torch.randn(100, 5), label_img=label_img)
  # writer.add_embedding(torch.randn(100, 5), metadata=meta)

if __name__ == "__main__":
  # add_scalar()
  # add_image()
  # net_loss()
  # add_embedding()
  # add_graph()
  # add_images()
  # add_image()
  add_embedding_v2()
  print("run hello_tensorboard.py successfully !!!")