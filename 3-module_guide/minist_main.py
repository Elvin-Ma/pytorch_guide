from __future__ import print_function
import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

torch.nn.BatchNorm2d(64)

# 1. model: 类：继承自 torch.nn.Module
class Net(nn.Module):
    # 2. 自己的__init__
    def __init__(self):
        # 3. 初始化父类
        super(Net, self).__init__()
        # 4. 实例化我们需要用到的模块 (在init里初始化)
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # 1.构造了weight 和 bias；2. 完成了科学的初始化；3. requires_grad 自动设置
        self.conv3 = nn.Conv2d(1, 32, 3, 1) # 
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.flatten = nn.Flatten(1)
        self.relu = nn.ReLU()
        # nn.Conv2d() --> F.conv2d --> torch.conv2d ---> torch/_C/_VariableFunctions.pyi::conv2d

      # 5. 实现我们自己的forward
    def forward(self, x):
        # self.conv2d = nn.Conv2d(1, 32, 3, 1)
        # self.relu = nn.ReLU() # 没有任何可学习参数的，可以放到forward里
        # forward 里我们callable
        x = self.conv1(x)
        x = self.conv3(x) # 
        x = self.relu(x)
        x = self.conv2(x)
        x = F.relu(x) # 
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        # 6. 把最终的结果返回
        return output 

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 64, 5, 1, 2) # 子模块
        self.relu1 = nn.SELU() # 直接用官方自己的module
        self.bn1 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(21632, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # self.conv3 = nn.Conv2d(1, 64, 5, 1, 2) # 万万不可以
        # conv4 = nn.Conv2d(1, 64, 5, 1, 2)
        x = self.conv1(x)
        # x = self.conv5(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        x = torch.selu(x) # 可以的
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.net = Net2()
        
    def forward(self, x):
        return self.net(x)

def train(args, model: Net2, device, train_loader, optimizer: torch.optim.Optimizer, epoch):
    model.train() # mode 模式改为 train 模式
    for batch_idx, (data, target) in enumerate(train_loader): # 变量数据集
        data, target = data.to(device), target.to(device) # input data -> device
        optimizer.zero_grad() # 梯度清0
        output = model(data) # 前向计算 
        loss = F.nll_loss(output, target) # loss
        loss.backward() # 反向传播 --> 完成梯度的计算
        optimizer.step() # 更新参数 --> 更新参数
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

# @torch.no_grad
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # no_grad
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    # elif use_mps:
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 数据准备
    # 1. transform 数据类型
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]) # 转化--> 对数据进行前处理
    
    # 2. 数据下载以及处理： dataset1 属性类型是什么？
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform) # torch.utils.data.Dataset --> 
    
    # 3. train_loader 是什么数据类型
    # pytorch的接口
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs) # 真正的取数据
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # 模型准备
    model = Net().to(device) # 1. 实例化一个模型，并且把模型加载到device ？ 
    
    # 优化器
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) # 
    
    # 变量 epoch 开始训练： 一个数据集 跑完一遍 叫一个 epoch
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step() # 学习率更新 是按照 epoch 来进行的
        
    # x = torch.rand(1, 1, 28, 28).to(device)
    # torch.onnx.export(model, x, "minist.onnx")

    if args.save_model:
        torch.save(model, "mnist_cnn.pt")

def parameters_demo():
    a = Net3()
    b = Net2()
    c = list(b.named_parameters())
    # c = list(b.named_children())
    d = b._modules
    dd = b._parameters
    f = b.conv1._parameters
    stat_dict = a.state_dict()
    output = b(torch.randn(4, 1, 28, 28))
    # print(stat_dict)
        
    # print(c)
    # print("======================")
    print(f)
    print(stat_dict)
    
def function_demo():
    tensor0 = torch.rand(4, 1, 28, 28)
    a = torch.relu(tensor0) # 方式1 --> 最原始的方式

    b = nn.ReLU()(tensor0) #方式2 -->在forward里调用我们的方式3
    
    c = F.relu(tensor0) # 方式3 --> 调用我们的方式1，也可能直接调用方式1
    
    # 说白了：torch里面函数啊，转来转去，底层就一份函数
    
    print(a.shape)
    print(b.shape)
    print(c.shape)

def container_demo():
    # model = nn.Sequential(
    #               nn.Conv2d(1,20,5, padding=2),
    #               nn.ReLU(),
    #               nn.Conv2d(20,64,5, padding=2),
    #               nn.ReLU()
    #         )
    x = torch.rand(4, 1, 112, 112)
    # output = model(tensor)
    
    linears = nn.ModuleList([nn.Conv2d(1,20,5, padding=2),
            nn.ReLU(),
            nn.Conv2d(20,64,5, padding=2),
            nn.ReLU()])
    # output = linears(tensor)
    for layer in linears:
        x = layer(x)
        
    print("output shape: ", x.shape)  
    
if __name__ == '__main__':
    main()
    # parameters_demo()
    # function_demo()
    # container_demo()
    # torch.Tensor()
    print("run minist_main.py successfully !!!")
