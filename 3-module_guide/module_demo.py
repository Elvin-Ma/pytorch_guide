import torch
import torch.nn as nn
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # 一层算子
        aa = issubclass(nn.Flatten, nn.Module)
        print("aa: ", aa)
        # 算子序列
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def run_network():
    model = NeuralNetwork()
    aa = model.state_dict()
    print("module state_dict: ", aa.keys())
    # bb = issubclass(NeuralNetwork, nn.Module)
    # print("bb : ", bb)
    # input = torch.randn(1, 28, 28)
    # output = model(input) 
    # print("output data: ", output)
    
def forward_demo():
    X = torch.rand(1, 28, 28)
    model = NeuralNetwork()
    logits = model(X) # 不要直接调用forward
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    print("run forware_demo successfully !!!")
    
def sequential_demo():
    seq_modules = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
        
    input_image = torch.rand(3,28*28)
    logits = seq_modules(input_image)
    print("logits: ", logits.shape)
    nn.ModuleList()
    
def model_end2end():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 5)
            # self.conv1.weight.requires_grad = False
            self.conv2 = nn.Conv2d(6, 16, 5)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # If the size is a square, you can specify with a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net() # requires_grad 不需要太关系
    print(net)

    params = list(net.parameters())
    print(len(params)) # 可学习的参数
    print(params[0].size())  # conv1's .weight

    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    target = torch.randn(10)  #fake data:  a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss() # 损失函数定义

    loss = criterion(output, target) # 计算损失
    net.zero_grad()  # zeroes the gradient buffers of all parameters
    loss.backward() # 计算出来所需要的梯度
    
    optimizer = optim.Adam(net.parameters())
    # optimizer.zero_grad() # 梯度清0 ：
    # net.zero_grad()
    optimizer.step() # 更新参数
    print(loss) 
    # print(loss.grad_fn)  # MSELoss
    # print(loss.grad_fn.next_functions[0][0])  # Linear
    # print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    # 梯度是累计的
    # print('conv1.bias.grad before backward')
    # print(net.conv1.bias.grad)
    # print('conv1.bias.grad after backward')
    # print(net.conv1.bias.grad)
   
def function_compare():
    weight = torch.randn(32, 3, 3, 3)
    input = torch.randn(1, 3, 224, 224)
    # torch.nn.Conv2d() # module 类
    F.conv2d() # 也是普通的函数
    
    #联系：module 里 最后调用的也是 F.conv2d
    #torch.conv2d() 和 F.conv2d() 是同一个东西
    
class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.fc1 = nn.Linear(10, 10)  # 5*5 from image dimension
        self.fc2 = nn.Linear(10, 84)
        self.fc3 = nn.Linear(84, 10)
        self.data = []
        self.data_grad = []
        self.fc1.register_forward_pre_hook(self.forward_pre_hook)
        self.fc1.register_forward_hook(self.forward_hook)
        self.fc1.register_full_backward_hook(self.backward_hook)
        
    def forward(self, input):
        x = self.fc1(input)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def forward_pre_hook(self, module, input):
        self.data.append(input)
        print("===============data: {}\n".format(input[0][0]))
        # return input * 2
        
    def forward_hook(self, module, input, output):
        self.data.append(output)
        # print("===========fc1 output: ", output)
        
    def backward_hook(self, module, grad_input, grad_output):
        print("=================backward_hook")
        self.data_grad.append((grad_input, grad_output))     
        
def module_hook_demo():
    model = A()
    input = torch.randn(5, 10)
    output = model(input)
    output.backward(torch.ones_like(output))
    # print("model.data count: {}\n".format(len(model.data)))
    print("model.data_grad count: {}\n".format(len(model.data_grad)))
    print("output shape: ", output.shape)
     
# model
class FullConnect(nn.Module):
    def __init__(self, k, n):
        super(FullConnect, self).__init__() # 初始化父类
        self.full_connect1 = nn.Linear(30, 20) 
        self.full_connect2 = nn.Linear(20, 10)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()
        
    def forward(self, input):
        x = self.full_connect1(input) # type(input) = Tensor
        x = self.activation1(x)
        x = self.full_connect2(x)
        x = self.activation2(x)
        
        return x
           
def full_connect_demo():
    model = FullConnect(30, 10) # model的实例化
    input = torch.rand(4, 30) # 我们拿不到input的梯度
    
    loss_function = nn.CrossEntropyLoss()
    lable = torch.Tensor([3, 4, 2, 4]).to(torch.int64)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    for i in  range(100):
        model.train()
        optimizer.zero_grad()      
        output = model(input) # output: float
        loss = loss_function(output, lable)
        loss.backward()
        # print("================model weight grad before update: ", model.full_connect1.weight[0])
        optimizer.step()
        print("=======loss: ", loss)
        # print("================model weight grad before update: ", model.full_connect1.weight[0])
 
if __name__ == "__main__":
    # run_network()
    # forward_demo()
    # sequential_demo()
    # model_end2end()
    # function_compare()
    # module_hook_demo()
    full_connect_demo()
    
    print("run module_demo.py successfully !!!")