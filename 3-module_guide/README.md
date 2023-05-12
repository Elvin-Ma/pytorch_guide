# torch.nn
[torch.nn官方地址](https://pytorch.org/docs/stable/nn.html)

- 通过继承 nn.module 来定义我们的神经网络；
- 在 __init__ 中初始化 neral network;
- 在 forward 方法中 指定具体的计算流程.

# code example
```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
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
```

# move neural network to device
```python
model = NeuralNetwork().to(device)
print(model) # 打印model
```

# 执行module
**自动调用Module的forward 方法**
```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X) # 不要直接调用forward
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```
# Module 官方地址
[Module 官方实现](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866)

# 内置的 module
```python
input_image = torch.rand(3,28,28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```
# module 序列
[Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

**code**
```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```
# 模型中的 parameters
```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

**训练一个神经网络的典型步骤**
- 定义具有一些可学习参数(或重量)
- 迭代输入数据集
- 通过网络处理输入
- 计算损失
- 将梯度传播回网络的参数
- 更新网络权重

# one end-to-end example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
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

net = Net()
print(net)

params = list(net.parameters())
print(len(params)) # 可学习的参数
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad() # why
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# 梯度是累计的
net.zero_grad()  # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

# parameters 进阶
- torch module的一个重要的行为是注册 parameters；
- 如果一个module 子类 有可学习的权重的话，这些权重以 torch.nn.parameter 实例的形式体现；
- Parameter 是 torch.Tensor 的一个子类；
- Parameter 性质：当 parameter 被 作为一个Module的 属性的话，就会被添加到module的 parameters的列表中；

**parameters代码展示**
```python
import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)

```

# bias 初始化
- bias 通常初始化较小
- bias 不参与权重更新，不需进行特殊的初始化来避免梯度消失或梯度下降

# extension 扩展
- self.training: 处于训练模式还是 inference 模式
- self._parameters 和 self._buffers：分别表示模型中的可学习参数和非可学习参数, 非可学习参数通常是某些计算所需的缓存变量，如均值和方差等。
- self._modules: 表示模型的子模块，也就是其他 nn.Module 对象。子模块可以嵌套在父模块中，形成复杂的神经网络结构;
- self._forward_pre_hooks 和 self._backward_hooks: 模型正向传播和反向传播的钩子函数。钩子函数是一种在模型计算过程中执行的自定义函数，可以用于实现各种调试和监控功能。
- _non_persistent_buffers_set : 存储模型中非持久化缓存（non-persistent buffer）的名称。
- _is_full_backward_hook : 指示当前模块是否注册了完整的反向传播钩子;
- _forward_pre_hooks : 模型的输入通过某一层之前被调用；
- _forward_hooks : 模型的的输出张量通过某一层之后被调用；
- _state_dict_hooks：存储模型中所有的状态字典钩子函数 register_state_dict_hook；
- _load_state_dict_pre_hooks：存储模型中所有加载状态字典前的钩子函数；、
- _backward_hooks：反向传播钩子可以在模型的输入梯度或输出梯度通过某一层之后被调用，从而可以获取梯度信息并对其进行处理或记录。

**_backward_hooks 案例**
```python
import torch
from torch import nn

# 定义一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义一个反向传播钩子函数
def my_backward_hook(module, grad_input, grad_output):
    print(f"backward hook called for {module}")
    print(f"grad_input: {grad_input}")
    print(f"grad_output: {grad_output}")

# 创建模型和输入数据
model = MyModel()
x = torch.randn(1, 10)

# 注册反向传播钩子
handle = model.fc1.register_backward_hook(my_backward_hook)

# 执行模型前向传播和反向传播
y = model(x)
loss = y.mean()
loss.backward()

# 移除反向传播钩子
handle.remove()
```

**_forward_pre_hooks 案例**
```python
import torch
from torch import nn

# 定义一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义一个前置钩子函数
def my_forward_pre_hook(module, input):
    print(f"forward pre-hook called for {module}")
    print(f"input: {input}")

# 创建模型和输入数据
model = MyModel()
x = torch.randn(1, 10)

# 注册前置钩子
handle = model.fc1.register_forward_pre_hook(my_forward_pre_hook)

# 执行模型前向传播
y = model(x)

# 移除前置钩子
handle.remove()
```

**冻结一部分参数**
requires_grad

