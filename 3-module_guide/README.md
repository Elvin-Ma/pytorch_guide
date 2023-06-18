# mudule模块官方地址 ：
[源码地址](https://github.com/pytorch/pytorch/tree/270111b7b611d174967ed204776985cefca9c144/torch/nn)
[doc 地址](https://pytorch.org/docs/stable/nn.html)
**上节课的autograd 我们是基于 function 来实现的**
**layer层(operator)：神经网络中的层，是在module 下面的.**
** pytorch 自带的模块大概23类左右: 都继承自nn.Module**
- conv.py
- linear.py
- rnn.py
- transformer.py
- dropout.py
- activition.py
- loss.py 
- ...

# torch.nn.Module
- class Module: 所有这些模块的父类；
- class Module 完成了我们深度学习中的通用行为；
- 

# 写一个自己的model 步骤：
- 1. 继承自nn.Module
- 2. __init__ 函数：实例化一些标准的layer，
- 3. __inti__函数中必须初始化父类（super(FullConnect，self).__init__())
- 4. forward: 我们具体的实现过程，计算过程

# 自己 module 注意事项
- __init__ 中写的self.conv/ self.relu ... --> 也是一个类；
- Conv2d--> 同样继承自nn.Module
- 官方已经帮我们实现了很多的网络层（layer）；
- conv2d 的定义方式和定义我们自己网络的方式是一样的；
- 递归的调用，方便我们搭建网络；
- weight (Parameter)： activation(Tensor)
- nn.Moule 类是 所有神经网络（neural network）的基类
- torch.nn 里的大部分 nn 都需要继承它
[torch.nn官方地址](https://pytorch.org/docs/stable/nn.html)
- 通过继承 nn.module 来定义我们的神经网络；
- 在 __init__ 中初始化 neral network;
- 在 forward 方法中 指定具体的计算流程.

# moudle参考文档：
[nn.Module地址](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py)
**基于nn.Module torch 实现了很多现有的模块**
[torch.nn官方地址](https://pytorch.org/docs/stable/nn.html)
**上述每一个nn（conv2d Linear）我们都可以认为是pytorch帮我们封装好的算子(初始化后可直接运行)

# module 作用
- 帮助我们搭建神经网络
- 那个是weight 那个是activation
- 模型的保存和加载需要它
- 权重初始化需要它

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

# 需要注意的几点：
- 必须继承nn.Module （因为要完成一些统一的操作）
- __init__ 构造函数（初始化父类、定义一些网络层）<--> 算子定义
- forward 函数：自定义网络前向图
- pytorch自带的网络层：conv linear Relu 也都继承自nn.Module
- torch.nn 代码地址：
[torch.nn 代码地址](https://github.com/pytorch/pytorch/tree/main/torch/nn/modules)
- 本地地址：
*https://github.com/pytorch/pytorch/tree/main/torch/nn/modules*
- nn.Module 可以是一种嵌套结构
- self.training = True （默认）
- nn.Module 两种模式：train() / eval() 验证集时使用

# nn.Module 类总结
- 路径：torch/nn/modules/module.py
**nn.Module属性总结**
- 1. self.training = True #  module : train() / eval()
- 2. self._parameters: Dict[str, Optional[Parameter]] = OrderedDict() # 存储weight的地方
- 3. self._modules --> 存储模块的地方
- self._buffers
- self._backward_hooks: Dict[int, Callable] = OrderedDict()
- self._forward_hooks: Dict[int, Callable] = OrderedDict()
- self._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
- self._state_dict_hooks: Dict[int, Callable] = OrderedDict()
- sefl._load_state_dict_pre_hooks --> 加载状态字典的时候的钩子函数

**nn.Moudle方法总结**
1. forward 的调用过程
**forward: Callable[..., Any] = _forward_unimplemented**
**__call__ = _call_impl --> net(Tensor)
** forward_call = self.forward
2. cuda、cpu
**将模块搬迁到cuda（GPU）上；
3. state_dict() # 保存模型的状态
4. load_state_dict() # 加载module时的方法
5. parameters() # 提取model的所有与 parameters
6. named_parameters() # 调用_named_members, 另外 被 parameters调用
7. chilren() --> 获取当前module的子moudule
8. train() / eval() --> 设置所有的module 的mode；
9. require_grad_() --> 统一设置所有parameters的requires_grad_;
10. zero_grad() --> 所有parameters梯度置0；
11. apply(fn) --> 遍历所有的模块，每个模块进操作（fn指定的操作）

**其它注意事项**
- requires_grad_ ： 设置 所有的parameters 的requires_grad 属性；
- zero_grad : p.grad.zero_() 把 所有 parameters的梯度设置成0；
- train ： 遍历所有子模块的 module，设置其 self.training
- eval： train(False) : model.eval() == model.train(False)
- children 和 named_children:  遍历 self._modules(nn.Module) 并返回所有模块
- parameters 和 named_parameters : 遍历module的 parameter的效果，
- state_dict: 获取module的所有 parameters --> {para_name: para_date}
- load_state_dict：加载 {para_name: para_date}；
- register_forward_hook：前向传播钩子注册函数
- register_forward_pre_hook：计算前钩子函数注册
- register_backward_hook
- to：完成整个模型的数据类型 或 device的转化
- cpu：
- cuda：
- get_parameter
- get_submodule
- add_module
- register_parameter
- add_module
- get_submodule
- apply：将函数递归应用到 每个module
- parameters: 返回parameters的迭代器
- buffers 中间 activation保存

# Parameter(torch.Tensor)
- torch.Tensor 的子类
- 作为Module的参数而存在
- 对参数进行操作（weight 或bias）
- 通过这个Parameter我们就可以把 参数 和act
- requires_grad 默认都是 true

# linear 案例分析
- super(Linear, self).__init__() 完成Module的初始化工作
- 矩阵相乘[m,k]*[k,n] = [m,n]
- self.in_features = k
- self.out_features = n
- self.weight = Parameter (权重是以Parameter的形式存在的)
- 完成了一个权重初始化的工作
- forward：直接调用funciton里的函数
*site-packages/torch/nn/functional.py*

# function 和 module 的区别
- module 里 最终调用的是 function（forward）；
- __init__ 是 module的 存储的地方。

# move neural network to device
```python
model = NeuralNetwork().to(device)
print(model) # 打印model
```

# 执行module
**自动调用Module的forward 方法**
```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X) # 不要直接调用forward --> 首先要调用_call_impl
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```
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
# Sequential 将多个 module 组合成一个module
- 实现一个块的串联功能；
- append ： 往sequential里增加模块：add_module
- [0]： 所有功能
- len 功能
- 注意：与 nn.ModuleList()的区别
  1. nn.ModuleList()没有 forward方法；
  2.不能直接用module()来run；
  3.这能一个一个run

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

# 设计Module 总结及注意事项：
- loss.backward() : 计算梯度 --> grad 保存在梯度里的；
- optimizer : 操作的对象就是 module 里的 Parameters（tensor的子类）
- optimizer.zero_grad() 和 module 的zero_grad都完成梯度清0；
- step() 更新参数
- 有Parameter的算子（module）一定要放到__init__里；
- relu 没有任何可学习参数的函数或module，放到forward里也ok；
- __init__中定义的层，在forward里重复使用会发生权重共享；

# 函数对比
torch.conv2d()
torch.nn.functional.conv2d() # 同 torch.conv2d, 底层调用的是同一个函数
torch.nn.conv2d # module 最终调用的是F.conv2d

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

**冻结一部分参数**
net.fc1.weight.requires_grad = False
