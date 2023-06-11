# pytorch doc
[pytorch doc](https://pytorch.org/docs/stable/index.html)

# pytorch auto grad 展示
![官网原理图](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

# pytorch backward 原理图
![auto grad](https://miro.medium.com/v2/resize:fit:640/format:webp/1*viCEZbSODfA8ZA4ECPwHxQ.png)

# auto grad 动态图
![auto grad dynamic](https://pytorch.org/assets/images/computational_graph_creation.gif)
# backward introduce
[autograd addr](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95)

[![Watch the video](https://pytorch.org/assets/images/computational_graph_backward_pass.png)](https://youtu.be/MswxJw-8PvE)

# 自动微分机制(auto grad) 重点：
- pytorch中 正向forward 对我们用户是可见的，但是backward对我们用户是不可见的；
- 一般情况下，每一个正向的函数，都对应一个反向的函数（grad_fn--> Tensor中）；
- tensor：requires_grad = True 
- tensor: grad --> tensor 中存储grad的地方；
- tensor: grad_fn --> 存储我们反向函数的地方
- tesnor: is_leaf --> 这个tensor 是不是 叶子节点；
- net::all weight --> 都是leaf
- 叶子节点的梯度会自动保存下来的（weight）；
- 中间的 activation 的梯度会计算，但是不保留；
- pytorch 动态图 vs tensorflow 静态图；
- 我们不能改变一个非叶子节点的 requires_grad;
- 非叶子（一个函数的output）节点它的 requires_grad 自动推导的；
- 非叶子节点对应函数的inputs 中只要有一个 requires_grad = True, 那么这个非叶子节点的requires_grad = True;
- torch.no_grad() 会使得里面的新的tensor requires_grad = False
- inplace的操作，非常大的风险：覆盖了原来的值，导致反向传播时计算不准确；
- 标量的梯度才能被隐式创建，隐式创建（.backward(1)）；
- 一般情况下，.backward(gradient)是有输入的: ;

#反向传播算法
- pytorch ：正向（forward） 和 反向 （backward）
- 反向 和 autograd 有密切的关系
- 因为反向求梯度的
- 根据loss相对于给定参数的梯度来调整parameters(模型权重)
- 为了计算这些梯度，PyTorch有一个内置的微分引擎，名为torch.autograd。
- 它支持任何计算图的梯度自动计算。
- 这个torch.autograd 对用户不可知

# 如何保存中间Tensor的梯度
```python
with torch.no_grad():
    v = y.view(5, 5)
    v.sub_(y.grad * 0.1) 

    # 底层的实现机制，sub_ 在c++ 层面上是经过多层的函数调用；
    # sub_ 底层会有一个新tensor的创建过程的；
    y.sub_(0.1 * y.grad) 
```
1. 使用retain_grad 保存中间tensor 的梯度
** retain_grad**
```python
for i in range(100):
    y.retain_grad()
    z = torch.matmul(y, x) + b # linear layer    
    output = torch.sigmoid(z)
    label = torch.Tensor([0, 0, 1, 0, 0])
    loss = (output-label).var() # l2 loss
    loss.backward()
    y = y - 0.1 * y.grad # y 中间计算过程 --> y 变为非叶子节点
```
2. grad_hook
```python
gard_list = []
def aa(grad):
    grad_list.append(grad)
    # return 0.001*grad

a = torch.Tensor([1, 2, 3])
a.regiter_hook(aa)
b = a.mul(c)
b.backward()
```

# tensor 的梯度
- requires_grad: 设置我们是否需要求这个tensor的梯度
- [requres_grad 链接接](site-packages/torch/_C/_VariableFunctions.pyi)
- 只有浮点、复数可以设置 require grdients
- reqires_grad: 告诉torch 我这个tensor 需要计算梯度的，正向的时候，torch会做额外的一些事情；
- requires_grad: default = False
- 输入的tensor 只要有一个requires_grad=True, output.requires_grad = True
- 中间的tensor 的 梯度会计算，但不保存
- weight：不释放，中间activation的梯度最好释放；
- 因为内存；GPU显存是很宝贵的。

#example：全连接层梯度求解
```python
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
loss.backward()
print(w.grad)
print(b.grad)
```

# 反向求导原理
- grad_fn(grad function): 反向传播用到的函数；
- grad_fn 在正向的时候自动生成（requires_grad=True时)；
- .backend(）触发反向求导操作
- 求导操作针对整个反向图来进行，而不是仅仅一个算子；
- 冻结权重：.requires_grad = False
- weight 是需要计算权重并保存梯度的, matmul等算子如何设置weight和act???

**关闭梯度求导**

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

**require_grad的自动机制**
```python
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")
```

**前向传播发生的两件事**
- 运行请求的操作来计算结果张量；
- 在DAG中保存算子的梯度函数；
- directed acylic graph

**动态图机制**
- 每次运行.backward()后，autogard模块才会填充一个新的图；
- 上述正是模型中可以使用控制流语句的原因；
- 如有必要：可以在每次迭代中更改形状、大小和操作

**输出非标量时**
```python
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

# auto grad 问题解答
- 默认保存梯度的是 叶子节点: 整个网络最外层的tensor
- 网络的input(是)，网络中的weight(是)，网络中的bias(是)
- z = torch.matmul --> 算子（中间activation）
- .backward(上一层的梯度)
- loss 的 backward 不需要传入梯度, 直接.backward()就可以
- 对谁（tensor）进行backward: 从 Tensor 开始启动的
- 反向图: torch.autograd.backward

# 梯度累加 和 梯度清0**
- 多条路径：梯度累加 --> 正确的
- 多次run backward的梯度累加 --> 我们应该避免
- 怎么样避免： tensor.grad.zero_() 
- 下节课：model.zero_grad()

**中间tensor的梯度控制**
- middle_tensor.register_hook(hook)
- register_hook : 钩子注册函数
- hook: 钩子。类型：函数（lambda 函数 和 自定义函数都可以）
- hook 函数有一个参数：grad
- 这个grad 就是上一层传下来的梯度值
- 拿到这个梯度值，就可以进行操作（改变、保存）
- mid_tensor.retain_grad() ：不释放tensor的梯度

# 大部分正向算子都对应一个反向算子
[对应关系](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)

# customer 自定义自己的反向传播函数
```python
 class Exp(Function):
     @staticmethod
     def forward(ctx, i):
         result = i.exp()
         ctx.save_for_backward(result)
         return result

     @staticmethod
     def backward(ctx, grad_output):
         result, = ctx.saved_tensors
         return grad_output * result

 # Use it by calling the apply method:
 output = Exp.apply(input)
 ```
 # auto grad 机制不足
 - 用起来不方便，搭建模型也不方便；
 - 哪些参数需要设置requires_grad？
 - 保存模型：保存哪些东西？
 - matmul：那一个是weight 哪一个是activation呢？ 
 - weight 的初始化；
 - 解决思路：nn

 # examples
 - /tutorials/beginner_source/examples_autograd/polynomial_custom_function.py

 # 扩展 extension（不需要去看）
 - grad_fn 和 grad 等相关信息 是放到Tensor中的；
  
 ## autograd是什么
- variable.h: struct TORCH_API AutogradMeta
- grad_ ：存储当前Variable实例的梯度，本身也是一个Variable。
- grad_fn ：是个Node实例，非叶子节点才有。通过 grad_fn() 方法来访问，实际上，PyTorch中就是通过 grad_fn是否为空 来判断一个Variable是否是leaf variable。
- grad_accumulator_ ：也是Node的实例，只有叶子节点才有。
- requires_grad_ ：表明此Variable实例是否需要grad。
- retains_grad_ ： 只有非叶子节点才有意义，意义为是否需要保持图。
- is_view_ ：是个flag，表明此Variable实例是否是个view（没有实际存储，基于base的variable）。
- version_counter_ ：version number。
- output_nr_：是个数字。output_nr_表明是 Node 的第几个输出，比如为 0 就 表明这个Variable是Node 的第 1 个输出。
- base_ ：是view的base variable。

## tensor 中的autogradmeta**
- 每个tensor都有一个 autogradmeta：
- TensorBase.h：c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
- std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;

## grad_fun
- grad_fn 属性还包含 _saved_self 和 _saved_other 两个属性;
- _saved_self: 保存计算当前张量梯度所需要的计算图中的自身张量；
- _saved_other: 保存计算当前张量梯度所需要的计算图中的其他张量;