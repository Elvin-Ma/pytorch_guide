# pytorch doc
[pytorch doc](https://pytorch.org/docs/stable/index.html)

# 自动微分机制(auto grad)
- 求梯度
- 需不需要我们自己来求 no 
- auto grad 是pytorch非常强大的功能（动态图）
- pytroch 动态图 和 tensorflow 静态图

#反向传播算法
- pytorch ：正向（forward） 和 反向 （backward）
- 反向 和 autograd 有密切的关系
- 因为反向求梯度的
- 根据loss相对于给定参数的梯度来调整parameters(模型权重)
- 为了计算这些梯度，PyTorch有一个内置的微分引擎，名为torch.autograd。
- 它支持任何计算图的梯度自动计算。
- 这个torch.autograd 对用户不可知

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

# autograd 原理
![官网原理图](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

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

**梯度清0**
- x.grad.zero_()
- model.zero_grad()

**获取中间tensor的梯度**
- mid_tensor.retain_grad()
- mid_tensor.register_hook(save_grad)

# 自定义自己的反向传播函数
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

 # examples
 - /tutorials/beginner_source/examples_autograd/polynomial_custom_function.py

 # 扩展
 **autograd是什么**
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

**tensor 中的autogradmeta**
- 每个tensor都有一个 autogradmeta：
- TensorBase.h：c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
- std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;

**grad_fun**
- grad_fn 属性还包含 _saved_self 和 _saved_other 两个属性;
- _saved_self: 保存计算当前张量梯度所需要的计算图中的自身张量；
- _saved_other: 保存计算当前张量梯度所需要的计算图中的其他张量;

