# 自动微分机制
**反向传播算法**
*根据loss相对于给定参数的梯度来调整parameters(模型权重)*

*为了计算这些梯度，PyTorch有一个内置的微分引擎，名为torch.autograd。*
*它支持任何计算图的梯度自动计算。*

**example**

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

