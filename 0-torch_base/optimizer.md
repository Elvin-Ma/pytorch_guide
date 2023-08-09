# 优化器展示


## 1 adam
[参考链接](https://pytorch.org/docs/master/generated/torch.optim.Adam.html?highlight=adam#torch.optim.Adam)

```python
CLASS torch.optim.Adam(
    params, # 可迭代的parameters, 或者是装有parameter组的字典
    lr=0.001, # 学习率
    betas=(0.9, 0.999), # 用于计算梯度及其平方的移动平均值系数
    eps=1e-08, # 防止分母为0
    weight_decay=0, # 权重衰减系数，L2 惩罚， 默认为0
    amsgrad=False, # 是否使用该算法的amsgrad 变体
    *,
    foreach=None, #
    maximize=False, # 最大化梯度--> 梯度提升
    capturable=False, # 在CUDA图中捕获此实例是否安全
    differentiable=False, # 是否可进行自动微分
    fused=None # 是否使用融合实现(仅CUDA)。
)

  add_param_group(param_group)
  '''
  用于向优化器中添加新的参数组；
  参数组：一组共享相同超参数（学习率、权重衰减等）的模型参数；
  通过定义不同的参数组，可以为模型的不同部分或不同层，设置不同的超参数；
  这在微调预训练的网络时很有用.
  '''

```

*添加参数组
```python
import torch
import torch.optim as optim

# 创建模型和优化器
model = torch.nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 创建新的参数组
new_params = [{'params': model.parameters(), 'lr': 0.01}]

# 将新的参数组添加到优化器中
optimizer.add_param_group(new_params)
```

## 2 sgd