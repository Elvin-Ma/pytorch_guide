# 1. DP（DataParalle）Summary

## 数据并行的概念
当一张 GPU 可以存储一个模型时，可以采用数据并行得到更准确的梯度或者加速训练：<br>
即每个 GPU 复制一份模型，将一批样本分为多份输入各个模型并行计算。<br>
因为求导以及加和都是线性的，数据并行在数学上也有效。<br>

## DP原理及步骤
- Parameter Server 架构 --> 单进程 多线程的方式 --> 只能在单机多卡上使用;
- DP 基于单机多卡，所有设备都负责计算和训练网络；
- 除此之外， device[0] (并非 GPU 真实标号而是输入参数 device_ids 首位) 还要负责整合梯度，更新参数。
- 大体步骤：
1. 各卡分别计算损失和梯度；
2. 所有梯度整合到 device[0]；
3. device[0] 进行参数更新，其他卡拉取 device[0] 的参数进行更新；

![DP 原理图1](https://pic3.zhimg.com/80/v2-1cee4e8fd9e2d4dce24b0aa0a47f8c86_1440w.webp)
![DP 原理图2](https://pic1.zhimg.com/80/v2-5c5b0d8e3d7d6653a9ebd47bac93090c_1440w.webp)

# 2. code implement
## pytorch 相关源码
```python
import torch.nn as nn
model = nn.DataParallel(model) # 只需要将原来单卡的 module 用 DP 改成多卡
class DataParallel(Module):
```

## train mode use pytorch DP
**运行 dp_hello.py**
```shell
python dp_hello.py
```
>>> output: Let's use 2 GPUs!

**运行 dp_demo.py**
```shell
python dp_demo.py

result:
>>> data shape:  torch.Size([64, 1, 28, 28])
>>>  =============x shape:  torch.Size([32, 1, 28, 28])
>>> =============x shape:  torch.Size([32, 1, 28, 28])
```

# 3. DP 的优缺点
- 负载不均衡：device[0] 负载大一些；
- 通信开销大；
- 单进程；
- Global Interpreter Lock (GIL)全局解释器锁，简单来说就是，一个 Python 进程只能利用一个 CPU kernel，<br>
  即单核多线程并发时，只能执行一个线程。考虑多核，多核多线程可能出现线程颠簸 (thrashing) 造成资源浪费，<br>
  所以 Python 想要利用多核最好是多进程。<br>

# 4. [references]
1. [pytorch 源码](https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/data_parallel.py)
2. [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html?highlight=data+parallel#torch.nn.DataParallel)
3. [代码参考链接](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel)
4. [DP 和 DDP](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/notes/cuda.html%3Fhighlight%3Dbuffer)
