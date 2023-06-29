# DDP(Distribute DataParallel) Summary

# 1. 概念总结

## 1.1 DataParallel概念
- 当一张 GPU 可以存储一个模型时，可以采用数据并行得到更准确的梯度或者加速训练，
- 即每个 GPU 复制一份模型，将一批样本分为多份输入各个模型并行计算。
- 因为求导以及加和都是线性的，数据并行在数学上也有效。

## 1.2 DDP 概念及原理
- **DDP 也是数据并行，所以每张卡都有模型和输入。我们以多进程多线程为例，每起一个进程，该进程的 device[0] 都会从本地复制模型，如果该进程仍有多线程，就像 DP，模型会从 device[0] 复制到其他设备。**
- **DDP 通过 Reducer 来管理梯度同步。为了提高通讯效率， Reducer 会将梯度归到不同的桶里（按照模型参数的 reverse order， 因为反向传播需要符合这样的顺序），一次归约一个桶。其中桶的大小为参数 bucket_cap_mb 默认为 25，可根据需要调整。下图即为一个例子。可以看到每个进程里，模型参数都按照倒序放在桶里，每次归约一个桶。**
[桶的案例](https://user-images.githubusercontent.com/16999635/72401724-d296d880-371a-11ea-90ab-737f86543df9.png)
[pytorch 案例地址](https://pytorch.org/docs/stable/notes/ddp.html)
- **DDP 通过在构建时注册 autograd hook 进行梯度同步。反向传播时，当一个梯度计算好后，相应的 hook 会告诉 DDP 可以用来归约。当一个桶里的梯度都可以了，Reducer 就会启动异步 allreduce 去计算所有进程的平均值。allreduce 异步启动使得 DDP 可以边计算边通信，提高效率。当所有桶都可以了，Reducer 会等所有 allreduce 完成，然后将得到的梯度写到 param.grad。**

## 1.3 DDP 优势
- DDP 采用多进程，最推荐的做法是每张卡一个进程从而避免上一节所说单进程带来的影响；
- DDP 同样支持单进程多线程多卡操作，自然也支持多进程多线程；
- DP 的通信成本随着 GPU 数量线性增长，而 DDP 支持 Ring AllReduce，其通信成本是恒定的，与 GPU 数量无关；
- DDP 通过保证初始状态相同并且改变量也相同（指同步梯度） ，保证模型同步；
- 使用 Ring AllReduce 加速数据同步；

## 1.4 Ring AllReduce 原理
- [Ring AllReduce 1](https://picture.iczhiku.com/weixin/message1570798743118.html)
- [Ring AllReduce 2](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/#)

**scatter reduce**
- ![scatter reduce](https://pic3.zhimg.com/v2-4590aeb5fd981b1e6f926cc68605884a_b.webp)

**all gather**
- ![all gather](https://pic4.zhimg.com/80/v2-c9df34575d7d95ec87d85575d25d6f37_720w.webp)

# 2. 常用参数（ARGS）总结
- node: 物理节点，可以是一台机器，也可以是一个容器，节点内部可以有多个GPU；
- rank：用于表示进程的编号/序号（在一些结构图中rank指的是软节点，rank可以看成一个计算单位），每一个进程对应了一个rank的进程，整个分布式由许多rank完成；
- nproc_per_node: 每个物理节点上进程的数量;
- work_size: 全局(一个分布式任务)中, rank的数量;
- local_rank: rank是指在整个分布式任务中进程的序号；local_rank是指在一个node上进程的相对序号;
- nnodes: 物理节点数量；
- node_rank: 物理节点序号,用于设置主节点；
- group:进程组,一个分布式任务对应了一个进程组.只有用户需要创立多个进程组时才会用到group来管理,一般情况只有一个group;

# 3. 源码实现相关
```python
import torch.utils.data.distributed
import torch.distributed as dist
from torch.multiprocessing import Process
```
- 通信：因为 DDP 依赖 c10d 的 ProcessGroup 进行通信，所以开始前我们先要有个 ProcessGroup 实例。这步可以通过 torch.distributed.init_process_group 实现。
- 构建：我们先贴上 DDP 初始化的源码，最重要的是 _ddp_init_helper 这个函数，负责多线程时复制模型、将 parameters 分组、创建 reducer 以及为 SyncBN 做准备等。这部分代码看 comment 就能懂，我们会重点说一下 dist.Reducer，作为管理器，自然很重要了。

# 4. 代码实现

## 4.1 DDP 训练有哪些不同呢？（自己总结...）
- 启动的方式引入了一个多进程机制；
- 引入了几个环境变量；
- DataLoader多了一个sampler参数；
- 网络被一个DistributedDataParallel(net)又包裹了一层；
- ckpt的保存方式发生了变化。

## 4.2 单机多卡：自己控制process
```python
# set yourself WORLD_SIZE = 2
# bug exist，please checkout code to run successfully
python process_demo.py
```

## 4.3 单机多卡：用spawn 自动控制process
```shell
python spawn_demo.py
```
>>> =========rank: 1, world_size: 2
>>> =========rank: 0, world_size: 2
>>> run ddp_hello.py successfully !!!

## 4.4 单机多卡(也可以多机多卡)：用命令行启动
```shell
# use bin command
python -m torch.distributed.launch --nproc_per_node=2 --nnode=1 --node_rank=0 command_demo.py --local_world_size=2

# use python file
python /home/mtn/miniconda3/lib/python3.8/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=2 command_demo.py --local_world_size=2
```

## 4.5 多机分布式：每个进程占用一张卡
**注意事项*
- dist.init_process_group里面的rank需要根据node以及GPU的数量计算；
- world_size的大小=节点数 x GPU 数量。
- ddp 里面的device_ids需要指定对应显卡。

*代码实现*
```shell
#节点1
python python multi_machine_one_process_one_gpu.py --world_size=16 --node_rank=0 --master_addr="192.168.0.1" --master_port=22335
#节点2
python python multi_machine_one_process_one_gpu.py --world_size=16 --node_rank=1 --master_addr="192.168.0.1" --master_port=22335
```

## 4.6 多机分布式：单个进程占用多张卡
1. dist.init_process_group里面的rank等于节点编号；
2. world_size等于节点的总数量；
3. DDP不需要指定device。
```shell
 #节点1
python demo.py --world_size=2 --rank=0 --master_addr="192.168.0.1" --master_port=22335
 #节点2
python demo.py --world_size=2 --rank=2 --master_addr="192.168.0.1" --master_port=22335
```

## 4.7 多机分布式：用命令行启动
**用torch.distributed.launch启动，查看源码可以看出launch实际上主要完成的工作：**
1. 参数定义与传递。解析环境变量，并将变量传递到子进程中。
2. 起多进程。调用subprocess.Popen启动多进程。

**命令行要解决的问题：就是上述参数如何传入**
- 需要添加一个解析 local_rank的参数：parser.add_argument("--local_rank", type=int)
- dist初始化的方式 int_method取env：dist.init_process_group("gloo", init_method='env://')
- DDP的设备都需要指定local_rank：
  net = torch.nn.parallel.DistributedDataParallel(net,
        device_ids=[args.local_rank], output_device=args.local_rank)

**假设一共有两台机器（节点1和节点2），每个节点上有8张卡，节点1的IP地址为192.168.0.1**
**占用的端口12355（端口可以更换），启动的方式如下：**
```shell
 #节点1
python -m torch.distributed.launch --nproc_per_node=8
       --nnodes=2 --node_rank=0 --master_addr="192.168.0.1"
       --master_port=12355 MNIST.py
 #节点2
python -m torch.distributed.launch --nproc_per_node=8
       --nnodes=2 --node_rank=1 --master_addr="192.168.0.1"
       --master_port=12355 MNIST.py
```

**假如只启动一台机器时：**
```shell
#节点1
python -m torch.distributed.launch --nproc_per_node=8
       --nnodes=1 --node_rank=0 --master_addr="192.168.0.1"
       --master_port=12355 MNIST.py
```

# 5. 其它注意事项

## 5.1 多种后端的选择
**torch.distributed 支持 3 种后端，分别为 NCCL，Gloo，MPI**
*Gloo后端*
*- gloo 后端支持 CPU 和 GPU，其支持集体通信（collective Communication），并对其进行了优化。*
*- 由于 GPU 之间可以直接进行数据交换，而无需经过 CPU 和内存，因此，在 GPU 上使用 gloo 后端速度更快。*

**NCCL后端**
*NCCL 的全称为 Nvidia 聚合通信库（NVIDIA Collective Communications Library），*
*是一个可以实现多个 GPU、多个结点间聚合通信的库，在 PCIe、Nvlink、InfiniBand 上可以实现较高的通信速度。*
*对于每台主机均使用多进程的情况，使用 NCCL 可以获得最大化的性能。每个进程内，不许对其使用的 GPUs 具有独占权。*

**MPI后端**
*MPI 即消息传递接口（Message Passing Interface），是一个来自于高性能计算领域的标准的工具。*
*它支持点对点通信以及集体通信，并且是 torch.distributed 的 API 的灵感来源。*
*使用 MPI 后端的优势在于，在大型计算机集群上，MPI 应用广泛，且高度优化。*
*torch.distributed 对 MPI 并不提供原生支持。因此，要使用 MPI，必须从源码编译 Pytorch。*
*是否支持 GPU，视安装的 MPI 版本而定*

# 6. [references]
[参考资料0](https://pytorch.org/docs/stable/notes/ddp.html)<br>
[参考资料1](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)<br>
[参考资料2](https://pytorch.org/tutorials/intermediate/dist_tuto.html)<br>
[参考资料3](https://pytorch.org/docs/master/nn.html#dataparallel-layers-multi-gpu-distributed)<br>
[参考资料4](https://pytorch.org/docs/master/distributed.html)<br>
[参考资料5](https://zhuanlan.zhihu.com/p/72939003)<br>
[参考资料6-RingAllReduce](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)<br>
[参考资料7](https://mp.weixin.qq.com/s?__biz=MzI4ODg3NDY2NQ==&mid=2247484022&idx=2&sn=1ba381dc846257507c30599fa2b24c18&chksm=ec368bb0db4102a68378203f667e698e078d623c347c5499541435a3e77a5f7b504452e22b29&token=1461052041&lang=zh_CN&scene=21#wechat_redirect)<br>
[参考资料8](https://mp.weixin.qq.com/s?__biz=MzI4ODg3NDY2NQ==&mid=2247483977&idx=1&sn=01780e110a09d8b16c9e15dd2ebe1613&chksm=ec368b8fdb41029971242dd982bcdf805a8527a71c3c4d26dedc6486ac025e554c1758da9f0b&token=1618065730&lang=zh_CN&scene=21#wechat_redirect)<br>
[参考资料9](https://zhuanlan.zhihu.com/p/343951042)<br>
[参考资料10](https://zhuanlan.zhihu.com/p/358974461?utm_id=0)<br>
[参考资料11](https://www.cnblogs.com/rossiXYZ/p/15142807.html)<br>
[参考资料12](https://zhuanlan.zhihu.com/p/76638962)<br>
[参考资料13-torch.distribute](https://pytorch.org/docs/stable/distributed.html?highlight=torch+distributed#module-torch.distributed)<br>
