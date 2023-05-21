# 分布式：
- DP(data parallism) : 单机多卡
- DDP(Distribute data parallism) ： 可以单机多卡，也可以 多机多卡
- 数据并行 --> 拆分数据 --> 在batch 维度上拆分
**除了数据并行，模型并行（）**

# DP（是一种数据并行方式）
- 它将模型的参数复制到多个GPU上，每个GPU都使用相同的数据集进行训练；
- 每个GPU上的模型参数独立更新，然后将所有GPU上的参数进行同步。
- DP的优点是易于使用和实现，并且适用于大多数模型。
- 速度比DDP 慢

# DDP
- DDP是一种更高级的分布式训练方式，它可以更好地处理大型模型和小批量训练。
- 在DDP中，每个GPU都有一份完整的模型副本，并独立计算梯度。
- 然后，所有GPU上的梯度会被收集和聚合，以更新模型参数。
- 相比DP，DDP的优点是更灵活和高效，可以更好地处理大型模型和小批量训练的情况。
- 此外，DDP还支持更多的分布式设置，如多节点训练和混合精度训练。
- 速度更快
- 核心点：通过 rank 和 worl_size 来控制我们的机器的分发，和代码的整理

# DDP 环境变量设置
- python -m torch.distributed.launch train.py --args 命令时会被自动设置。
- 具体来说python -m torch.distributed.launch会为每个进程设置以下环境变量：
- MASTER_ADDR：主节点的IP地址或主机名。
- MASTER_PORT：主节点监听的端口号。
- WORLD_SIZE：总共使用的进程数，包括主节点和工作节点。
- RANK：当前进程的排名，从0开始计数。

# DP example
```python
1. 定义分布式环境
dist.init_process_group(backend="gloo", rank=0, world_size=2)

2. 数据准备
train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=2, rank=0)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, sampler=train_sampler)

3. 模型准备
model = nn.DataParallel(model)
```

## 参数
- backend：指定分布式训练使用的后端，PyTorch 支持多种后端，如 gloo、nccl、mpi 等，不同的后端有不同的优缺点，可以根据具体情况选择合适的后端。
- rank：指定当前进程的编号，每个进程都有一个唯一的编号，用于在分布式环境中进行通信和协调工作。一般来说，rank 的取值范围是从 0 到 world_size-1。
- world_size：指定参与分布式训练的进程数量，也就是分布式环境中的进程总数。world_size 的取值应该等于参与训练的 GPU 数量或节点数量。

## 三种后端
- gloo：是 PyTorch 默认的后端，它是一个基于消息传递的后端，支持在 CPU 和 GPU 上运行，适用于多节点和多 GPU 的分布式训练。gloo 的优点是易于使用和部署，支持大规模训练，缺点是在某些特殊情况下性能可能不如其他后端。
- nccl：是 NVIDIA Collective Communications Library 的缩写，是 NVIDIA 推出的用于在 GPU 上进行高效通信的库。nccl 后端相比于其他后端在单节点多 GPU 的情况下能够提供更好的性能，但不支持多节点训练。
- mpi：是 Message Passing Interface 的缩写，是一种用于多计算机之间通信的标准。mpi 后端支持多节点训练，可以在不同的计算机之间进行通信，适用于大规模训练，但需要安装并配置额外的软件包，使用起来相对复杂。

## 参数解释
MASTER_ADDR：主节点的IP地址或主机名。
MASTER_PORT：主节点监听的端口号。
WORLD_SIZE：总共使用的进程数，包括主节点和工作节点。
RANK：当前进程的排名，从0开始计数。
这些环境变量在使用python -m torch.distributed.launch命令时会被自动设置。具体来说，python -m torch.distributed.launch会为每个进程设置以下环境变量：

LOCAL_RANK：当前进程在本节点中的排名，从0开始计数。
NODE_RANK：当前节点在所有节点中的排名，从0开始计数。
WORLD_SIZE：总共使用的进程数，包括主节点和工作节点。
RANK：当前进程的排名，从0开始计数。
MASTER_ADDR：主节点的IP地址或主机名。
MASTER_PORT：主节点监听的端口号。

