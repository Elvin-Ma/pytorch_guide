# run ddp_demo.py
```shell
python ddp_demo.py
```

# run example.py
```shell
python -m torch.distributed.launch --nproc_per_node=2 --nnode=1 --node_rank=0 example.py --local_world_size=2

python /home/mtn/miniconda3/lib/python3.8/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=2 example.py --local_world_size=2
```

# important packages
```python
import torch.utils.data.distributed
import torch.distributed as dist
from torch.multiprocessing import Process

train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
if rank == 0:
        torch.save(net, "my_net.pth")


# 多进程进行处理
size = 3 # 启动3个进程
processes = []
for rank in range(size):
    p = Process(target=main, args=(rank,)) #将函数送到进程中
    p.start() # 启动进程
    processes.append(p) #加到进程 list

for p in processes:
    p.join()

# 上述写法可以写成 spawn 方式
# spawn格式：
def main():
    world_size= 3
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)
```

# important env var
```python
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
```

# args explain
- node: 物理节点，可以是一台机器，也可以是一个容器，节点内部可以有多个GPU；
- rank：用于表示进程的编号/序号（在一些结构图中rank指的是软节点，rank可以看成一个计算单位），每一个进程对应了一个rank的进程，整个分布式由许多rank完成；
- local_rank: rank是指在整个分布式任务中进程的序号；local_rank是指在一个node上进程的相对序号;
- nnodes: 物理节点数量；
- node_rank: 物理节点序号,用于设置主节点；
- nproc_per_node: 每个物理节点上进程的数量;
- work_size: 全局(一个分布式任务)中, rank的数量;
- group:进程组,一个分布式任务对应了一个进程组.只有用户需要创立多个进程组时才会用到group来管理,一般情况只有一个group;

# 多机分布式(一个进程占用一张卡):multi_machine_v1.py
- 1. dist.init_process_group里面的rank需要根据node以及GPU的数量计算；
- 2. world_size的大小=节点数 x GPU 数量。
- 3. ddp 里面的device_ids需要指定对应显卡。

```shell
python python demo.py --world_size=16 --node_rank=0 --master_addr="192.168.0.1" --master_port=22335
python python demo.py --world_size=16 --node_rank=1 --master_addr="192.168.0.1" --master_port=22335
```

# 多机分布式（torch.distributed.launch）
```shell
python -m torch.distributed.launch --nproc_per_node=8
           --nnodes=2 --node_rank=0 --master_addr="192.168.0.1"
           --master_port=12355 MNIST.py

# 启动两台机器
python -m torch.distributed.launch --nproc_per_node=8
           --nnodes=2 --node_rank=1 --master_addr="192.168.0.1"
           --master_port=12355 MNIST.py

python -m torch.distributed.launch --nproc_per_node=8
           --nnodes=1 --node_rank=0 --master_addr="192.168.0.1"
           --master_port=12355 MNIST.py
```
