import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# rank 表示当前节点的编号，从0开始，在分布式训练中，每个计算节点有一个唯一的rank编号；
# rank 编号用于区分不同的节点， rank 可以通过 torch.distributed.get_rank()来获取
# world_size 表示分布式训练中计算节点的总数，包括主节点和工作节点，可以通过torch.distributed.get_world_size()来获取；
# 通过rank 和world_size 这两个参数，不同的计算节点可以进行通信和协调，共同完成模型训练的任务；
def example(rank, world_size):
    # create default process group
    print("=========rank: {}, world_size: {}".format(rank, world_size))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank) #分发到不同的节点
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank]) #拿到不同节点的model
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    '''
    在进行分布式训练时，需要进行进程间通信来实现数据的交换和同步。为了使不同进程能够相互通信，需要指定一个进程作为主节点，其他进程则连接到主节点。
    os.environ["MASTER_ADDR"]和os.environ["MASTER_PORT"]就是用来指定主节点地址和端口的环境变量。
    '''
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
    print("run ddp_hello.py successfully !!!")
