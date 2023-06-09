import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=int)
parser.add_argument("--node_rank", type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="12355", type=str)
args = parser.parse_args()


def example(local_rank, node_rank, local_size, world_size):
    # 初始化
    rank = local_rank + node_rank * local_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl",
                            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            rank=rank,
                            world_size=world_size)
    # 创建模型
    model = nn.Linear(10, 10).to(local_rank)
    # 放入DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    # 进行前向后向计算
    for i in range(1000):
        outputs = ddp_model(torch.randn(20, 10).to(local_rank))
        labels = torch.randn(20, 10).to(local_rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()


def main():
    local_size = torch.cuda.device_count()
    print("local_size: %s" % local_size)
    mp.spawn(example,
        args=(args.node_rank, local_size, args.world_size,),
        nprocs=local_size,
        join=True)


if __name__=="__main__":
    main()
