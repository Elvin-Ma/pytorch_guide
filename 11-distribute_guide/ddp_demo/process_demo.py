import os
import torch
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms
from torch.multiprocessing import Process

# 主机地址 和 端口(端口) 设置
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

WORLD_SIZE = 1 # 一共启动多少个进程

def main(rank):
    dist.init_process_group("sccl", rank=rank, world_size= WORLD_SIZE) # 初始化进程组（后端选择，rank，world_size）
    torch.cuda.set_device(rank) #device 设置
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST("./data", train=True, transform=trans, target_transform=None, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set) # 同 DP 相同的接口
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set, batch_size=64, sampler=train_sampler)

    net = torchvision.models.resnet50(num_classes=10) # 适配mnist 数据集 --> class num: 10
    
    # monkey patch --> 猴子补丁
    net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False) # 修改模型: --> minst 是一个灰度图（channel = 1）
    net = net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank]) # 和 DP 类似了，将原来的model 转化为 distribute model
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        for i, data in enumerate(data_loader_train):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print("loss: {}".format(loss.item()))
            
    # 指定一个device 保存模型    
    if rank == 0:
        torch.save(net, "my_net.pth")

if __name__ == "__main__":
    size = 1
    processes = []
    # for 循环启动进程
    for rank in range(size):
        p = Process(target=main, args=(rank,))
        p.start()
        processes.append(p)

    # join 起来
    for p in processes:
        p.join()