# singe proc with multi gpu
import torchvision
from torchvision import transforms
import torch.distributed as dist
import torch.utils.data.distributed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="12355", type=str)
args = parser.parse_args()

def main(rank, world_size):
    # 一个节点就一个rank，节点的数量等于world_size
    dist.init_process_group("gloo",
                            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            rank=rank,
                            world_size=world_size)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST('~/DATA/', train=True,
                                          transform=trans, target_transform=None, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set,
                                                    batch_size=256,
                                                    sampler=train_sampler,
                                                    num_workers=16,
                                                    pin_memory=True)
    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    net = net.cuda()
    # net中不需要指定设备！
    net = torch.nn.parallel.DistributedDataParallel(net)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(1):
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


if __name__ == '__main__':
    main(args.rank, args.world_size)
