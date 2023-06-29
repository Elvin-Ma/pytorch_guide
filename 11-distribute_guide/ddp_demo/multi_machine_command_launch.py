import torch
import torchvision
import torch.utils.data.distributed
import argparse
import torch.distributed as dist
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)  # 增加local_rank
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

def main():
    dist.init_process_group("nccl", init_method='env://')    # init_method方式修改
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
    # DDP 输出方式修改：
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                                    output_device=args.local_rank)
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(1):
        for i, data in enumerate(data_loader_train):
            images, labels = data
            # 要将数据送入指定的对应的gpu中
            images.to(args.local_rank, non_blocking=True)
            labels.to(args.local_rank, non_blocking=True)
            opt.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print("loss: {}".format(loss.item()))


if __name__ == "__main__":
    main()