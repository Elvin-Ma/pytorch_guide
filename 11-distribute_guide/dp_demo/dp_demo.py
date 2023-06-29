import torch
import torch.nn as nn
import torch.optim as optim
# import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义分布式环境
# dist.init_process_group(backend="gloo", rank=0, world_size=2)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

# 加载数据集
train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=2, rank=0)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, sampler=train_sampler)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("=============x shape: ", x.shape)
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

# 将模型复制到多个GPU上
model = nn.DataParallel(model).to(device)

# 定义损失函数和优化器
content = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10):
    # train_sampler.set_epoch(epoch)
    for data, target in train_loader:
        print("data shape: ", data.shape)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = content(output, target)
        loss.backward()
        optimizer.step()

    print("Epoch {} completed".format(epoch))