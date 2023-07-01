import torch
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0') # net1 --> gpu0 上
        self.relu = torch.nn.ReLU() # 没有任何可学习参数所以不需要to
        self.net2 = torch.nn.Linear(10, 5).to('cuda:0') # net2 --> gpu1 上

    def forward(self, x):
        x = x.to('cuda:0') # h2d : gpu0 上；
        x = self.net1(x) # 运算
        x = self.relu(x) # relu操作 x --> gpu0 上
        x = x.to('cuda:0') # 把 x --> gpu1 上
        x = self.net2(x) # 在 GPU1 上执行此操作
        return x # x 在gpu1 --> cuda:1
    
if __name__ == "__main__":
    print("start to run model_parallel_demo.py !!!")
    model = ToyModel() # 实例化一个模型
    loss_fn = nn.MSELoss() # 损失函数定义
    optimizer = optim.SGD(model.parameters(), lr=0.001) # 定义优化器

    optimizer.zero_grad() # 梯度清0
    outputs = model(torch.randn(20, 10)) # forward 过程
    labels = torch.randn(20, 5).to('cuda:0') # label --> cuda:1 上
    loss = loss_fn(outputs, labels) # 计算损失
    loss.backward() # loss 的反向传播
    optimizer.step() # 更新权重
    print("run model_parallel_demo.py successfully !!!")
    

