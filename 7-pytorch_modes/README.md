# 静态图和动态图
- tensorflow 刚开始是静态图 
- pytorch 刚开始是动态图；
- tensorflow到后来也支持了动态图；
- pytorch也是支持静态图的；
- 动态图不利于部署，尤其在边缘测部署，性能很差，优化手段有限；
- 静态图：保存了完整的图的结构信息，可以有很多的优化手段：
  eg：常量折叠、量化、裁剪、算子融合、缓存优化；

# 
# pytorch 几种mode
## eager mode ： 动态图的方式

## eager jit mode：
1. torch.jit.script: 可以训练的，基本上保持了 eager mode 的所有功能（条件判断、动态shape等），
2. torch.jit.trace ： （部署的时候） 追踪eager mode 的前向图，导出一个pt文件
3. 不局限于python，c++ 直接读取，并完成部署；
4. libtorch： pytroch的 C++ 部分，
5. trace：适用于没有控制流层模型（if for while);
6. trace mode 需要多给个模型的输入，因为它真的要把这个模型跑一遍；
6. script：允许使用控制流

## dynamo （pytorch 2.0新功能）
[dynomo 官方blog](https://pytorch.org/get-started/pytorch-2.0/)
1. 一种新的图编译模式；
2. pytorch 版本：1.9.0 1.10.0 1.11.2 1.12.0 1.13.0 --> 2.0
3. 2023年最新发布

## onnx 中间格式：
1. google 的protobuffer 文件编码格式；
2. onnx : 跨平台、跨框架，
3. 业内都认可的一种形式；
4. 为推理而生的；
5. 把训练好的模型保存成onnx 格式，再送入到各式各样的推理引擎中完成推理；

# torch.jit.script
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# 定义数据集
class MyDataset(data.Dataset):
    def __init__(self):
        self.data = torch.randn(100, 10)
        self.target = torch.randint(0, 2, (100,))
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return len(self.data)

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# 实例化数据集dataset = MyDataset()

# 实例化模型
model = MyModel()

# 将模型转换为script模式
scripted_model = torch.jit.script(model)

# 定义优化器和损失函数
optimizer = optim.SGD(scripted_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 定义训练函数
def train(model, dataset, optimizer, criterion, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in dataset:
            # 将数据和目标转换为Tensor类型
            data = torch.tensor(data)
            target = torch.tensor(target)

            # 前向传播
            output = model(data)
            loss = criterion(output, target)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和正确率
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # 打印训练信息
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch+1, epochs, running_loss/len(dataset), 100*correct/total))

# 实例化数据集和数据加载器
dataset = MyDataset()
dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)

# 调用训练函数
train(scripted_model, dataloader, optimizer, criterion, epochs=10)
```

#torch.jit.trace
```python
# 将模型转换为Torch脚本
scripted_model = torch.jit.trace(trained_model, torch.randn(1, 10))

# 保存模型到文件
scripted_model.save("my_model.pt")

# 重新加载模型
loaded_model = torch.jit.load("my_model.pt")

# 重新运行模型
input_data = torch.randn(1, 10)
output_data = loaded_model(input_data)
```
# onnx 格式的加载和run
```python
# 将模型转换为ONNX格式
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "my_model.onnx")

# 加载模型并运行
ort_session = ort.InferenceSession("my_model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outputs = ort_session.run(None, ort_inputs)
```

# torch dynamo 简介

```python
 import torch
      
 def train(model, dataloader):
   model = torch.compile(model)
   for batch in dataloader:
     run_epoch(model, batch)

 def infer(model, input):
   model = torch.compile(model)
   return model(\*\*input)
```

方式2：
```python
    @optimize('inductor')
    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

