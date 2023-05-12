# save and load tensor
```python
import torch

x = torch.rand(3, 4)
torch.save(x, 'tensor.pt')

x = torch.load('tensor.pt')
print(x)

x = torch.rand(3, 4)
y = torch.ones(2, 2)
z = torch.zeros(1, 5)

torch.save({'x': x, 'y': y, 'z': z}, 'tensors.pt')

data = torch.load('tensors.pt')
x_new = data['x']
y_new = data['y']
z_new = data['z']
```
# save and load model(structure and param)
```python
import torch
import torchvision.models as models

# 创建一个预训练的 resnet18 模型
model = models.resnet18(pretrained=True)

# 保存整个模型
torch.save(model, 'resnet18_full.pth')

#加载整个模型
model = torch.load('resnet18_full.pth')
```

# 仅保存模型参数

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)

# 保存模型参数
torch.save(model.state_dict(), 'resnet18_params.pth')

state_dict = torch.load('resnet18_params.pth')

# 必须先创建一个相同结构的模型
model = models.resnet18()

# 加载参数
model.load_state_dict(state_dict)

```

# 加载数据到GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)

# 如果使用 CPU
model = torch.load('model.pth', map_location=torch.device('cpu'))
```

# save and load ckpt
**.ckpt 文件通常包含整个训练状态，包括模型参数、优化器状态、学习率、训练轮数和其他训练信息等**
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # ...
    # 训练模型
    # ...
    if epoch % save_freq == 0:
        # 保存 checkpoint 文件
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            # 可以添加其他训练信息
        }
        torch.save(checkpoint, 'model.ckpt')


checkpoint = torch.load('model.ckpt')
model = MyModel()
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

# jit trace 保存
```python
import torch
import torchvision

model = torchvision.models.resnet18()
example_input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example_input)

traced_script_module.save("model.pt")

loaded_script_module = torch.jit.load("model.pt")
```

# onnx 格式保存

```python
import torch
import torchvision

model = torchvision.models.resnet18()
input_tensor = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, input_tensor, "model.onnx")
```

# what is state_dict
**reference: what_is_state_dict.py**




