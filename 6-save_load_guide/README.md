# save and load tensor

# 课后作业： ？？？
**一个模型通过torch.save(model, "model.pt")完整保存下来；**
**把这个模型直接给第三方；**
**第三方能直接运行这个模型吗？？？**
## 答案：no##
**这时候的模型是工作在eager mode 下的，模型的结构信息它没有保存下来，
eager mode：forward中的一个个的算子，走一步看一步。**

# 保存及加载整个模型
```python
torch.save(model, "mnist.pt")
x = torch.load('tensor.pt')
```

# 保存及加载模型参数
```python
torch.save(model.state_dict(), "mnist_para.pth")
param = torch.load("mnist_para.pth")
model.load_state_dict(param)
```
# 注意
- pt 和 pth 用谁都行
- state_dict : 只保存模型权重
- torch.save 和 torch.load 还可以保存 tensor 和 optimizer

# 加载到GPU
model = torch.load('mnist.pt', map_location=device)

# 保存和加载 ckpt
```python
torch.save(checkpoint, 'model.ckpt')
checkpoint = torch.load('model.ckpt') # load --> dict
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



