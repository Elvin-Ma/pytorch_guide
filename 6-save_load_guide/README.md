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

# 加载到GPU
```python
model = torch.load('mnist.pt', map_location=device)
```

# 保存和加载 ckpt
```python
torch.save(checkpoint, 'model.ckpt')
checkpoint = torch.load('model.ckpt') # load --> dict
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```
# 注意事项
- pt 和 pth 、ckpt 用谁都行
- state_dict : 只保存模型权重（parameters）和 buffer：tensor的信息
- torch.save 和 torch.load 还可以保存 tensor、optimizer、 lr_schedule、 epoch 等
- 使用state_dict 这种保存方式的话，加载的时候要用 load_state_dict 方法来加载；
- 进一步，要使用load_state_dict 必须要实现建立一个对象；
- torch.save 保存的是什么，torch.load 加载出来就是什么；
- torch.save 保存模型--> 并没有保存具体的模型结构信息, 只是保存了模块和parameters/tensor等信息;
- 可以直接load模型到我们的 cpu上,通过设置map_location 参数
- 可以通过torch.jit 这种方式来保存成静态图模型（保存了模型的结构信息），重新加载不依赖与原来的图；

# 静态图和动态图
- tensorflow 刚开始是静态图 
- pytorch 刚开始是动态图；
- tensorflow到后来也支持了动态图；
- pytorch也是支持静态图的；
- 动态图不利于部署，尤其在边缘测部署，性能很差，优化手段有限；
- 静态图：保存了完整的图的结构信息，可以有很多的优化手段：
  eg：常量折叠、量化、裁剪、算子融合、缓存优化；

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



