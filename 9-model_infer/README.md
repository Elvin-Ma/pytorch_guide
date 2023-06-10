# trainning --> infer 的过程：
- 训练和推理；
- 推理对性能要求更高；

# 推理的不同方式：
- model.eval() # pytorch eager mode 的推理方式；
- torch.jit.script() # 可训练，可推理，会保证不同分支的信息
- torch.jit.trace() # 需要给出模型的input，这个input追踪模型
- torch.onnx.export() # 跨框架的静态图模型，也需要给input --> 追踪模型

# 走不同的分支会带来 infer 时候的问题吗？
- input 的shape 是最重要的，这个会影响最终的分支选择；
- 很少说根据一个具体的tensor（activation） 的值来判断走哪个分支；
- 静态图的表达能力就是有一定的局限性，（trace 和 export）；
- 局限性的解决办法：我们算法人员，或者工程人员，手动实现分支；

# 推理（inference）
- 面向部署
- 加速，对性能要求特别高；
- 业界有很多种推理引擎 --> 就是加速我们深度学习模型的推理；
- TensorRT、MNN、NCNN TVM Onnxruntime都是比较成熟的推理硬气；
- 常见的优化策略：算子融合，常量折叠，剪枝、稀疏化、量化、内存优化、去除无效分支；

# traced_infer
```python
def traced_demo():
    model = MyModel()
    scripted_model = torch.jit.trace(model, torch.randn(1, 10))

    # 保存模型到文件
    scripted_model.save("traced_model.pt")

    # 重新加载模型
    loaded_model = torch.jit.load("traced_model.pt")

    # 重新运行模型
    input_data = torch.randn(1, 10)
    output_data = loaded_model(input_data)
    print("traced model output: ", output_data)
```

# onnx_infer
```python
def onnx_demo():
    # model = MyModel()
    # torch.onnx.export(model, torch.randn(4, 10), "onnx_model.onnx")
    
    input = torch.randn(4,10)
    # 加载模型并运行
    import onnxruntime as ort
    ort_session = ort.InferenceSession("onnx_model.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print("onnx run output: ", ort_outputs[0])
```      